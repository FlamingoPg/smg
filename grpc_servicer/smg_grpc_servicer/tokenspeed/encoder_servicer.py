"""TokenSpeed EPD encode servicer.

Receives ``Encode`` RPCs from the gateway and forwards them to an encoder-only
encode worker (the engine's ``run_encode_loop``) over the same AsyncLLM
scheduler-input channel the LM uses. The encode worker runs the requested
multimodal tower and ships the resulting embeddings to prefill workers; this
servicer only acks (the embeddings never flow back through the gateway).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

import grpc
from smg_grpc_proto.generated import (
    common_pb2,
    tokenspeed_encoder_pb2,
    tokenspeed_encoder_pb2_grpc,
)

from smg_grpc_servicer.mm_rdma import RdmaPixelPuller
from smg_grpc_servicer.tokenspeed.rdma_config import rdma_pixel_config_from_server_args
from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _lazy_encode_request():
    from tokenspeed.runtime.pd.epd.encode_worker import EncodeRequest

    return EncodeRequest


def _lazy_mm_item():
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    return Modality, MultimodalDataItem


class TokenSpeedEncoderServicer(tokenspeed_encoder_pb2_grpc.TokenSpeedEncoderServicer):
    """gRPC servicer fronting the engine's encode worker over AsyncLLM."""

    def __init__(
        self,
        async_llm: AsyncLLM,
        server_args: ServerArgs,
        scheduler_info: dict,
        health_servicer=None,
    ):
        self.async_llm = async_llm
        # Ship encoder inputs to the scheduler process as POSIX-SHM
        # handles instead of pickling the raw tensor over ZMQ (the dominant
        # per-item ingest cost). Explicit ServerArgs switches support the
        # inline ZMQ and on-loop fallbacks without process-global state.
        self._pixel_shm = server_args.epd_pixel_shm
        self._ingest_offloop = server_args.epd_ingest_offloop
        self._unlink_mm_shm_after_read = server_args.unlink_mm_shm_after_read

        # The encode worker hosts its OWN Mooncake bootstrap server (it is the
        # data source); prefill workers discover it at (this host, the
        # disaggregation bootstrap port). Both live on this node.
        from tokenspeed.runtime.utils.network import get_local_ip_by_remote

        self._bootstrap_host = get_local_ip_by_remote()
        self._bootstrap_port = server_args.disaggregation_bootstrap_port

        self._rdma_pixel_puller = RdmaPixelPuller(
            agent_name=f"smg-encode-{self._bootstrap_host}-{self._bootstrap_port}",
            log_prefix="EPD RDMA",
            config=rdma_pixel_config_from_server_args(server_args),
        )

        logger.info("TokenSpeedEncoderServicer initialized")

    def _items_from_proto(self, mm_inputs, bootstrap_room: int = 0):
        """Reconstruct the engine MultimodalDataItem(s) for the encode worker.

        Unlike the prefill leg, the encode worker NEEDS each item's encoder_input
        (it runs the tower). Proto placeholders carry the output token spans the
        executor uses to split the packed tower output back into items.
        """
        Modality, MultimodalDataItem = _lazy_mm_item()
        model_dtype = getattr(self.async_llm.model_config, "dtype", None)

        # mm_inputs is itemized (one MultimodalItem per media item, each owning its
        # encoder_input + model_specific_tensors). The gateway sends one item per
        # Encode RPC keyed by bootstrap_room, but iterate generally.
        items = []
        for item_proto in mm_inputs.items:
            if item_proto.modality == common_pb2.MODALITY_UNSPECIFIED:
                item_modality = Modality.IMAGE
            else:
                item_modality = TokenSpeedSchedulerServicer._modality_from_proto(
                    item_proto.modality
                )

            if not item_proto.placeholders:
                raise ValueError("encode MultimodalItem carried no placeholders")
            if any(p.length <= 0 for p in item_proto.placeholders):
                raise ValueError("encode MultimodalItem.placeholders.length must be > 0")
            offsets = [(p.offset, p.offset + p.length - 1) for p in item_proto.placeholders]

            model_specific = {
                name: TokenSpeedSchedulerServicer._tensor_from_proto(
                    t,
                    unlink_shm_after_read=self._unlink_mm_shm_after_read,
                )
                for name, t in item_proto.model_specific_tensors.items()
            }

            # The feature's CROSS-PROCESS representation is decided here, once, for
            # both payload arms: a plain CPU tensor by default, or (when enabled) a
            # POSIX-SHM handle so the ZMQ hop to the scheduler pickles ~KB instead of
            # the full encoder tensor. The content hash is computed on the real bytes
            # before the swap and pre-set on the item.
            td = item_proto.encoder_input
            if td.WhichOneof("payload") == "remote":
                feature = self._rdma_pixel_puller.feature_from_remote(
                    td,
                    explicit_room=bootstrap_room,
                    cast_to=model_dtype,
                )
            else:
                feature = TokenSpeedSchedulerServicer._tensor_from_proto(
                    td,
                    cast_to=model_dtype,
                    unlink_shm_after_read=self._unlink_mm_shm_after_read,
                )

            # The SHM path publishes the feature to scheduler SHM so the ZMQ hop
            # pickles ~KB instead of the full encoder tensor; the content hash is
            # taken on the real bytes first.
            feat_hash = None
            if self._pixel_shm:
                from tokenspeed.runtime.multimodal.hash import hash_feature
                from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle

                feat_hash = hash_feature(feature)
                feature = ShmTensorHandle.publish(feature)

            item = MultimodalDataItem(
                modality=item_modality,
                hash=feat_hash,
                feature=feature,
                model_specific_data=model_specific,
                offsets=offsets,
            )
            item.set_pad_value()
            items.append(item)
        return items

    async def Encode(self, request, context):
        # The engine EncodeRequest currently carries one bootstrap_room scalar,
        # so the gateway must send exactly one multimodal item per Encode RPC.
        # Reject malformed/grouped requests instead of silently using room 0 or
        # applying the first room to every item.
        if not request.HasField("mm_inputs"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "EncodeRequest.mm_inputs is required",
            )
        if len(request.mm_inputs.items) != 1 or len(request.items) != 1:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "EncodeRequest must contain exactly one mm item and one room assignment",
            )

        bootstrap_room = request.items[0].bootstrap_room

        if self._ingest_offloop:
            # Per-item ingest (proto->tensor + pickle) BLOCKS the lone asyncio
            # event loop, so grpc.aio cannot deliver the next Encode message until
            # the previous one is fully ingested -- a per-worker serial encoder-input lane.
            # Split it: parse + pickle on a worker thread (overlapping across
            # items; the GIL is released in the tensor copy/cast), then the
            # cheap zmq send back ON the loop -- send_to_scheduler is a
            # zmq.asyncio socket whose send() needs the running loop (and this
            # keeps it single-writer).
            try:
                payload = await asyncio.to_thread(self._parse_and_pickle, request, bootstrap_room)
                await self.async_llm.engine_core_client.send_to_scheduler.send(payload)
            except ValueError as e:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            except Exception as e:  # noqa: BLE001
                logger.exception("TokenSpeed encode ingest failed")
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
        else:
            try:
                self._ingest(request, bootstrap_room)
            except ValueError as e:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            except Exception as e:  # noqa: BLE001
                logger.exception("TokenSpeed encode ingest failed")
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
        return tokenspeed_encoder_pb2.EncodeResponse(accepted=True)

    def _build_encode_request(self, request, bootstrap_room):
        """Proto -> engine EncodeRequest (the expensive per-item parse)."""
        items = self._items_from_proto(request.mm_inputs, bootstrap_room)

        EncodeRequest = _lazy_encode_request()
        return EncodeRequest(
            request_id=request.request_id or uuid.uuid4().hex,
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
            bootstrap_room=bootstrap_room,
            items=items,
        )

    def _parse_and_pickle(self, request, bootstrap_room) -> bytes:
        """Worker-thread half of the off-loop ingest: parse + pickle. The
        pickled bytes match what send_pyobj would produce, so the scheduler's
        recv_pyobj is unchanged."""
        import pickle

        encode_request = self._build_encode_request(request, bootstrap_room)
        return pickle.dumps(encode_request, protocol=pickle.DEFAULT_PROTOCOL)

    def _ingest(self, request, bootstrap_room) -> None:
        """Legacy on-loop ingest (parse + submit on the event loop)."""
        encode_request = self._build_encode_request(request, bootstrap_room)
        self.async_llm.submit_encode(encode_request)

    async def shutdown(self) -> None:
        """No persistent per-request state to drain (encode is fire-and-forget)."""
        return None
