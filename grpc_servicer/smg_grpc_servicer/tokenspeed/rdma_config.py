"""Map TokenSpeed's explicit server settings to the shared RDMA transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smg_grpc_servicer.mm_rdma import RdmaPixelPullerConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


def rdma_pixel_config_from_server_args(server_args: ServerArgs) -> RdmaPixelPullerConfig:
    """Build one immutable RDMA config from the parsed TokenSpeed arguments."""
    return RdmaPixelPullerConfig(
        enabled=server_args.mm_pixel_rdma,
        slot_bytes=server_args.mm_rdma_slot_bytes,
        landing_slots=server_args.mm_rdma_landing_slots,
        send_metadata=server_args.mm_rdma_send_metadata,
        landing_wait_seconds=server_args.mm_rdma_landing_wait_seconds,
        read_timeout_seconds=server_args.mm_rdma_read_timeout_seconds,
    )
