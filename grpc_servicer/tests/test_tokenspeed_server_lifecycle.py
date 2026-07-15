"""CPU-only tests for TokenSpeed gRPC configuration and engine lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer import mm_rdma as mm_rdma_module
from smg_grpc_servicer.tokenspeed import encoder_servicer as encoder_module
from smg_grpc_servicer.tokenspeed import rdma_config as rdma_config_module
from smg_grpc_servicer.tokenspeed import scheduler_launcher
from smg_grpc_servicer.tokenspeed import server as server_module
from smg_grpc_servicer.tokenspeed import servicer as servicer_module

_LEGACY_PRODUCT_ENV_VARS = (
    "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES",
    "TOKENSPEED_SKIP_GRPC_WARMUP",
    "TOKENSPEED_HEALTH_CHECK_TIMEOUT",
    "TOKENSPEED_LOG_MM_TENSOR_DATA",
    "TOKENSPEED_LOG_MM_TIMING",
    "TOKENSPEED_UNLINK_MM_SHM_AFTER_READ",
    "EPD_PIXEL_SHM",
    "EPD_INGEST_OFFLOOP",
    "SMG_MM_PIXEL_RDMA",
    "SMG_RDMA_SLOT_BYTES",
    "SMG_RDMA_LANDING_SLOTS",
    "SMG_RDMA_SEND_MD",
    "SMG_RDMA_LANDING_WAIT_S",
    "SMG_RDMA_READ_TIMEOUT_S",
)


def _server_args(**overrides):
    values = {
        "host": "127.0.0.1",
        "port": 50051,
        "grpc_max_message_bytes": 256 * 1024 * 1024,
        "skip_grpc_warmup": False,
        "health_check_timeout": 20.0,
        "log_mm_tensor_data": False,
        "enable_log_mm_timing": False,
        "unlink_mm_shm_after_read": True,
        "epd_pixel_shm": True,
        "epd_ingest_offloop": True,
        "mm_pixel_rdma": False,
        "mm_rdma_slot_bytes": 32 * 1024 * 1024,
        "mm_rdma_landing_slots": 64,
        "mm_rdma_send_metadata": None,
        "mm_rdma_landing_wait_seconds": 120.0,
        "mm_rdma_read_timeout_seconds": 60.0,
        "disaggregation_mode": "null",
        "disaggregation_bootstrap_port": 8998,
        "kv_events_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeAsyncLLM:
    def __init__(self):
        self.auto_create_calls = []
        self.close_calls = 0
        self.gracefully_exit = False
        self.rid_to_state = {}
        self.model_config = SimpleNamespace(dtype=None)

    def auto_create_handle_loop(self, *, manage_signals: bool = True) -> None:
        self.auto_create_calls.append(manage_signals)

    async def close(self) -> None:
        self.close_calls += 1


def test_launcher_starts_async_llm_without_signal_ownership(monkeypatch):
    async_llm = _FakeAsyncLLM()
    scheduler_info = {"max_total_num_tokens": 1024, "max_req_input_len": 512}
    monkeypatch.setattr(
        scheduler_launcher,
        "_launch_subprocesses",
        lambda **_kwargs: (async_llm, object(), scheduler_info),
    )

    launched, info = scheduler_launcher.launch_engine(_server_args())

    assert launched is async_llm
    assert info is scheduler_info
    assert async_llm.auto_create_calls == [False]


def test_launcher_rejects_non_serving_rank(monkeypatch):
    monkeypatch.setattr(
        scheduler_launcher,
        "_launch_subprocesses",
        lambda **_kwargs: (None, object(), {}),
    )

    with pytest.raises(RuntimeError, match="only rank 0"):
        scheduler_launcher.launch_engine(_server_args())


@pytest.fixture
def no_rdma(monkeypatch):
    monkeypatch.setattr(servicer_module, "RdmaPixelPuller", lambda **_kwargs: object())
    monkeypatch.setattr(encoder_module, "RdmaPixelPuller", lambda **_kwargs: object())


def _scheduler_servicer(async_llm, no_rdma, **arg_overrides):
    return servicer_module.TokenSpeedSchedulerServicer(
        async_llm=async_llm,
        server_args=_server_args(**arg_overrides),
        scheduler_info={},
    )


@pytest.mark.asyncio
async def test_shutdown_drains_before_async_llm_close(no_rdma):
    events = []

    class _Health:
        def set_not_serving(self):
            events.append("not_serving")

    class _AsyncLLM(_FakeAsyncLLM):
        async def close(self):
            events.append("close")
            assert not self.rid_to_state
            await super().close()

    async_llm = _AsyncLLM()
    async_llm.rid_to_state["request"] = object()
    scheduler = servicer_module.TokenSpeedSchedulerServicer(
        async_llm=async_llm,
        server_args=_server_args(),
        scheduler_info={},
        health_servicer=_Health(),
    )

    async def finish_request():
        while not async_llm.gracefully_exit:
            await asyncio.sleep(0)
        events.append("drained")
        async_llm.rid_to_state.clear()

    finisher = asyncio.create_task(finish_request())
    await scheduler.shutdown(drain_timeout_secs=1.0)
    await finisher

    assert events == ["not_serving", "drained", "close"]
    assert async_llm.close_calls == 1


@pytest.mark.asyncio
async def test_shutdown_timeout_still_closes_owned_engine(no_rdma):
    async_llm = _FakeAsyncLLM()
    async_llm.rid_to_state["stuck"] = object()
    scheduler = _scheduler_servicer(async_llm, no_rdma)

    await scheduler.shutdown(drain_timeout_secs=0.0)

    assert async_llm.gracefully_exit is True
    assert async_llm.close_calls == 1


def test_servicers_take_explicit_server_args(no_rdma, monkeypatch):
    args = _server_args(
        health_check_timeout=7.5,
        log_mm_tensor_data=True,
        enable_log_mm_timing=True,
        unlink_mm_shm_after_read=False,
        epd_pixel_shm=False,
        epd_ingest_offloop=False,
    )
    async_llm = _FakeAsyncLLM()
    scheduler = servicer_module.TokenSpeedSchedulerServicer(
        async_llm=async_llm,
        server_args=args,
        scheduler_info={},
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.utils.network.get_local_ip_by_remote",
        lambda: "127.0.0.1",
    )
    encoder = encoder_module.TokenSpeedEncoderServicer(
        async_llm=async_llm,
        server_args=args,
        scheduler_info={},
    )

    assert scheduler._health_check_timeout == 7.5
    assert scheduler._log_mm_tensor_data is True
    assert scheduler._log_mm_timing is True
    assert scheduler._unlink_mm_shm_after_read is False
    assert encoder._pixel_shm is False
    assert encoder._ingest_offloop is False
    assert encoder._unlink_mm_shm_after_read is False
    assert async_llm.auto_create_calls == []


def test_rdma_config_defaults_match_previous_runtime_behavior():
    config = mm_rdma_module.RdmaPixelPullerConfig()

    assert config.enabled is False
    assert config.slot_bytes == 32 * 1024 * 1024
    assert config.landing_slots == 64
    assert config.send_metadata is None
    assert config.landing_wait_seconds == 120.0
    assert config.read_timeout_seconds == 60.0


def test_rdma_config_maps_all_server_args_without_environment_state():
    args = _server_args(
        mm_pixel_rdma=True,
        mm_rdma_slot_bytes=4096,
        mm_rdma_landing_slots=3,
        mm_rdma_send_metadata=False,
        mm_rdma_landing_wait_seconds=4.5,
        mm_rdma_read_timeout_seconds=6.5,
    )

    assert rdma_config_module.rdma_pixel_config_from_server_args(args) == (
        mm_rdma_module.RdmaPixelPullerConfig(
            enabled=True,
            slot_bytes=4096,
            landing_slots=3,
            send_metadata=False,
            landing_wait_seconds=4.5,
            read_timeout_seconds=6.5,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"enabled": 1}, TypeError),
        ({"slot_bytes": 0}, ValueError),
        ({"slot_bytes": 1.5}, TypeError),
        ({"landing_slots": 0}, ValueError),
        ({"landing_slots": True}, TypeError),
        ({"send_metadata": "auto"}, TypeError),
        ({"landing_wait_seconds": 0}, ValueError),
        ({"landing_wait_seconds": float("inf")}, ValueError),
        ({"read_timeout_seconds": -1}, ValueError),
        ({"read_timeout_seconds": "60"}, TypeError),
    ],
)
def test_rdma_config_rejects_invalid_values(overrides, error):
    with pytest.raises(error):
        mm_rdma_module.RdmaPixelPullerConfig(**overrides)


@pytest.mark.parametrize(
    ("configured", "is_local", "expected_send"),
    [
        (None, True, True),
        (None, False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_rdma_metadata_policy_is_typed_and_explicit(
    monkeypatch,
    configured,
    is_local,
    expected_send,
):
    class _Agent:
        def __init__(self):
            self.sent = False

        def fetch_remote_metadata(self, *_args):
            return None

        def send_local_metadata(self, *_args):
            self.sent = True

        def check_remote_metadata(self, *_args):
            return True

    puller = mm_rdma_module.RdmaPixelPuller(
        agent_name="test",
        log_prefix="test",
        config=mm_rdma_module.RdmaPixelPullerConfig(send_metadata=configured),
    )
    agent = _Agent()
    puller._nixl_agent = agent
    monkeypatch.setattr(mm_rdma_module, "_ip_is_local", lambda _ip: is_local)

    puller._ensure_remote_ready("192.0.2.1", 1234, object(), room=1)

    assert agent.sent is expected_send


@pytest.mark.parametrize("value", [0, -1, (1 << 31)])
def test_grpc_message_limit_rejects_values_outside_grpc_range(value):
    with pytest.raises(ValueError, match="grpc_max_message_bytes"):
        server_module._grpc_max_message_bytes(_server_args(grpc_max_message_bytes=value))


def test_grpc_message_limit_comes_from_server_args():
    assert (
        server_module._grpc_max_message_bytes(_server_args(grpc_max_message_bytes=123456)) == 123456
    )


def test_explicit_skip_warmup_marks_server_ready_without_network():
    class _Health:
        serving = False

        def set_serving(self):
            self.serving = True

    health = _Health()
    server_module._wait_and_warmup(
        _server_args(skip_grpc_warmup=True),
        health,
    )
    assert health.serving is True


@pytest.mark.parametrize("unlink", [False, True])
def test_shm_unlink_policy_is_an_explicit_argument(monkeypatch, unlink):
    calls = []
    monkeypatch.setattr(servicer_module.os, "open", lambda *_args: 10)
    monkeypatch.setattr(servicer_module.os, "pread", lambda *_args: b"data")
    monkeypatch.setattr(servicer_module.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(servicer_module.os, "unlink", lambda path: calls.append(("unlink", path)))
    handle = common_pb2.ShmHandle(name="smg-test", offset=0, nbytes=4)

    assert (
        servicer_module.TokenSpeedSchedulerServicer._tensor_payload_bytes_from_shm(
            handle,
            unlink_shm_after_read=unlink,
        )
        == b"data"
    )
    assert calls[0] == ("close", 10)
    assert any(call[0] == "unlink" for call in calls) is unlink


def test_tokenspeed_adapter_has_no_legacy_product_environment_reads():
    package_dir = Path(servicer_module.__file__).parent
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_dir.glob("*.py")
    )
    production_source += Path(mm_rdma_module.__file__).read_text(encoding="utf-8")

    for name in _LEGACY_PRODUCT_ENV_VARS:
        assert name not in production_source
    assert "kill_process_tree" not in Path(servicer_module.__file__).read_text(encoding="utf-8")
