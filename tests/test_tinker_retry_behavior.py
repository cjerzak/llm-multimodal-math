import concurrent.futures
import importlib
from types import SimpleNamespace

import httpx
import pytest
import tinker

from core.TinkerClient import (
    API_CALL_TIMEOUT,
    TinkerClient,
    build_tinker_sampling_retry_config,
)
from experiments.LoRANudgeTest import NudgeTester

TINKER_CLIENT_MODULE = importlib.import_module("core.TinkerClient")
LORA_NUDGE_MODULE = importlib.import_module("experiments.LoRANudgeTest")


class _TimeoutFuture:
    def result(self, timeout=None):
        _ = timeout
        raise concurrent.futures.TimeoutError


class _CountingForwardClient:
    def __init__(self):
        self.calls = 0

    def forward(self, _datums, _loss_name):
        self.calls += 1
        return _TimeoutFuture()


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CountingSampler:
    def __init__(self):
        self.calls = 0

    def sample(self, *, prompt, sampling_params, num_samples):
        _ = (prompt, sampling_params, num_samples)
        self.calls += 1
        return _TimeoutFuture()


class _FakeSamplingServiceClient:
    def __init__(self, side_effects):
        self.calls = []
        self._side_effects = list(side_effects)

    def create_sampling_client(self, *, model_path=None, retry_config=None):
        self.calls.append((model_path, retry_config))
        if not self._side_effects:
            raise AssertionError("unexpected create_sampling_client call")
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def _make_status_error(status_code: int, error_type):
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return error_type(
        f"status {status_code}",
        response=response,
        body={"detail": f"status {status_code}"},
    )


def _make_nudge_tester(service_client, *, adapter_paths=None):
    tester = NudgeTester.__new__(NudgeTester)
    tester.client = SimpleNamespace(service_client=service_client)
    tester.model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    tester.adapter_paths = adapter_paths or {}
    tester.state_paths = {}
    tester._sampling_clients = {}
    tester._training_clients = {}
    return tester


def test_build_tinker_sampling_retry_config_uses_api_timeout() -> None:
    config = build_tinker_sampling_retry_config()
    assert config.progress_timeout == API_CALL_TIMEOUT


def test_compute_heuristic_losses_multi_calls_forward_once_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        TINKER_CLIENT_MODULE,
        "get_multi_heuristic_templates",
        lambda a, b: {"OT": [f"ot-{a}-{b}"], "DD": [f"dd-{a}-{b}"], "RC": [f"rc-{a}-{b}"]},
    )
    monkeypatch.setattr(
        TINKER_CLIENT_MODULE,
        "get_neutral_baseline_template",
        lambda a, b: f"neutral-{a}-{b}",
    )

    training_client = _CountingForwardClient()
    client = TinkerClient.__new__(TinkerClient)
    client.verbose = False
    client.get_training_client = lambda: training_client
    client._build_text_response_datum = lambda _prompt, _continuation: ("datum", [1.0])

    results = TinkerClient.compute_heuristic_losses_multi(
        client,
        [(12, 13)],
        batch_size=30,
        include_neutral=True,
    )

    assert training_client.calls == 1
    assert len(results) == 1
    assert results[0]["best_heuristic"] == "UNKNOWN"


def test_nudge_generate_answer_calls_sample_once_on_timeout() -> None:
    sampler = _CountingSampler()
    tester = NudgeTester.__new__(NudgeTester)
    tester.prompt_template = "What is {a} x {b}?"
    tester.generation_max_tokens = 32
    tester._get_sampling_client = lambda adapter_name=None: sampler
    tester.client = SimpleNamespace(
        build_text_generation_input=lambda _prompt: "model-input",
        tokenizer=SimpleNamespace(encode=lambda text: [1], decode=lambda tokens: "decoded"),
        tinker=SimpleNamespace(types=SimpleNamespace(SamplingParams=_FakeSamplingParams)),
    )

    answer, text = NudgeTester.generate_answer(tester, 2, 3)

    assert sampler.calls == 1
    assert answer is None
    assert text is None


def test_nudge_sampling_client_skips_registry_fallback_on_auth_error(monkeypatch) -> None:
    service_client = _FakeSamplingServiceClient(
        [_make_status_error(401, tinker.AuthenticationError)]
    )
    tester = _make_nudge_tester(
        service_client,
        adapter_paths={"rc_lora": "tinker://stored-path"},
    )
    discovery_calls = []
    monkeypatch.setattr(
        LORA_NUDGE_MODULE,
        "discover_checkpoint_from_registry",
        lambda *args, **kwargs: discovery_calls.append((args, kwargs)) or "tinker://discovered-path",
    )

    with pytest.raises(RuntimeError, match="skipping registry fallback"):
        NudgeTester._get_sampling_client(tester, "rc_lora")

    assert discovery_calls == []
    assert len(service_client.calls) == 1
    assert service_client.calls[0][1] is not None
    assert service_client.calls[0][1].progress_timeout == API_CALL_TIMEOUT


def test_nudge_sampling_client_rediscovers_on_missing_checkpoint(monkeypatch) -> None:
    service_client = _FakeSamplingServiceClient(
        [
            _make_status_error(404, tinker.NotFoundError),
            {"path": "tinker://discovered-path"},
        ]
    )
    tester = _make_nudge_tester(
        service_client,
        adapter_paths={"rc_lora": "tinker://stored-path"},
    )
    discovery_calls = []
    monkeypatch.setattr(
        LORA_NUDGE_MODULE,
        "discover_checkpoint_from_registry",
        lambda *args, **kwargs: discovery_calls.append((args, kwargs)) or "tinker://discovered-path",
    )

    result = NudgeTester._get_sampling_client(tester, "rc_lora")

    assert result == {"path": "tinker://discovered-path"}
    assert len(discovery_calls) == 1
    assert tester.adapter_paths["rc_lora"] == "tinker://discovered-path"
    assert [call[0] for call in service_client.calls] == [
        "tinker://stored-path",
        "tinker://discovered-path",
    ]
    assert all(call[1] is not None for call in service_client.calls)
    assert all(call[1].progress_timeout == API_CALL_TIMEOUT for call in service_client.calls)
