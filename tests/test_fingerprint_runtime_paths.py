import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.FingerprintParsers import FingerprintResult, Heuristic
from core.TinkerClient import VisionTinkerClient
from experiments.BaselineFingerprint import HDSRow, _run_fingerprinting_async_impl

TINKER_CLIENT_MODULE = importlib.import_module("core.TinkerClient")


class _UnknownParser:
    def fingerprint(self, *_args, **_kwargs) -> FingerprintResult:
        return FingerprintResult(Heuristic.UNKNOWN, 0.0, {})


class _FakeEmbeddingClassifier:
    def __init__(self, heuristic: Heuristic = Heuristic.DD):
        self.heuristic = heuristic

    def fingerprint(self, _trace: str) -> FingerprintResult:
        return FingerprintResult(
            heuristic=self.heuristic,
            confidence=0.75,
            details={
                "resolved": True,
                "status": "ok",
                "model": "fake-embedder",
                "margin": 0.5,
                "support_mass": {"DD": 0.75, "OT": 0.1, "RC": 0.05, "STYLE": 0.1},
            },
        )


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        _ = timeout
        return self._result


class _FakeForwardResult:
    def __init__(self, losses):
        self.loss_fn_outputs = [{"logprobs": [-loss]} for loss in losses]


class _FakeTrainingClient:
    def __init__(self, losses):
        self._losses = list(losses)

    def forward(self, datums, _loss_name):
        return _FakeFuture(_FakeForwardResult(self._losses[: len(datums)]))

    async def forward_async(self, datums, _loss_name):
        return asyncio.sleep(0, result=_FakeForwardResult(self._losses[: len(datums)]))


def _make_fake_vision_client(losses):
    client = VisionTinkerClient.__new__(VisionTinkerClient)
    client.verbose = False
    client._image_token_cache = {}
    client._image_dimension_token_cache = {}
    client._load_image_bytes = lambda _path: b"image-bytes"
    client._get_image_format = lambda _path: "png"
    client._build_vision_datum = lambda **_kwargs: ("datum", [1], 0, 0)
    client._resolve_image_token_count = lambda **_kwargs: 1
    training_client = _FakeTrainingClient(losses)
    client.get_training_client = lambda force_new=False: training_client

    async def _get_training_client_async(force_new=False):
        _ = force_new
        return training_client

    client.get_training_client_async = _get_training_client_async
    return client


class _FakeVisionTokenizer:
    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        if text == TINKER_CLIENT_MODULE.CHAT_VISION_PREFIX:
            return [101, 102]
        if text == TINKER_CLIENT_MODULE.CHAT_VISION_SUFFIX:
            return [201, 202, 203]
        return list(range(max(1, len(text.split()))))


class _FakeVisionTypes:
    class EncodedTextChunk:
        def __init__(self, tokens):
            self.tokens = tokens

    class ImageChunk:
        def __init__(self, data, format, expected_tokens):
            self.data = data
            self.format = format
            self.expected_tokens = expected_tokens

    class ModelInput:
        def __init__(self, chunks):
            self.chunks = chunks

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs


class _FakeVisionTokenTrainingClient:
    def __init__(self, actual_image_tokens: int, loss: float = 0.25):
        self.actual_image_tokens = actual_image_tokens
        self.loss = loss
        self.calls = []

    def _build_result(self, datums):
        outputs = []
        for datum in datums:
            expected_tokens = datum.model_input.chunks[1].expected_tokens
            self.calls.append(expected_tokens)
            if expected_tokens != self.actual_image_tokens:
                raise ValueError(
                    f"Error code: 400 - {{'detail': 'Expected {expected_tokens} tokens, "
                    f"got {self.actual_image_tokens} from image'}}"
                )
            weight_count = len(datum.loss_fn_inputs["weights"])
            outputs.append({"logprobs": [-self.loss] * weight_count})
        return SimpleNamespace(loss_fn_outputs=outputs)

    def forward(self, datums, _loss_name):
        return _FakeFuture(self._build_result(datums))

    async def forward_async(self, datums, _loss_name):
        async def _resolve():
            return self._build_result(datums)

        return _resolve()


def _make_runtime_vision_client(actual_image_tokens: int, loss: float = 0.25):
    client = VisionTinkerClient.__new__(VisionTinkerClient)
    client.verbose = False
    client._image_token_cache = {}
    client._image_dimension_token_cache = {}
    client._tokenizer = _FakeVisionTokenizer()
    client._tinker = SimpleNamespace(types=_FakeVisionTypes)
    client._load_image_bytes = lambda _path: b"image-bytes"
    client._get_image_format = lambda _path: "png"
    training_client = _FakeVisionTokenTrainingClient(actual_image_tokens, loss=loss)
    client.get_training_client = lambda force_new=False: training_client

    async def _get_training_client_async(force_new=False):
        _ = force_new
        return training_client

    client.get_training_client_async = _get_training_client_async
    return client, training_client


def test_async_image_fingerprinting_uses_image_generation_and_scoring(tmp_path: Path) -> None:
    image_path = tmp_path / "hds_001.png"
    image_path.write_bytes(b"fake-image")

    class _FakeServiceClient:
        async def create_sampling_client_async(self, base_model, retry_config=None):
            _ = base_model
            assert retry_config is not None
            assert retry_config.progress_timeout == TINKER_CLIENT_MODULE.API_CALL_TIMEOUT
            return object()

    class _FakeClient:
        def __init__(self, expected_path: Path):
            self.model_name = "fake-model"
            self._service_client = _FakeServiceClient()
            self._tokenizer = object()
            self._tinker = SimpleNamespace()
            self.expected_path = expected_path
            self.image_generation_calls = []
            self.image_probe_calls = []

        def build_text_generation_input(self, _prompt):
            raise AssertionError("text generation path should not run for image modality")

        async def compute_heuristic_losses_multi_async(self, *_args, **_kwargs):
            raise AssertionError("text probe path should not run for image modality")

        async def generate_with_image_async(self, image_path, a, b, sampler, with_reasoning=False, max_tokens=None, prompt_text=None):
            self.image_generation_calls.append(
                (Path(image_path), a, b, sampler, with_reasoning, max_tokens, prompt_text)
            )
            assert Path(image_path) == self.expected_path
            return a * b, f"{a} x {b} = {a * b}"

        async def compute_heuristic_losses_multi_image_async(self, problems, batch_size=30, include_neutral=True, max_in_flight=4):
            self.image_probe_calls.append((list(problems), batch_size, include_neutral, max_in_flight))
            return [
                {
                    "losses": {"OT": 0.2, "DD": 0.6, "RC": 0.8},
                    "neutral_loss": 1.0,
                    "delta_losses": {"OT": -0.8, "DD": -0.4, "RC": -0.2},
                    "per_template_losses": {
                        "OT_0": {"prompt": "ot", "loss": 0.2},
                        "DD_0": {"prompt": "dd", "loss": 0.6},
                        "RC_0": {"prompt": "rc", "loss": 0.8},
                        "NEUTRAL": {"prompt": "neutral", "loss": 1.0},
                    },
                    "best_heuristic": "OT",
                    "confidence": 0.5,
                }
            ]

    class _FakeFingerprinter:
        modality = "image"

        def __init__(self, image_path: Path):
            self.client = _FakeClient(image_path)
            self.error_parser = _UnknownParser()
            self.trace_classifier = _UnknownParser()
            self.embedding_classifier = _FakeEmbeddingClassifier()

        def _get_image_path(self, _hds_id: str) -> Path:
            return image_path

    row = HDSRow(
        id="hds_001",
        a=12,
        b=13,
        product=156,
        target_heuristic="OT",
        ot_score=0.9,
        dd_score=0.2,
        rc_score=0.1,
        category="test",
        notes="",
        split="test",
    )

    results = asyncio.run(
        _run_fingerprinting_async_impl(
            [row],
            _FakeFingerprinter(image_path),
            verbose=False,
            batch_size=8,
            output_dir=None,
            concurrency=1,
            score_max_in_flight=1,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.model_answer == 156
    assert result.is_correct is True
    assert result.detected_heuristic == "OT"
    assert result.embedding_heuristic == "DD"
    assert result.embedding_confidence == pytest.approx(0.75)
    assert result.embedding_resolution_status == "ok"
    assert result.per_template_losses == {
        "OT_0": {"prompt": "ot", "loss": 0.2},
        "DD_0": {"prompt": "dd", "loss": 0.6},
        "RC_0": {"prompt": "rc", "loss": 0.8},
        "NEUTRAL": {"prompt": "neutral", "loss": 1.0},
    }
    assert result.is_contaminated is False


def test_compute_heuristic_losses_multi_image_returns_template_details(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_heuristic_templates", lambda a, b: {"OT": ["ot-a", "ot-b"], "DD": ["dd-a"], "RC": ["rc-a"]})
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_neutral_baseline_template", lambda a, b: "neutral-a")
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda image_path: 1)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda image_path: None)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    client = _make_fake_vision_client([0.2, 0.4, 0.7, 0.9, 1.2])

    results = VisionTinkerClient.compute_heuristic_losses_multi_image(
        client,
        [(image_path, 12, 13)],
        batch_size=30,
        include_neutral=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result["best_heuristic"] == "OT"
    assert result["confidence"] > 0.0
    assert result["losses"]["OT"] == pytest.approx(0.3)
    assert set(result["per_template_losses"]) == {"OT_0", "OT_1", "DD_0", "RC_0", "NEUTRAL"}
    assert result["per_template_losses"]["NEUTRAL"]["loss"] == 1.2
    assert result["delta_losses"]["OT"] == pytest.approx(-0.9)
    assert result["probe_resolved"] is True
    assert result["probe_resolution_status"] == "ok"
    assert result["probe_image_token_count"] == 1


def test_compute_heuristic_losses_multi_image_async_returns_template_details(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_heuristic_templates", lambda a, b: {"OT": ["ot-a", "ot-b"], "DD": ["dd-a"], "RC": ["rc-a"]})
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_neutral_baseline_template", lambda a, b: "neutral-a")
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda image_path: 1)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda image_path: None)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    client = _make_fake_vision_client([0.2, 0.4, 0.7, 0.9, 1.2])

    results = asyncio.run(
        VisionTinkerClient.compute_heuristic_losses_multi_image_async(
            client,
            [(image_path, 12, 13)],
            batch_size=30,
            include_neutral=True,
            max_in_flight=1,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result["best_heuristic"] == "OT"
    assert result["confidence"] > 0.0
    assert result["losses"]["OT"] == pytest.approx(0.3)
    assert set(result["per_template_losses"]) == {"OT_0", "OT_1", "DD_0", "RC_0", "NEUTRAL"}
    assert result["per_template_losses"]["NEUTRAL"]["loss"] == 1.2
    assert result["delta_losses"]["OT"] == pytest.approx(-0.9)
    assert result["probe_resolved"] is True
    assert result["probe_resolution_status"] == "ok"
    assert result["probe_image_token_count"] == 1


def test_compute_heuristic_losses_multi_image_preserves_alignment_on_partial_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_heuristic_templates", lambda a, b: {"OT": ["ot-a", "ot-b"], "DD": ["dd-a"], "RC": ["rc-a"]})
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_neutral_baseline_template", lambda a, b: "neutral-a")
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda image_path: 1)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda image_path: None)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    client = _make_fake_vision_client([0.2, 0.4])

    results = VisionTinkerClient.compute_heuristic_losses_multi_image(
        client,
        [(image_path, 12, 13)],
        batch_size=30,
        include_neutral=True,
    )

    result = results[0]
    assert result["per_template_losses"]["OT_0"]["loss"] == 0.2
    assert result["per_template_losses"]["OT_1"]["loss"] == 0.4
    assert result["per_template_losses"]["DD_0"]["loss"] == float("inf")
    assert result["per_template_losses"]["RC_0"]["loss"] == float("inf")
    assert result["per_template_losses"]["NEUTRAL"]["loss"] == float("inf")


def test_extract_token_count_from_error_supports_image_and_total_formats() -> None:
    client = VisionTinkerClient.__new__(VisionTinkerClient)

    image_only = VisionTinkerClient._extract_token_count_from_error(
        client, "Error code: 400 - {'detail': 'Expected 84 tokens, got 93 from image'}"
    )
    assert image_only is not None
    assert image_only.count == 93
    assert image_only.scope == "image"

    total = VisionTinkerClient._extract_token_count_from_error(
        client, "bad request: token_count=117 sequence too long"
    )
    assert total is not None
    assert total.count == 117
    assert total.scope == "total"


def test_compute_perplexity_with_image_retries_using_image_only_count(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda _image_path: 84)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda _image_path: (963, 67))

    image_path = tmp_path / "hds_936.png"
    image_path.write_bytes(b"fake-image")

    client, training_client = _make_runtime_vision_client(actual_image_tokens=93, loss=0.25)

    loss = VisionTinkerClient.compute_perplexity_with_image(
        client,
        image_path=image_path,
        text_continuation="probe continuation",
    )

    assert loss == pytest.approx(0.25)
    assert training_client.calls == [84, 93]
    assert client._image_token_cache[str(image_path)] == 93
    assert client._image_dimension_token_cache[(963, 67)] == 93


def test_resolve_image_token_count_uses_image_only_mismatch_count(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda _image_path: 84)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda _image_path: (963, 67))

    image_path = tmp_path / "hds_936.png"
    image_path.write_bytes(b"fake-image")

    client, training_client = _make_runtime_vision_client(actual_image_tokens=93, loss=0.25)

    resolved = VisionTinkerClient._resolve_image_token_count(
        client,
        image_path=image_path,
        image_bytes=b"image-bytes",
        image_format="png",
        text_continuation="probe continuation",
        training_client=training_client,
        force_refresh=True,
    )

    assert resolved == 93
    assert training_client.calls == [84]
    assert client._image_token_cache[str(image_path)] == 93
    assert client._image_dimension_token_cache[(963, 67)] == 93


def test_dimension_cache_reuses_corrected_token_count_across_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda _image_path: 84)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda _image_path: (963, 67))

    first_path = tmp_path / "hds_936.png"
    second_path = tmp_path / "hds_937.png"
    first_path.write_bytes(b"fake-image-1")
    second_path.write_bytes(b"fake-image-2")

    client, training_client = _make_runtime_vision_client(actual_image_tokens=93, loss=0.25)

    first_loss = VisionTinkerClient.compute_perplexity_with_image(
        client,
        image_path=first_path,
        text_continuation="probe continuation",
    )
    second_loss = VisionTinkerClient.compute_perplexity_with_image(
        client,
        image_path=second_path,
        text_continuation="probe continuation",
    )

    assert first_loss == pytest.approx(0.25)
    assert second_loss == pytest.approx(0.25)
    assert training_client.calls == [84, 93, 93]
    assert client._image_token_cache[str(first_path)] == 93
    assert client._image_token_cache[str(second_path)] == 93
    assert client._image_dimension_token_cache[(963, 67)] == 93


def test_compute_heuristic_losses_multi_image_async_recovers_after_image_token_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        TINKER_CLIENT_MODULE,
        "get_image_heuristic_templates",
        lambda a, b: {"OT": [f"ot-{a}-{b}"], "DD": [f"dd-{a}-{b}"], "RC": [f"rc-{a}-{b}"]},
    )
    monkeypatch.setattr(
        TINKER_CLIENT_MODULE,
        "get_image_neutral_baseline_template",
        lambda a, b: f"neutral-{a}-{b}",
    )
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_token_count", lambda _image_path: 84)
    monkeypatch.setattr(TINKER_CLIENT_MODULE, "get_image_dimensions", lambda _image_path: (963, 67))

    image_path = tmp_path / "hds_936.png"
    image_path.write_bytes(b"fake-image")

    client, training_client = _make_runtime_vision_client(actual_image_tokens=93, loss=0.3)

    results = asyncio.run(
        VisionTinkerClient.compute_heuristic_losses_multi_image_async(
            client,
            [(image_path, 12, 13)],
            batch_size=16,
            include_neutral=True,
            max_in_flight=1,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result["losses"]["OT"] == pytest.approx(0.3)
    assert result["losses"]["DD"] == pytest.approx(0.3)
    assert result["losses"]["RC"] == pytest.approx(0.3)
    assert result["neutral_loss"] == pytest.approx(0.3)
    assert result["delta_losses"] == {"OT": 0.0, "DD": 0.0, "RC": 0.0}
    assert set(result["per_template_losses"]) == {"OT_0", "DD_0", "RC_0", "NEUTRAL"}
    assert training_client.calls[0] == 84
    assert 93 in training_client.calls[1:]
    assert result["probe_resolved"] is True
    assert result["probe_resolution_status"] in {"retry_resolved", "ok"}
    assert result["probe_image_token_count"] == 93
