import asyncio
from pathlib import Path
from types import SimpleNamespace

from Scripts.experiments import LoRANudgeTest as nudge_module


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, tokens) -> str:
        return "decoded"


class _FakeSamplingClient:
    async def sample_async(self, prompt, sampling_params, num_samples):
        return SimpleNamespace(completions=["Answer: 0"])


class _FakeTextClient:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.tinker = SimpleNamespace(
            types=SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs,
                ModelInput=SimpleNamespace(from_ints=lambda tokens: tokens),
            )
        )

    def build_text_generation_input(self, prompt_text: str):
        return {"prompt": prompt_text}

    async def compute_heuristic_losses_multi_async(self, batch_problems, **kwargs):
        return [_fake_loss_payload() for _ in batch_problems]


class _FakeImageClient:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.tinker = SimpleNamespace(
            types=SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)
        )
        self._renderer = None

    def _load_image_bytes(self, image_path: Path) -> bytes:
        return b"image-bytes"

    def _get_image_format(self, image_path: Path) -> str:
        return "png"

    def _build_multimodal_input_for_sampling(self, image_bytes: bytes, image_format: str, prompt_text: str):
        return {"image_format": image_format, "prompt": prompt_text}

    async def compute_heuristic_losses_multi_image_async(self, batch, **kwargs):
        return [_fake_loss_payload() for _ in batch]


def _fake_loss_payload() -> dict:
    return {
        "losses": {"RC": 1.0, "DD": 2.0},
        "neutral_loss": 1.5,
        "delta_losses": {"RC": -0.5, "DD": 0.5},
    }


def _make_rows(count: int) -> list[nudge_module.HDSRow]:
    return [
        nudge_module.HDSRow(
            id=f"hds_{i}",
            a=i + 2,
            b=3,
            product=(i + 2) * 3,
            target_heuristic="RC",
        )
        for i in range(count)
    ]


def test_progress_heartbeat_logs_on_count_threshold() -> None:
    logs = []
    clock = _FakeClock()
    heartbeat = nudge_module._ProgressHeartbeat(
        "Generation progress",
        total=25,
        item_interval=10,
        time_interval_seconds=60.0,
        emit_fn=logs.append,
        now_fn=clock,
    )

    heartbeat.log_start("Starting async generation")
    heartbeat.maybe_log(5)
    heartbeat.maybe_log(10)

    assert logs == [
        "  Starting async generation for 25 problems (heartbeat every 10 items or 60s)...",
        "    Generation progress: 10/25 (elapsed 0s)",
    ]


def test_progress_heartbeat_logs_on_time_threshold() -> None:
    logs = []
    clock = _FakeClock()
    heartbeat = nudge_module._ProgressHeartbeat(
        "Scoring progress",
        total=25,
        item_interval=10,
        time_interval_seconds=60.0,
        emit_fn=logs.append,
        now_fn=clock,
    )

    clock.advance(61)
    heartbeat.maybe_log(2)

    assert logs == ["    Scoring progress: 2/25 (elapsed 1m01s)"]


def test_progress_heartbeat_logs_on_final_completion() -> None:
    logs = []
    clock = _FakeClock()
    heartbeat = nudge_module._ProgressHeartbeat(
        "Generation progress",
        total=25,
        item_interval=10,
        time_interval_seconds=60.0,
        emit_fn=logs.append,
        now_fn=clock,
    )

    heartbeat.maybe_log(25)

    assert logs == ["    Generation progress: 25/25 (elapsed 0s)"]


def test_text_async_evaluation_emits_generation_and_scoring_heartbeats(monkeypatch) -> None:
    logs = []
    rows = _make_rows(4)
    tester = object.__new__(nudge_module.NudgeTester)
    tester.client = _FakeTextClient()
    tester.generation_max_tokens = 32
    tester.prompt_template = "What is {a} x {b}?"
    tester.score_max_in_flight = 4

    async def fake_get_sampling_client_async(adapter_name=None):
        return _FakeSamplingClient()

    async def fake_get_training_client_async(adapter_name=None):
        return object()

    tester._get_sampling_client_async = fake_get_sampling_client_async
    tester._get_training_client_async = fake_get_training_client_async

    monkeypatch.setattr(nudge_module, "PROGRESS_HEARTBEAT_ITEMS", 2)
    monkeypatch.setattr(nudge_module, "PROGRESS_HEARTBEAT_SECONDS", 9999.0)
    monkeypatch.setattr(
        nudge_module,
        "tprint",
        lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(
        nudge_module,
        "extract_answer_enhanced",
        lambda text, a, b: SimpleNamespace(
            answer=a * b,
            confidence=1.0,
            strategy="test",
            is_truncated=False,
        ),
    )

    results = asyncio.run(
        tester._evaluate_hds_async(
            rows,
            adapter_name=None,
            verbose=True,
            batch_size=2,
            concurrency=1,
        )
    )

    assert len(results) == 4
    assert any("Starting async generation for 4 problems" in line for line in logs)
    assert any("Generation progress: 2/4" in line for line in logs)
    assert any("Generation progress: 4/4" in line for line in logs)
    assert any("Starting async scoring for 4 problems" in line for line in logs)
    assert any("Scoring progress: 2/4" in line for line in logs)
    assert any("Scoring progress: 4/4" in line for line in logs)


def test_image_async_evaluation_emits_generation_and_scoring_heartbeats(monkeypatch, tmp_path: Path) -> None:
    logs = []
    rows = _make_rows(4)
    for row in rows:
        (tmp_path / f"{row.id}.png").write_bytes(b"image")

    tester = object.__new__(nudge_module.ImageNudgeTester)
    tester.client = _FakeImageClient()
    tester.images_dir = tmp_path
    tester.generation_max_tokens = 32
    tester.prompt_text = "Solve the problem."
    tester.score_max_in_flight = 4

    async def fake_get_sampling_client_async(adapter_name=None):
        return _FakeSamplingClient()

    async def fake_get_training_client_async(adapter_name=None):
        return object()

    tester._get_sampling_client_async = fake_get_sampling_client_async
    tester._get_training_client_async = fake_get_training_client_async

    monkeypatch.setattr(nudge_module, "PROGRESS_HEARTBEAT_ITEMS", 2)
    monkeypatch.setattr(nudge_module, "PROGRESS_HEARTBEAT_SECONDS", 9999.0)
    monkeypatch.setattr(
        nudge_module,
        "tprint",
        lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args)),
    )
    monkeypatch.setattr(
        nudge_module,
        "extract_answer_enhanced",
        lambda text, a, b: SimpleNamespace(
            answer=a * b,
            confidence=1.0,
            strategy="test",
            is_truncated=False,
        ),
    )

    results = asyncio.run(
        tester._evaluate_hds_async(
            rows,
            adapter_name=None,
            verbose=True,
            batch_size=2,
            concurrency=1,
        )
    )

    assert len(results) == 4
    assert any("Starting async image generation for 4 problems" in line for line in logs)
    assert any("Image generation progress: 2/4" in line for line in logs)
    assert any("Image generation progress: 4/4" in line for line in logs)
    assert any("Starting async image scoring for 4 problems" in line for line in logs)
    assert any("Image scoring progress: 2/4" in line for line in logs)
    assert any("Image scoring progress: 4/4" in line for line in logs)
