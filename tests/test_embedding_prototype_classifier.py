from pathlib import Path

import pytest

from core.FingerprintParsers import Heuristic, PrototypeEmbeddingClassifier


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [self._embed_one(text) for text in texts]

    @staticmethod
    def _embed_one(text: str):
        lowered = text.lower()
        if "decompose" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        if "column" in lowered or "carry" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "round" in lowered or "compensate" in lowered:
            return [0.0, 0.0, 1.0, 0.0]
        if "organize" in lowered or "exact product" in lowered:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.25, 0.25, 0.25, 0.25]


def _write_training_csv(path: Path, trace: str) -> None:
    path.write_text(
        "id,a,b,product,heuristic,prompt,reasoning_trace,full_text,split\n"
        f"ex1,12,34,408,x,ignored,\"{trace}\",ignored,train\n"
    )


def test_prototype_embedding_classifier_normalizes_trace_and_returns_style_aware_support(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    _write_training_csv(training_dir / "dd_training.csv", "What is 12 × 34? Let me decompose 34 into 30 + 4. Answer: 408")
    _write_training_csv(training_dir / "ot_training.csv", "What is 12 × 34? Let me use column multiplication and carry the 1. Answer: 408")
    _write_training_csv(training_dir / "rc_training.csv", "What is 49 × 51? Let me round to 50 and compensate. Answer: 2499")
    _write_training_csv(training_dir / "style_training.csv", "What is 22 × 33? Let me organize the work clearly. Answer: 726")

    classifier = PrototypeEmbeddingClassifier(
        "fake-embed-model",
        training_data_dir=training_dir,
        cache_dir=tmp_path / "cache",
        embedder=_FakeEmbedder(),
    )

    result = classifier.fingerprint(
        "What is 98 × 14?\nLet me decompose 14 into 10 + 4.\nAnswer: 1372"
    )

    assert result.heuristic == Heuristic.DD
    assert result.confidence > 0.9
    assert result.details is not None
    assert result.details["normalized_trace"].startswith("Let me decompose")
    assert "<NUM>" in result.details["normalized_trace"]
    assert "Answer" not in result.details["normalized_trace"]
    support_mass = result.details["support_mass"]
    assert set(support_mass) == {"DD", "OT", "RC", "STYLE"}
    assert pytest.approx(sum(support_mass.values())) == 1.0


def test_prototype_embedding_classifier_reuses_cached_prototypes(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    _write_training_csv(training_dir / "dd_training.csv", "What is 12 × 34? Let me decompose 34 into 30 + 4. Answer: 408")
    _write_training_csv(training_dir / "ot_training.csv", "What is 12 × 34? Let me use column multiplication and carry the 1. Answer: 408")
    _write_training_csv(training_dir / "rc_training.csv", "What is 49 × 51? Let me round to 50 and compensate. Answer: 2499")
    _write_training_csv(training_dir / "style_training.csv", "What is 22 × 33? Let me organize the work clearly. Answer: 726")

    cache_dir = tmp_path / "cache"

    first_embedder = _FakeEmbedder()
    classifier = PrototypeEmbeddingClassifier(
        "fake-embed-model",
        training_data_dir=training_dir,
        cache_dir=cache_dir,
        embedder=first_embedder,
    )
    classifier.fingerprint("What is 12 × 34? Let me decompose 34 into 30 + 4. Answer: 408")
    assert first_embedder.calls >= 2  # prototypes + runtime trace
    assert any(cache_dir.glob("*.json"))

    second_embedder = _FakeEmbedder()
    classifier_cached = PrototypeEmbeddingClassifier(
        "fake-embed-model",
        training_data_dir=training_dir,
        cache_dir=cache_dir,
        embedder=second_embedder,
    )
    classifier_cached.fingerprint("What is 22 × 33? Let me organize the work clearly. Answer: 726")

    assert second_embedder.calls == 1


def test_prototype_embedding_classifier_warmup_materializes_prototypes_early(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    _write_training_csv(training_dir / "dd_training.csv", "What is 12 × 34? Let me decompose 34 into 30 + 4. Answer: 408")
    _write_training_csv(training_dir / "ot_training.csv", "What is 12 × 34? Let me use column multiplication and carry the 1. Answer: 408")
    _write_training_csv(training_dir / "rc_training.csv", "What is 49 × 51? Let me round to 50 and compensate. Answer: 2499")
    _write_training_csv(training_dir / "style_training.csv", "What is 22 × 33? Let me organize the work clearly. Answer: 726")

    embedder = _FakeEmbedder()
    cache_dir = tmp_path / "cache"
    classifier = PrototypeEmbeddingClassifier(
        "fake-embed-model",
        training_data_dir=training_dir,
        cache_dir=cache_dir,
        embedder=embedder,
    )

    prototype_pack = classifier.warmup()

    assert embedder.calls == 4
    assert prototype_pack["prototype_counts"] == {"DD": 1, "OT": 1, "RC": 1, "STYLE": 1}
    assert any(cache_dir.glob("*.json"))
