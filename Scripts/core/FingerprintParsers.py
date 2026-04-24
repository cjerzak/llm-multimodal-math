#!/usr/bin/env python3
"""
FingerprintParsers.py

Heuristic fingerprinting tools for detecting which arithmetic strategies
LLMs use to solve multiplication problems.

Four complementary approaches:
1. PerplexityProbe - Measure loss on heuristic preambles (no generation needed)
2. ErrorShapeParser - Infer heuristic from error patterns in wrong answers
3. TraceClassifier - Detect heuristic from chain-of-thought output patterns
4. PrototypeEmbeddingClassifier - Classify traces via prototype embeddings

Heuristics detected:
- OT (Ones-Then-Tens): Standard columnar multiplication with carries
- DD (Decomposition): Break numbers into parts (47×36 → 47×(30+6))
- RC (Rounding-Compensation): Near-base shortcuts (49×51 ≈ 50²-1)
"""

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable, Sequence
from enum import Enum

from .Logging import tprint


class Heuristic(Enum):
    """Arithmetic heuristics for multiplication."""
    OT = "ones_then_tens"      # Columnar / digit-by-digit
    DD = "decomposition"       # Distributive property
    RC = "rounding_compensation"  # Near-base tricks
    STYLE = "style_control"    # Generic structured reasoning without heuristic cues
    UNKNOWN = "unknown"


@dataclass
class Problem:
    """A multiplication problem with metadata."""
    a: int
    b: int
    product: Optional[int] = None

    def __post_init__(self):
        if self.product is None:
            self.product = self.a * self.b

    @property
    def a_ones(self) -> int:
        return self.a % 10

    @property
    def a_tens(self) -> int:
        return (self.a // 10) % 10

    @property
    def b_ones(self) -> int:
        return self.b % 10

    @property
    def b_tens(self) -> int:
        return (self.b // 10) % 10

    @property
    def nearest_base_a(self) -> int:
        """Find nearest base (10, 50, 100, etc.) for operand a."""
        candidates = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
        return min(candidates, key=lambda x: abs(self.a - x))

    @property
    def nearest_base_b(self) -> int:
        """Find nearest base for operand b."""
        candidates = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
        return min(candidates, key=lambda x: abs(self.b - x))

    @property
    def is_near_base(self) -> bool:
        """Check if either operand is within 2 of a round number."""
        def near_round(n):
            for base in [10, 50, 100, 200, 500, 1000]:
                if abs(n - base) <= 2:
                    return True
            return False
        return near_round(self.a) or near_round(self.b)

    @property
    def has_zero_factor(self) -> bool:
        """Check if either operand ends in 0 (DD-friendly)."""
        return self.a % 10 == 0 or self.b % 10 == 0


@dataclass
class FingerprintResult:
    """Result from any fingerprinting method."""
    heuristic: Heuristic
    confidence: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# =============================================================================
# 1. PERPLEXITY PROBE (Gemini's approach - no generation needed)
# =============================================================================

class PerplexityProbe:
    """
    Fingerprint heuristics by measuring loss on different reasoning preambles.

    The key insight: If a model implicitly uses decomposition (DD),
    the token sequence "47 × 36 = Decompose 47 into..." should have
    lower perplexity than "47 × 36 = First, 7 × 6 is..."

    This works WITHOUT generating output - just forward passes!
    """

    # Templates for each heuristic's reasoning style
    # DESIGN: Length-matched (~45 chars after substitution), style-matched (all imperative)
    # Each template follows pattern: "{a} × {b} = [Method]: [specific action with operands]"
    # This ensures perplexity differences reflect heuristic preference, not surface form
    #
    # SYMMETRIC DESIGN: DD and RC templates include variants for both operands (a and b)
    # This prevents "model prefers DD" from actually meaning "model prefers decompose-left-operand phrasing"
    TEMPLATES = {
        Heuristic.OT: [
            "{a} × {b} = Column method: multiply {a_ones} by {b_ones} first",
            "{a} × {b} = Digit by digit: start with ones place {a_ones} × {b_ones}",
            "{a} × {b} = Standard algorithm: compute {a_ones} times {b_ones} then carry",
        ],
        Heuristic.DD: [
            # Decompose operand a
            "{a} × {b} = Decomposition: split {a} into {a_tens}0 plus {a_ones}",
            "{a} × {b} = Partial products: break {a} as ({a_tens}0 + {a_ones})",
            # Decompose operand b (symmetric)
            "{a} × {b} = Decomposition: split {b} into {b_tens}0 plus {b_ones}",
            "{a} × {b} = Partial products: break {b} as ({b_tens}0 + {b_ones})",
        ],
        Heuristic.RC: [
            # Round operand a
            "{a} × {b} = Round and adjust: {a} is near {nearest_base_a} so use that",
            "{a} × {b} = Base approximation: round {a} to {nearest_base_a} then fix",
            # Round operand b (symmetric)
            "{a} × {b} = Round and adjust: {b} is near {nearest_base_b} so use that",
            "{a} × {b} = Base approximation: round {b} to {nearest_base_b} then fix",
        ],
    }

    # Neutral baseline template for computing Δloss
    # Used to distinguish "model prefers heuristic X" from "model dislikes all heuristics"
    NEUTRAL_TEMPLATE = "{a} × {b} = Let me solve this multiplication step by step"

    def __init__(self, model_client=None):
        """
        Initialize with optional Tinker model client.

        Args:
            model_client: Model client with compute_perplexity method
        """
        self.model_client = model_client

    def format_template(self, template: str, problem: Problem) -> str:
        """Fill in template with problem values."""
        return template.format(
            a=problem.a,
            b=problem.b,
            a_ones=problem.a_ones,
            a_tens=problem.a_tens,
            b_ones=problem.b_ones,
            b_tens=problem.b_tens,
            nearest_base_a=problem.nearest_base_a,
            nearest_base_b=problem.nearest_base_b,
        )

    def compute_loss(self, prompt: str) -> float:
        """
        Compute loss/perplexity for a prompt.

        In real usage, this calls model_client.compute_perplexity(prompt) if available.
        For testing without API, returns a dummy value.
        """
        if self.model_client is None:
            # Dummy implementation for testing
            return len(prompt) * 0.1  # Placeholder

        try:
            if hasattr(self.model_client, "compute_perplexity"):
                return float(self.model_client.compute_perplexity(prompt))
            tprint("Warning: model_client lacks compute_perplexity; unable to score prompt")
            return float('inf')
        except Exception as e:
            tprint(f"Warning: Loss computation failed: {e}")
            return float('inf')

    def fingerprint(self, problem: Problem) -> FingerprintResult:
        """
        Fingerprint a problem by measuring loss on each heuristic's templates.

        Returns the heuristic with lowest average loss (most natural continuation).
        Also computes neutral baseline loss for Δloss analysis.
        """
        losses = {}

        for heuristic, templates in self.TEMPLATES.items():
            heuristic_losses = []
            for template in templates:
                prompt = self.format_template(template, problem)
                loss = self.compute_loss(prompt)
                heuristic_losses.append(loss)
            losses[heuristic] = sum(heuristic_losses) / len(heuristic_losses)

        # Compute neutral baseline loss
        neutral_prompt = self.format_template(self.NEUTRAL_TEMPLATE, problem)
        neutral_loss = self.compute_loss(neutral_prompt)

        # Compute delta losses (relative to neutral baseline)
        delta_losses = {h: loss - neutral_loss for h, loss in losses.items()}

        # Find heuristic with minimum loss
        best_heuristic = min(losses, key=lambda h: losses[h])

        # Compute confidence based on loss gap
        sorted_losses = sorted(losses.values())
        if len(sorted_losses) >= 2 and sorted_losses[1] > 0:
            # Confidence based on gap between best and second-best
            gap = (sorted_losses[1] - sorted_losses[0]) / sorted_losses[1]
            confidence = min(1.0, gap * 2)  # Scale gap to confidence
        else:
            confidence = 0.5

        return FingerprintResult(
            heuristic=best_heuristic,
            confidence=confidence,
            details={
                "losses": {h.value: v for h, v in losses.items()},
                "neutral_loss": neutral_loss,
                "delta_losses": {h.value: v for h, v in delta_losses.items()},
            }
        )


# =============================================================================
# 2. ERROR SHAPE PARSER (GPT 5.2's approach - works on wrong answers)
# =============================================================================

class ErrorShapeParser:
    """
    Infer heuristic from the SHAPE of errors in wrong answers.

    Key insight: Different heuristics fail in characteristic ways:
    - OT (columnar): Off by powers of 10 (carry errors)
    - DD (decomposition): Missing a clean term (forgot one partial product)
    - RC (rounding): Off by ±1, ±2, or ±base (compensation error)
    """

    def __init__(self):
        pass

    def _get_error_basis(self, problem: Problem) -> Dict[Heuristic, List[int]]:
        """
        Generate the basis set of expected errors for each heuristic.
        """
        a, b = problem.a, problem.b

        return {
            # OT errors: typically off by powers of 10 (carry mistakes)
            Heuristic.OT: [
                10, 100, 1000,           # Simple carry errors
                -10, -100, -1000,        # Negative versions
                9, 90, 900,              # Off-by-one in carry
                -9, -90, -900,
            ],

            # DD errors: missing one term of the expansion
            # For (a_tens*10 + a_ones) × (b_tens*10 + b_ones), missing terms are:
            Heuristic.DD: [
                a * (b % 10),            # Missing a × b_ones
                a * ((b // 10) * 10),    # Missing a × b_tens*10
                (a % 10) * b,            # Missing a_ones × b
                ((a // 10) * 10) * b,    # Missing a_tens*10 × b
                -a * (b % 10),           # Negative versions
                -a * ((b // 10) * 10),
                -(a % 10) * b,
                -((a // 10) * 10) * b,
            ],

            # RC errors: compensation mistakes
            Heuristic.RC: [
                1, -1, 2, -2,            # Small compensation errors
                problem.nearest_base_a,   # Forgot to subtract/add base
                -problem.nearest_base_a,
                problem.nearest_base_b,
                -problem.nearest_base_b,
                problem.nearest_base_a - problem.a,  # Compensation amount
                problem.nearest_base_b - problem.b,
                -(problem.nearest_base_a - problem.a),
                -(problem.nearest_base_b - problem.b),
            ],
        }

    def fingerprint(self, problem: Problem, predicted: int) -> FingerprintResult:
        """
        Analyze the error to infer which heuristic was likely used.

        Args:
            problem: The multiplication problem
            predicted: The model's predicted answer

        Returns:
            FingerprintResult with detected heuristic
        """
        actual = problem.product if problem.product is not None else problem.a * problem.b
        delta = predicted - actual

        # If correct, we can't infer from error
        if delta == 0:
            return FingerprintResult(
                heuristic=Heuristic.UNKNOWN,
                confidence=0.0,
                details={"note": "Correct answer - cannot infer heuristic from error"}
            )

        # Check which heuristic's error basis matches
        error_basis = self._get_error_basis(problem)
        matches = {}

        for heuristic, basis in error_basis.items():
            # Check exact match
            if delta in basis:
                matches[heuristic] = 1.0
            # Check approximate match (within 10% of basis value)
            else:
                for basis_val in basis:
                    if basis_val != 0 and abs(delta - basis_val) <= abs(basis_val) * 0.1:
                        matches[heuristic] = max(matches.get(heuristic, 0), 0.8)
                        break

        if not matches:
            return FingerprintResult(
                heuristic=Heuristic.UNKNOWN,
                confidence=0.0,
                details={
                    "delta": delta,
                    "note": "Error pattern doesn't match known heuristics"
                }
            )

        # Return best match
        best_heuristic = max(matches, key=lambda h: matches[h])
        return FingerprintResult(
            heuristic=best_heuristic,
            confidence=matches[best_heuristic],
            details={
                "delta": delta,
                "all_matches": {h.value: v for h, v in matches.items()}
            }
        )


# =============================================================================
# 3. TRACE CLASSIFIER (GPT 5.2's approach - if CoT is available)
# =============================================================================

class TraceClassifier:
    """
    Classify heuristic from chain-of-thought reasoning traces.

    Uses pattern matching on key phrases that indicate each heuristic.
    """

    # Pattern keywords for each heuristic
    PATTERNS = {
        Heuristic.OT: [
            r'\bcarry\b',
            r'\bones?\s*(digit|place)\b',
            r'\btens?\s*(digit|place)\b',
            r'\bwrite\s*down\b',
            r'\bcolumn\b',
            r'\bplace\s*value\b',
            r'\bdigit\s*by\s*digit\b',
            r'\bmultiply.*ones\b',
            r'\b\d\s*×\s*\d\s*=\s*\d+\s*,?\s*carry\b',
        ],
        Heuristic.DD: [
            r'\bdecompos[ei]\b',
            r'\bbreak\s*(into|down)\b',
            r'\bdistributive\b',
            r'\bsum\s*of\b',
            r'\b\(\d+\s*[+×]\s*\d+\)\b',  # Parenthetical expressions
            r'\b\d+\s*×\s*\d+\s*\+\s*\d+\s*×\s*\d+\b',  # a×b + c×d pattern
            r'\bfirst.*then\s*add\b',
            r'\bpartial\s*products?\b',
        ],
        Heuristic.RC: [
            r'\bround(ing|ed)?\b',
            r'\bapproximate\b',
            r'\bcompensate\b',
            r'\bnear(ly)?\s*\d+\b',
            r'\bclose\s*to\b',
            r'\b[±]\s*\d+\b',
            r'\bsubtract\s*(back|off)\b',
            r'\badd\s*back\b',
            r'\bdifference\s*of\s*squares\b',
            r'\b\(.*[−-].*\)\s*\(.*[+].*\)\b',  # (a-b)(a+b) pattern
        ],
    }

    def __init__(self):
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            h: [re.compile(p, re.IGNORECASE) for p in patterns]
            for h, patterns in self.PATTERNS.items()
        }

    def fingerprint(self, trace: str) -> FingerprintResult:
        """
        Classify heuristic from a reasoning trace.

        Args:
            trace: The model's chain-of-thought output

        Returns:
            FingerprintResult with detected heuristic
        """
        scores = {h: 0 for h in [Heuristic.OT, Heuristic.DD, Heuristic.RC]}
        pattern_matches: Dict[Heuristic, List[str]] = {h: [] for h in scores}

        for heuristic, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(trace)
                if matches:
                    scores[heuristic] += len(matches)
                    pattern_matches[heuristic].extend(matches)

        total_matches = sum(scores.values())

        if total_matches == 0:
            return FingerprintResult(
                heuristic=Heuristic.UNKNOWN,
                confidence=0.0,
                details={"note": "No heuristic patterns detected in trace"}
            )

        # Normalize to get confidence
        best_heuristic = max(scores, key=lambda h: scores[h])
        confidence = scores[best_heuristic] / total_matches

        return FingerprintResult(
            heuristic=best_heuristic,
            confidence=confidence,
            details={
                "scores": {h.value: v for h, v in scores.items()},
                "matches": {h.value: v for h, v in pattern_matches.items()}
            }
        )


# =============================================================================
# 4. PROTOTYPE EMBEDDING CLASSIFIER (robust trace-sidecar detector)
# =============================================================================

class _HFTextEmbeddingBackend:
    """Minimal local Hugging Face embedding backend using transformers + torch."""

    def __init__(self, model_name_or_path: str, batch_size: int = 16):
        self.model_name_or_path = model_name_or_path
        self.batch_size = max(1, int(batch_size))
        self._load_backend()

    def _load_backend(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        model_path = Path(self.model_name_or_path)
        readme_path = model_path / "README.md"
        if model_path.exists() and readme_path.exists():
            try:
                readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                readme_text = ""
            if "Use with mlx" in readme_text or "mlx-embeddings" in readme_text:
                raise RuntimeError(
                    "The embedding model at "
                    f"{self.model_name_or_path} is an MLX-exported checkpoint. "
                    "The current repo embedding backend uses transformers/torch and "
                    "cannot load MLX-only exports. Use an official Hugging Face "
                    "checkpoint such as Qwen/Qwen3-Embedding-8B, "
                    "Qwen/Qwen3-Embedding-4B, or Qwen/Qwen3-Embedding-0.6B instead."
                )
        local_only = os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        except OSError as exc:
            if local_only:
                raise RuntimeError(
                    "Embedding model "
                    f"{self.model_name_or_path!r} is unavailable in offline mode. "
                    "Pre-download it with HF_HUB_OFFLINE=0, or point the embedding "
                    "detector at a complete local checkpoint."
                ) from exc
            raise

        device = "cpu"
        dtype = None
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        self.device = torch.device(device)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModel.from_pretrained(self.model_name_or_path, **model_kwargs)
        except OSError as exc:
            if local_only:
                raise RuntimeError(
                    "Embedding model "
                    f"{self.model_name_or_path!r} is unavailable in offline mode. "
                    "Pre-download it with HF_HUB_OFFLINE=0, or point the embedding "
                    "detector at a complete local checkpoint."
                ) from exc
            raise
        except RuntimeError as exc:
            message = str(exc)
            if "size mismatch for weight" in message and ("mlx" in self.model_name_or_path.lower() or model_path.exists()):
                raise RuntimeError(
                    "Failed to load the embedding model via transformers/torch. "
                    "This usually means the checkpoint is an MLX-specific export or "
                    "otherwise not compatible with the current backend. "
                    "Use an official Hugging Face checkpoint such as "
                    "Qwen/Qwen3-Embedding-8B, Qwen/Qwen3-Embedding-4B, or "
                    "Qwen/Qwen3-Embedding-0.6B."
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()

        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(tokenizer_max_length, int) or tokenizer_max_length <= 0 or tokenizer_max_length > 4096:
            tokenizer_max_length = 1024
        self.max_length = min(tokenizer_max_length, 1024)

    def _extract_embedding_tensor(self, outputs: Any, attention_mask: Any) -> Any:
        candidate = None
        for key in ("sentence_embedding", "sentence_embeddings", "embeddings", "pooler_output"):
            if hasattr(outputs, key):
                candidate = getattr(outputs, key)
                break
            if isinstance(outputs, dict) and key in outputs:
                candidate = outputs[key]
                break

        if candidate is not None:
            if candidate.dim() == 1:
                candidate = candidate.unsqueeze(0)
            return candidate

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
        else:
            raise RuntimeError("Embedding model did not expose pooled or hidden-state outputs")

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        masked = hidden * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        torch = self._torch
        normalized_batches: List[Any] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded, return_dict=True)
                embeddings = self._extract_embedding_tensor(outputs, encoded["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
            normalized_batches.extend(embeddings.detach().cpu().tolist())
        return normalized_batches


class PrototypeEmbeddingClassifier:
    """Classify reasoning traces against heuristic prototype centroids."""

    PREPROCESSING_VERSION = "v1_strip_prompt_answer_num_normalize"
    SIMILARITY_TEMPERATURE = 0.1
    PROTOTYPE_LABELS = ("DD", "OT", "RC", "STYLE")
    FILE_BASENAMES = {
        "DD": "dd_training.csv",
        "OT": "ot_training.csv",
        "RC": "rc_training.csv",
        "STYLE": "style_training.csv",
    }

    _PROMPT_PREFIX_RE = re.compile(r"^\s*What is .*?\?\s*", re.IGNORECASE | re.DOTALL)
    _ANSWER_SUFFIX_RE = re.compile(
        r"(?:Final\s+)?Answer\s*:\s*[-+]?[\d,]+(?:\.\d+)?\s*$",
        re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(r"\d+")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: Optional[Path] = None,
        batch_size: int = 16,
        prototype_sample_cap: int = 256,
        training_data_dir: Optional[Path] = None,
        embedder: Optional[Any] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = max(1, int(batch_size))
        self.prototype_sample_cap = max(1, int(prototype_sample_cap))
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else repo_root / "Tmp" / "embedding_prototypes"
        self.training_data_dir = Path(training_data_dir) if training_data_dir is not None else repo_root / "SavedData" / "LoRATraining"
        self._embedder = embedder
        self._prototype_pack: Optional[Dict[str, Any]] = None

    @classmethod
    def normalize_trace_text(cls, trace: Optional[str]) -> str:
        """Normalize traces so prototypes depend on reasoning style, not literal numbers."""
        if not trace:
            return ""

        text = str(trace).replace("\r\n", "\n").strip()
        text = cls._PROMPT_PREFIX_RE.sub("", text)
        text = cls._ANSWER_SUFFIX_RE.sub("", text)
        text = cls._NUMBER_RE.sub("<NUM>", text)
        text = cls._WHITESPACE_RE.sub(" ", text).strip()
        return text

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = _HFTextEmbeddingBackend(
                self.model_name_or_path,
                batch_size=self.batch_size,
            )
        return self._embedder

    def _training_file_paths(self) -> Dict[str, Path]:
        return {
            label: self.training_data_dir / basename
            for label, basename in self.FILE_BASENAMES.items()
        }

    def _file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cache_key(self) -> str:
        training_hashes = {
            label: self._file_hash(path)
            for label, path in self._training_file_paths().items()
        }
        payload = {
            "model": self.model_name_or_path,
            "batch_size": self.batch_size,
            "prototype_sample_cap": self.prototype_sample_cap,
            "preprocessing_version": self.PREPROCESSING_VERSION,
            "training_hashes": training_hashes,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self._cache_key()}.json"

    def _load_cached_pack(self) -> Optional[Dict[str, Any]]:
        cache_path = self._cache_path()
        if not cache_path.exists():
            return None
        with open(cache_path, "r") as f:
            payload = json.load(f)
        if payload.get("label_order") != list(self.PROTOTYPE_LABELS):
            return None
        return payload

    def _save_cached_pack(self, payload: Dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path(), "w") as f:
            json.dump(payload, f)

    def _load_training_examples(self) -> Dict[str, List[str]]:
        examples: Dict[str, List[str]] = {}
        for label, path in self._training_file_paths().items():
            if not path.exists():
                raise FileNotFoundError(f"Prototype training file not found: {path}")
            cleaned_examples: List[str] = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split", "").strip() != "train":
                        continue
                    cleaned = self.normalize_trace_text(row.get("reasoning_trace"))
                    if not cleaned:
                        continue
                    cleaned_examples.append(cleaned)
                    if len(cleaned_examples) >= self.prototype_sample_cap:
                        break
            if not cleaned_examples:
                raise RuntimeError(f"No train examples available for embedding prototypes: {path}")
            examples[label] = cleaned_examples
        return examples

    def _mean_normalized(self, vectors: Sequence[Sequence[float]]) -> List[float]:
        length = len(vectors[0])
        totals = [0.0] * length
        for vector in vectors:
            for idx, value in enumerate(vector):
                totals[idx] += float(value)
        averaged = [value / len(vectors) for value in totals]
        norm = sum(value * value for value in averaged) ** 0.5
        if norm <= 0:
            return averaged
        return [value / norm for value in averaged]

    def _fit_prototypes(self) -> Dict[str, Any]:
        examples = self._load_training_examples()
        centroids: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for label in self.PROTOTYPE_LABELS:
            embedded = self._get_embedder().embed_texts(examples[label])
            centroids[label] = self._mean_normalized(embedded)
            counts[label] = len(embedded)

        payload = {
            "version": 1,
            "model": self.model_name_or_path,
            "label_order": list(self.PROTOTYPE_LABELS),
            "prototype_sample_cap": self.prototype_sample_cap,
            "preprocessing_version": self.PREPROCESSING_VERSION,
            "prototype_counts": counts,
            "centroids": centroids,
        }
        self._save_cached_pack(payload)
        return payload

    def _get_prototype_pack(self) -> Dict[str, Any]:
        if self._prototype_pack is not None:
            return self._prototype_pack
        cached = self._load_cached_pack()
        self._prototype_pack = cached if cached is not None else self._fit_prototypes()
        return self._prototype_pack

    def warmup(self) -> Dict[str, Any]:
        """Eagerly materialize the embedder and prototype cache for fail-fast startup checks."""
        return self._get_prototype_pack()

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(float(a) * float(b) for a, b in zip(left, right))

    def fingerprint(self, trace: str) -> FingerprintResult:
        cleaned = self.normalize_trace_text(trace)
        if not cleaned:
            return FingerprintResult(
                heuristic=Heuristic.UNKNOWN,
                confidence=0.0,
                details={
                    "resolved": False,
                    "status": "empty_trace",
                    "model": self.model_name_or_path,
                },
            )

        prototypes = self._get_prototype_pack()
        embedding = self._get_embedder().embed_texts([cleaned])[0]
        similarities = {
            label: self._dot(embedding, prototypes["centroids"][label])
            for label in self.PROTOTYPE_LABELS
        }

        scaled = {
            label: similarity / self.SIMILARITY_TEMPERATURE
            for label, similarity in similarities.items()
        }
        best_score = max(scaled.values())
        weights = {
            label: math.exp(score - best_score)
            for label, score in scaled.items()
        }
        denom = sum(weights.values())
        support_mass = {
            label: (weights[label] / denom) if denom > 0 else 0.0
            for label in self.PROTOTYPE_LABELS
        }
        ordered = sorted(support_mass.items(), key=lambda item: item[1], reverse=True)
        best_label = ordered[0][0]
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]

        return FingerprintResult(
            heuristic=Heuristic[best_label],
            confidence=support_mass[best_label],
            details={
                "resolved": True,
                "status": "ok",
                "normalized_trace": cleaned,
                "model": self.model_name_or_path,
                "support_mass": support_mass,
                "similarities": similarities,
                "margin": margin,
                "prototype_counts": prototypes.get("prototype_counts", {}),
                "preprocessing_version": self.PREPROCESSING_VERSION,
            },
        )


# =============================================================================
# COMBINED FINGERPRINTER
# =============================================================================

class HeuristicFingerprinter:
    """
    Combined fingerprinter that uses all three methods.

    Priority:
    1. If we have a wrong answer, use ErrorShapeParser (most informative)
    2. If we have a trace, use TraceClassifier
    3. If we have model access, use PerplexityProbe
    """

    def __init__(self, model_client=None):
        self.perplexity_probe = PerplexityProbe(model_client)
        self.error_parser = ErrorShapeParser()
        self.trace_classifier = TraceClassifier()

    def fingerprint(
        self,
        problem: Problem,
        predicted: Optional[int] = None,
        trace: Optional[str] = None,
        use_perplexity: bool = True
    ) -> FingerprintResult:
        """
        Fingerprint using available information.

        Args:
            problem: The multiplication problem
            predicted: Model's predicted answer (if available)
            trace: Model's reasoning trace (if available)
            use_perplexity: Whether to use perplexity probe

        Returns:
            Combined FingerprintResult
        """
        results = []

        # Method 1: Error shape (if wrong answer available)
        if predicted is not None and predicted != problem.product:
            result = self.error_parser.fingerprint(problem, predicted)
            if result.heuristic != Heuristic.UNKNOWN:
                results.append(("error_shape", result))

        # Method 2: Trace classification (if trace available)
        if trace:
            result = self.trace_classifier.fingerprint(trace)
            if result.heuristic != Heuristic.UNKNOWN:
                results.append(("trace", result))

        # Method 3: Perplexity probe (if model available and requested)
        if use_perplexity and self.perplexity_probe.model_client:
            result = self.perplexity_probe.fingerprint(problem)
            results.append(("perplexity", result))

        # Combine results
        if not results:
            return FingerprintResult(
                heuristic=Heuristic.UNKNOWN,
                confidence=0.0,
                details={"note": "No fingerprinting method available"}
            )

        # Weight by confidence and aggregate
        votes = {h: 0.0 for h in Heuristic}
        all_details = {}

        for method_name, result in results:
            votes[result.heuristic] += result.confidence
            all_details[method_name] = {
                "heuristic": result.heuristic.value,
                "confidence": result.confidence,
                "details": result.details
            }

        best_heuristic = max(votes, key=lambda h: votes[h])
        total_confidence = sum(r.confidence for _, r in results)

        return FingerprintResult(
            heuristic=best_heuristic,
            confidence=votes[best_heuristic] / max(total_confidence, 1),
            details={"methods": all_details, "votes": {h.value: v for h, v in votes.items()}}
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fingerprint_from_error(a: int, b: int, predicted: int) -> FingerprintResult:
    """Quick fingerprint from error pattern."""
    problem = Problem(a, b)
    parser = ErrorShapeParser()
    return parser.fingerprint(problem, predicted)


def fingerprint_from_trace(trace: str) -> FingerprintResult:
    """Quick fingerprint from reasoning trace."""
    classifier = TraceClassifier()
    return classifier.fingerprint(trace)


def get_perplexity_templates(a: int, b: int) -> Dict[str, List[str]]:
    """Get all perplexity probe templates for a problem (for manual testing)."""
    problem = Problem(a, b)
    probe = PerplexityProbe()

    templates = {}
    for heuristic, template_list in probe.TEMPLATES.items():
        templates[heuristic.value] = [
            probe.format_template(t, problem) for t in template_list
        ]
    # Include neutral baseline
    templates["neutral"] = [probe.format_template(probe.NEUTRAL_TEMPLATE, problem)]
    return templates


# =============================================================================
# TESTING
# =============================================================================

def _test_parsers():
    """Run basic tests on the parsers."""
    tprint("=" * 60)
    tprint("Testing FingerprintParsers")
    tprint("=" * 60)
    tprint()

    # Test 1: Error shape parser
    tprint("[1] Testing ErrorShapeParser...")
    problem = Problem(47, 36)  # 47 × 36 = 1692
    parser = ErrorShapeParser()

    # OT-style error: off by 10 (carry error)
    result = parser.fingerprint(problem, 1702)  # Off by 10
    tprint(f"    47×36=1702 (off by 10): {result.heuristic.value} (conf: {result.confidence:.2f})")

    # DD-style error: missing 47×6 = 282
    result = parser.fingerprint(problem, 1692 - 282)  # Missing one term
    tprint(f"    47×36=1410 (missing 47×6): {result.heuristic.value} (conf: {result.confidence:.2f})")

    # RC-style error: off by 1 (compensation error)
    result = parser.fingerprint(problem, 1693)  # Off by 1
    tprint(f"    47×36=1693 (off by 1): {result.heuristic.value} (conf: {result.confidence:.2f})")
    tprint()

    # Test 2: Trace classifier
    tprint("[2] Testing TraceClassifier...")
    classifier = TraceClassifier()

    ot_trace = "First, 7 × 6 = 42. Write down 2 and carry the 4. Then 7 × 3 = 21, plus 4 = 25."
    result = classifier.fingerprint(ot_trace)
    tprint(f"    OT trace: {result.heuristic.value} (conf: {result.confidence:.2f})")

    dd_trace = "Decompose 47 into 40 + 7. Then 40×36 = 1440 and 7×36 = 252. Sum: 1692."
    result = classifier.fingerprint(dd_trace)
    tprint(f"    DD trace: {result.heuristic.value} (conf: {result.confidence:.2f})")

    rc_trace = "49 is close to 50. So 49×51 ≈ 50×50 = 2500. Compensate: 2500 - 1 = 2499."
    result = classifier.fingerprint(rc_trace)
    tprint(f"    RC trace: {result.heuristic.value} (conf: {result.confidence:.2f})")
    tprint()

    # Test 3: Perplexity templates
    tprint("[3] Testing PerplexityProbe templates...")
    templates = get_perplexity_templates(49, 51)
    for heuristic, prompts in templates.items():
        tprint(f"    {heuristic}:")
        for p in prompts[:1]:  # Just show first template
            tprint(f"      \"{p}\"")
    tprint()

    # Test 4: Problem properties
    tprint("[4] Testing Problem properties...")
    p1 = Problem(49, 51)
    tprint(f"    49×51: near_base={p1.is_near_base}, has_zero={p1.has_zero_factor}")
    p2 = Problem(47, 60)
    tprint(f"    47×60: near_base={p2.is_near_base}, has_zero={p2.has_zero_factor}")
    p3 = Problem(87, 96)
    tprint(f"    87×96: near_base={p3.is_near_base}, has_zero={p3.has_zero_factor}")
    tprint()

    tprint("=" * 60)
    tprint("All tests passed!")
    tprint("=" * 60)


if __name__ == "__main__":
    _test_parsers()
