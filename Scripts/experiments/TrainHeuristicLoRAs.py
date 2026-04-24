#!/usr/bin/env python3
"""
TrainHeuristicLoRAs.py

Train heuristic-specific LoRA adapters (RC, DD, OT, STYLE) using Tinker API.

Usage:
    conda activate tm_env
    TINKER_API_KEY="..." python Scripts/TrainHeuristicLoRAs.py
    TINKER_API_KEY="..." python Scripts/TrainHeuristicLoRAs.py --early-stop-patience 3

This script:
1. Loads training data for each heuristic (train/val splits)
2. Trains a LoRA adapter for each heuristic with validation
3. Implements early stopping based on validation loss
4. Saves trained adapters for later use
5. Logs training metrics including validation performance
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Using VLM for both text and image to enable apples-to-apples cross-modal comparison
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LORA_RANK = 32
WARMUP_FRACTION = 0.05
MIN_WARMUP_STEPS = 10
MAX_WARMUP_STEPS = 100
NUM_EPOCHS = 3
EARLY_STOP_PATIENCE = 2  # Stop if val loss doesn't improve for N epochs
VAL_EVAL_INTERVAL = 10   # Evaluate on validation set every N steps
MIN_TRANSFORMERS_VERSION = "4.57.6"
# =============================================================================

import os
import sys
import csv
import json
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Paths (experiments/ -> Scripts/ -> repo root)
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
COOKBOOK_DIR = REPO_ROOT / "tinker-cookbook"
TRAINING_DATA_DIR = REPO_ROOT / "SavedData" / "LoRATraining"
OUTPUT_DIR = REPO_ROOT / "SavedResults" / "lora_training"
WEIGHTS_DIR = REPO_ROOT / "SavedResults" / "gradient_analysis" / "adapter_weights"


def get_weights_dir(model_name: Optional[str] = None) -> Path:
    """Get model-specific weights directory.

    Args:
        model_name: Full model name (e.g., "Qwen/Qwen3-VL-30B-A3B-Instruct").
                    If None, uses legacy path without model suffix.

    Returns:
        Path to weights directory for this model.
    """
    if model_name:
        model_slug = model_name.split("/")[-1].replace("-Instruct", "")
        return REPO_ROOT / "SavedResults" / f"gradient_analysis_{model_slug}" / "adapter_weights"
    return WEIGHTS_DIR  # Legacy fallback


def get_training_output_dir(
    model_name: str,
    modality: str = "text",
    seed_control: bool = False,
    seed: Optional[int] = None,
) -> Path:
    """Get model-specific output directory for LoRA training logs/artifacts."""
    model_slug = model_name.split("/")[-1].replace("-Instruct", "")
    modality_suffix = f"_{modality}" if modality != "text" else ""
    if seed_control:
        if seed is None:
            raise ValueError("seed_control output requires an explicit seed")
        return REPO_ROOT / "SavedResults" / f"lora_training{modality_suffix}_seed{seed}_{model_slug}"
    return REPO_ROOT / "SavedResults" / f"lora_training{modality_suffix}_{model_slug}"


# Add Scripts to path for imports when run directly
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(COOKBOOK_DIR))

from core.TinkerClient import (
    build_chat_prompt_response_tokens,
    build_tinker_sampling_retry_config,
    compute_weighted_loss,
)
from core.Logging import tprint
from core.TinkerStartup import create_tinker_service_client
from tinker_cookbook.hyperparam_utils import get_lr as get_recommended_lr


def download_adapter_weights(checkpoint_path: str, output_path: Path, verbose: bool = True) -> bool:
    """Download adapter weights from Tinker checkpoint.

    Args:
        checkpoint_path: Tinker checkpoint path (e.g., tinker://user/.../checkpoint-name)
        output_path: Local path to save extracted weights (.npz file)
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    import tarfile
    import tempfile
    import urllib.request
    import numpy as np

    try:
        import tinker
        service_client = create_tinker_service_client(
            tinker_module=tinker,
            api_key=os.getenv("TINKER_API_KEY"),
        )
        rest_client = service_client.create_rest_client()

        if verbose:
            tprint(f"    Getting download URL for checkpoint...")

        # Get signed download URL
        resp = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download tar archive
            tar_path = Path(tmpdir) / "checkpoint.tar"
            if verbose:
                tprint(f"    Downloading checkpoint archive...")
            urllib.request.urlretrieve(resp.url, tar_path)

            # Extract
            if verbose:
                tprint(f"    Extracting archive...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(tmpdir)

            # Look for adapter weights (safetensors or bin)
            weights_file = None
            for pattern in ['adapter_model.safetensors', 'adapter_model.bin', '*.safetensors']:
                matches = list(Path(tmpdir).rglob(pattern))
                if matches:
                    weights_file = matches[0]
                    break

            if weights_file is None:
                if verbose:
                    tprint(f"    Warning: No adapter weights found in checkpoint")
                    tprint(f"    Archive contents: {list(Path(tmpdir).rglob('*'))}")
                return False

            # Load and convert to numpy
            if weights_file.suffix == '.safetensors':
                from safetensors import safe_open
                weights = {}
                with safe_open(weights_file, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            else:
                import torch
                state_dict = torch.load(weights_file, map_location='cpu')
                weights = {k: v.numpy() for k, v in state_dict.items()}

            # Save as npz
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **weights)
            if verbose:
                total_params = sum(w.size for w in weights.values())
                tprint(f"    Saved {len(weights)} weight tensors ({total_params:,} params) to {output_path}")
            return True

    except Exception as e:
        if verbose:
            tprint(f"    Warning: Failed to download adapter weights: {e}")
        return False


@dataclass
class TrainingExample:
    """A training example with prompt and reasoning trace."""
    id: str
    a: int
    b: int
    product: int
    heuristic: str
    prompt: str
    reasoning_trace: str
    full_text: str


@dataclass
class TrainingMetrics:
    """Metrics for a single training run."""
    heuristic: str
    num_examples: int
    num_val_examples: int
    num_epochs: int
    total_steps: int
    final_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    losses: List[float]
    val_losses: List[float]
    early_stopped: bool
    elapsed_seconds: float
    model_name: str
    lora_rank: int
    learning_rate: float
    recommended_learning_rate: float
    learning_rate_source: str
    adapter_path: Optional[str]
    state_path: Optional[str]


def load_training_data(heuristic: str, split: str = "all") -> List[TrainingExample]:
    """Load training data for a specific heuristic.

    Args:
        heuristic: Heuristic name (RC, DD, OT, STYLE)
        split: Split to load ("train", "val", or "all")

    Returns:
        List of TrainingExample objects.
    """
    path = TRAINING_DATA_DIR / f"{heuristic.lower()}_training.csv"

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    examples = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by split if column exists
            if split != "all" and 'split' in row:
                if row['split'] != split:
                    continue

            examples.append(TrainingExample(
                id=row['id'],
                a=int(row['a']),
                b=int(row['b']),
                product=int(row['product']),
                heuristic=row['heuristic'],
                prompt=row['prompt'],
                reasoning_trace=row['reasoning_trace'],
                full_text=row['full_text']
            ))

    return examples


def _strip_prompt_from_trace(prompt: str, trace: str) -> str:
    """Remove leading prompt line from a reasoning trace if present."""
    prompt_clean = prompt.strip()
    lines = trace.splitlines()
    if lines and lines[0].strip() == prompt_clean:
        return "\n".join(lines[1:]).lstrip()
    return trace.lstrip()


def _should_use_chat_format(model_name: str) -> bool:
    """Heuristic to enable chat formatting for instruct-style models."""
    return "Qwen3" in model_name and "Instruct" in model_name


def _compute_warmup_steps(total_steps: int) -> int:
    """Compute warmup steps as a short fraction of total optimizer steps."""
    if total_steps <= 0:
        return 0
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    warmup_steps = max(MIN_WARMUP_STEPS, warmup_steps)
    warmup_steps = min(MAX_WARMUP_STEPS, warmup_steps, total_steps)
    return warmup_steps


def _get_warmup_lr(step: int, warmup_steps: int, base_learning_rate: float) -> float:
    """Linear warmup to the requested base learning rate."""
    if warmup_steps <= 0:
        return base_learning_rate
    if step <= warmup_steps:
        return base_learning_rate * (step / warmup_steps)
    return base_learning_rate


def _resolve_recommended_learning_rate(model_name: str) -> float:
    """Resolve the recommended LoRA learning rate from the vendored cookbook."""
    try:
        return float(get_recommended_lr(model_name))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve recommended learning rate for {model_name!r} via "
            "tinker_cookbook.hyperparam_utils.get_lr()"
        ) from exc


def _validate_qwen3_vl_transformers_support(model_name: str) -> None:
    """Fail fast with a clear env error if transformers is too old for Qwen3-VL MoE."""
    try:
        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        transformers_version = "unknown"

    if "qwen3_vl_moe" not in CONFIG_MAPPING:
        raise RuntimeError(
            f"Installed transformers {transformers_version} does not support Qwen3-VL MoE "
            "(missing config mapping for 'qwen3_vl_moe'). Refresh tm_env with "
            f"transformers>={MIN_TRANSFORMERS_VERSION},<5 before training {model_name!r}."
        )

    try:
        AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Installed transformers {transformers_version} cannot load {model_name!r} via "
            "AutoConfig.from_pretrained(). Refresh tm_env with "
            f"transformers>={MIN_TRANSFORMERS_VERSION},<5."
        ) from exc


def pre_tokenize_examples(
    examples: List[TrainingExample],
    tokenizer,
    max_workers: int = 4,
    use_chat_format: bool = False
) -> Dict[str, tuple]:
    """Pre-tokenize all examples in parallel using thread pool.

    Args:
        examples: List of training examples
        tokenizer: HuggingFace tokenizer
        max_workers: Number of threads for parallel tokenization
        use_chat_format: Whether to wrap prompt/response in chat template

    Returns:
        Dict mapping example.id -> (input_tokens, target_tokens, weights)
    """
    def tokenize_one(example):
        if use_chat_format:
            prompt = example.prompt
            response = _strip_prompt_from_trace(prompt, example.reasoning_trace)
            input_tokens, target_tokens, weights = build_chat_prompt_response_tokens(
                tokenizer, prompt, response
            )
            if not input_tokens:
                return (example.id, [], [], [])
            return (example.id, input_tokens, target_tokens, weights)

        tokens = tokenizer.encode(example.full_text, add_special_tokens=True)
        if len(tokens) < 2:
            return (example.id, [], [], [])
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [1] * len(target_tokens)
        return (example.id, input_tokens, target_tokens, weights)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(tokenize_one, examples))

    return {r[0]: (r[1], r[2], r[3]) for r in results}


def create_batch_data(
    examples: List[TrainingExample],
    tokenized: Dict[str, tuple],
    tinker_module
) -> List:
    """Create batch of Datum objects for forward_backward().

    Args:
        examples: List of training examples to batch
        tokenized: Pre-tokenized data dict from pre_tokenize_examples()
        tinker_module: The tinker module for types

    Returns:
        List of Datum objects ready for forward_backward()
    """
    import numpy as np

    data = []
    for example in examples:
        input_tokens, target_tokens, weights = tokenized[example.id]
        if not input_tokens:  # Skip empty tokenizations
            continue

        model_input = tinker_module.types.ModelInput.from_ints(tokens=input_tokens)
        datum = tinker_module.types.Datum(
            model_input=model_input,
            loss_fn_inputs=dict(
                weights=tinker_module.types.TensorData.from_numpy(
                    np.array(weights, dtype=np.float32)
                ),
                target_tokens=tinker_module.types.TensorData.from_numpy(
                    np.array(target_tokens, dtype=np.int64)
                )
            )
        )
        data.append(datum)
    return data


class LoRATrainer:
    """Trainer for heuristic-specific LoRA adapters using Tinker API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize trainer with Tinker API."""
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError("TINKER_API_KEY not set")

        import tinker

        # HuggingFace authentication: huggingface_hub auto-detects HF_TOKEN env var,
        # so explicit login() is not needed and can cause conflicts in parallel execution
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            tprint("  HF_TOKEN detected (auto-authenticated)")

        self.tinker = tinker
        self.service_client = create_tinker_service_client(
            tinker_module=tinker,
            api_key=self.api_key,
        )
        self.model_name = DEFAULT_MODEL_NAME  # Can be overridden via set_model()
        self.hf_token = hf_token

        # Store trained adapter paths
        self.adapter_paths: Dict[str, str] = {}

        # Store extracted adapter weights for gradient analysis
        self.adapter_weights: Dict[str, Dict] = {}

    def set_model(self, model_name: str):
        """Set the model to use for training (override default)."""
        self.model_name = model_name

    def _extract_and_save_adapter_weights(
        self,
        training_client,
        heuristic: str,
        verbose: bool = True
    ) -> Optional[Path]:
        """Extract LoRA adapter weights and save them for gradient analysis.

        Args:
            training_client: Tinker training client after training
            heuristic: Heuristic name (RC, DD, OT)
            verbose: Print progress

        Returns:
            Path to saved weights file, or None if extraction failed
        """
        import numpy as np

        try:
            if verbose:
                tprint(f"  Extracting adapter weights for gradient analysis...")

            # Try to get adapter state from Tinker API
            # Note: The exact API method depends on Tinker's interface
            # Common methods: get_state(), get_adapter_weights(), get_lora_weights()

            weights_dict = {}

            # Method 1: Try get_state() if available
            if hasattr(training_client, 'get_state'):
                state = training_client.get_state()
                if state is not None:
                    # Convert to serializable format
                    for key, value in state.items():
                        if hasattr(value, 'numpy'):
                            weights_dict[key] = value.numpy()
                        elif hasattr(value, 'tolist'):
                            weights_dict[key] = np.array(value.tolist())
                        else:
                            weights_dict[key] = np.array(value)

            # Method 2: Try get_adapter_weights() if get_state() didn't work
            if not weights_dict and hasattr(training_client, 'get_adapter_weights'):
                adapter_weights = training_client.get_adapter_weights()
                if adapter_weights is not None:
                    for key, value in adapter_weights.items():
                        if hasattr(value, 'numpy'):
                            weights_dict[key] = value.numpy()
                        elif hasattr(value, 'tolist'):
                            weights_dict[key] = np.array(value.tolist())
                        else:
                            weights_dict[key] = np.array(value)

            # Method 3: Try to access LoRA-specific attributes
            if not weights_dict:
                for attr in ['lora_A', 'lora_B', 'adapter_weights', '_lora_weights']:
                    if hasattr(training_client, attr):
                        weights = getattr(training_client, attr)
                        if weights is not None:
                            if isinstance(weights, dict):
                                for key, value in weights.items():
                                    if hasattr(value, 'numpy'):
                                        weights_dict[key] = value.numpy()
                                    else:
                                        weights_dict[key] = np.array(value)
                            break

            if not weights_dict:
                if verbose:
                    tprint(f"    Warning: Could not extract adapter weights from Tinker API")
                    tprint(f"    Available methods: {[m for m in dir(training_client) if not m.startswith('_')]}")
                return None

            # Save weights to model-specific directory
            weights_dir = get_weights_dir(self.model_name)
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / f"{heuristic.lower()}_adapter_weights.npz"

            np.savez(weights_path, **weights_dict)

            if verbose:
                tprint(f"    Saved {len(weights_dict)} weight tensors to {weights_path}")
                total_params = sum(w.size for w in weights_dict.values())
                tprint(f"    Total parameters: {total_params:,}")

            # Store for later use
            self.adapter_weights[heuristic] = weights_dict

            return weights_path

        except Exception as e:
            if verbose:
                tprint(f"    Warning: Failed to extract adapter weights: {e}")
            return None

    def _compute_validation_loss(
        self,
        client,
        val_examples: List[TrainingExample],
        tokenized: Dict[str, tuple],
        batch_size: int = 1
    ) -> float:
        """Compute average loss on validation set using pre-tokenized data (forward-only).

        Args:
            client: Tinker training client
            val_examples: Validation examples
            tokenized: Pre-tokenized data dict from pre_tokenize_examples()
            batch_size: Batch size for API calls (1 = no batching)

        Returns:
            Average validation loss
        """
        import numpy as np

        if not val_examples:
            return float('nan')

        val_losses = []

        # Process in batches
        for i in range(0, len(val_examples), batch_size):
            batch_examples = val_examples[i:i+batch_size]

            # Filter out examples with empty tokenizations
            valid_batch = []
            batch_weights = []
            for example in batch_examples:
                input_tokens, target_tokens, weights = tokenized[example.id]
                if input_tokens:  # Skip empty
                    valid_batch.append(example)
                    batch_weights.append(weights)

            if not valid_batch:
                continue

            try:
                if batch_size == 1:
                    # Single example (original behavior)
                    example = valid_batch[0]
                    input_tokens, target_tokens, weights = tokenized[example.id]
                    model_input = self.tinker.types.ModelInput.from_ints(tokens=input_tokens)
                    datum = self.tinker.types.Datum(
                        model_input=model_input,
                        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
                    )
                    fb_future = client.forward([datum], "cross_entropy")
                    fb_result = fb_future.result()

                    if hasattr(fb_result, 'loss_fn_outputs') and len(fb_result.loss_fn_outputs) > 0:
                        logprobs_raw = fb_result.loss_fn_outputs[0]['logprobs']
                        loss = compute_weighted_loss(logprobs_raw, weights)
                        val_losses.append(loss)
                else:
                    # Batched validation
                    batch_data = create_batch_data(valid_batch, tokenized, self.tinker)
                    if not batch_data:
                        continue

                    fb_future = client.forward(batch_data, "cross_entropy")
                    fb_result = fb_future.result()

                    if hasattr(fb_result, 'loss_fn_outputs'):
                        for j, example in enumerate(valid_batch):
                            if j < len(fb_result.loss_fn_outputs):
                                logprobs_raw = fb_result.loss_fn_outputs[j]['logprobs']
                                loss = compute_weighted_loss(logprobs_raw, batch_weights[j])
                                val_losses.append(loss)
            except Exception:
                continue

        return np.mean(val_losses) if val_losses else float('nan')

    def train_heuristic(
        self,
        heuristic: str,
        train_examples: List[TrainingExample],
        val_examples: Optional[List[TrainingExample]] = None,
        num_epochs: int = NUM_EPOCHS,
        early_stop_patience: int = EARLY_STOP_PATIENCE,
        learning_rate: float = 0.0,
        recommended_learning_rate: float = 0.0,
        learning_rate_source: str = "cookbook",
        batch_size: int = 1,
        tokenize_workers: int = 4,
        verbose: bool = True,
        seed_suffix: str = ""
    ) -> TrainingMetrics:
        """Train a LoRA adapter for a specific heuristic with validation.

        Args:
            heuristic: Heuristic name (RC, DD, OT)
            train_examples: Training examples
            val_examples: Validation examples (optional)
            num_epochs: Number of training epochs
            early_stop_patience: Stop if val loss doesn't improve for N epochs
            learning_rate: Effective learning rate for training
            recommended_learning_rate: Cookbook-recommended learning rate
            learning_rate_source: Whether LR came from the cookbook or CLI override
            batch_size: Batch size for API calls (1 = no batching)
            tokenize_workers: Number of threads for pre-tokenization
            verbose: Print progress
            seed_suffix: Suffix to add to adapter name (e.g., "_seed42" for orthogonality controls)

        Returns:
            TrainingMetrics with training results
        """
        import numpy as np

        val_examples = val_examples or []

        if verbose:
            tprint(f"\n{'='*60}")
            tprint(f"Training {heuristic} LoRA Adapter")
            tprint(f"{'='*60}")
            tprint(f"  Model: {self.model_name}")
            tprint(f"  LoRA rank: {LORA_RANK}")
            tprint(f"  Recommended learning rate: {recommended_learning_rate}")
            tprint(f"  Effective learning rate: {learning_rate} ({learning_rate_source})")
            tprint(f"  Train examples: {len(train_examples)}")
            tprint(f"  Val examples: {len(val_examples)}")
            tprint(f"  Epochs: {num_epochs}")
            tprint(f"  Early stop patience: {early_stop_patience}")
            tprint(f"  Batch size: {batch_size}")
            tprint(f"  Tokenize workers: {tokenize_workers}")
            tprint()

        # Create training client
        tprint("  Creating LoRA training client...")
        training_client = self.service_client.create_lora_training_client(
            base_model=self.model_name,
            rank=LORA_RANK
        )

        # Get tokenizer
        tprint("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        use_chat_format = _should_use_chat_format(self.model_name)
        if verbose:
            tprint(f"  Chat format: {'ON' if use_chat_format else 'OFF'}")

        # Pre-tokenize all examples in parallel (Phase A optimization)
        all_examples = train_examples + val_examples
        tprint(f"  Pre-tokenizing {len(all_examples)} examples ({tokenize_workers} workers)...")
        tokenize_start = time.time()
        tokenized = pre_tokenize_examples(
            all_examples,
            tokenizer,
            max_workers=tokenize_workers,
            use_chat_format=use_chat_format
        )
        tokenize_elapsed = time.time() - tokenize_start
        tprint(f"  Pre-tokenization complete in {tokenize_elapsed:.2f}s")

        # Training loop with early stopping
        losses = []
        val_losses_history = []
        total_steps = 0
        optim_steps = 0
        start_time = time.time()

        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopped = False
        if batch_size == 1:
            steps_per_epoch = len(train_examples)
        else:
            steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
        total_optim_steps = steps_per_epoch * num_epochs
        warmup_steps = _compute_warmup_steps(total_optim_steps)
        if verbose:
            tprint(f"  Warmup steps: {warmup_steps} (of {total_optim_steps} optim steps)")

        for epoch in range(num_epochs):
            epoch_losses = []
            if verbose:
                tprint(f"\n  Epoch {epoch + 1}/{num_epochs}")
                tprint(f"  {'-'*50}")

            if batch_size == 1:
                # Original single-example training loop
                for i, example in enumerate(train_examples):
                    input_tokens, target_tokens, weights = tokenized[example.id]

                    if not input_tokens:
                        continue

                    # Create datum
                    model_input = self.tinker.types.ModelInput.from_ints(tokens=input_tokens)
                    datum = self.tinker.types.Datum(
                        model_input=model_input,
                        loss_fn_inputs=dict(
                            weights=weights,
                            target_tokens=target_tokens
                        )
                    )

                    try:
                        # Forward-backward pass
                        fb_future = training_client.forward_backward([datum], "cross_entropy")
                        fb_result = fb_future.result()

                        # Extract loss
                        if hasattr(fb_result, 'loss_fn_outputs') and len(fb_result.loss_fn_outputs) > 0:
                            logprobs_raw = fb_result.loss_fn_outputs[0]['logprobs']
                            loss = compute_weighted_loss(logprobs_raw, weights)
                        else:
                            loss = 0.0

                        epoch_losses.append(loss)
                        losses.append(loss)
                        total_steps += 1

                        # Apply optimizer step
                        optim_steps += 1
                        current_lr = _get_warmup_lr(optim_steps, warmup_steps, learning_rate)
                        adam_params = self.tinker.types.AdamParams(
                            learning_rate=current_lr
                        )
                        optim_future = training_client.optim_step(adam_params)
                        optim_future.result()

                        if verbose and (i + 1) % 10 == 0:
                            avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                            tprint(f"    Step {i+1}/{len(train_examples)}: loss={loss:.4f} (avg10={avg_loss:.4f})")

                    except Exception as e:
                        tprint(f"    Warning: Training step failed: {e}")
                        continue
            else:
                # Batched training loop (Phase B optimization)
                num_batches = (len(train_examples) + batch_size - 1) // batch_size
                for batch_idx in range(0, len(train_examples), batch_size):
                    batch_examples = train_examples[batch_idx:batch_idx+batch_size]

                    # Create batch data
                    batch_data = create_batch_data(batch_examples, tokenized, self.tinker)
                    if not batch_data:
                        continue

                    try:
                        # Single API call for entire batch
                        fb_future = training_client.forward_backward(batch_data, "cross_entropy")
                        fb_result = fb_future.result()

                        # Extract losses for each example in batch
                        batch_losses = []
                        if hasattr(fb_result, 'loss_fn_outputs'):
                            for j, example in enumerate(batch_examples):
                                if j < len(fb_result.loss_fn_outputs):
                                    input_tokens, target_tokens, weights = tokenized[example.id]
                                    if input_tokens:
                                        logprobs_raw = fb_result.loss_fn_outputs[j]['logprobs']
                                        loss = compute_weighted_loss(logprobs_raw, weights)
                                        batch_losses.append(loss)
                                        epoch_losses.append(loss)
                                        losses.append(loss)
                                        total_steps += 1

                        # Single optimizer step per batch
                        optim_steps += 1
                        current_lr = _get_warmup_lr(optim_steps, warmup_steps, learning_rate)
                        adam_params = self.tinker.types.AdamParams(
                            learning_rate=current_lr
                        )
                        optim_future = training_client.optim_step(adam_params)
                        optim_future.result()

                        current_batch = batch_idx // batch_size + 1
                        if verbose and current_batch % 5 == 0:
                            avg_loss = np.mean(batch_losses) if batch_losses else 0.0
                            tprint(f"    Batch {current_batch}/{num_batches}: avg_loss={avg_loss:.4f}")

                    except Exception as e:
                        tprint(f"    Warning: Batch training step failed: {e}")
                        continue

            # Epoch summary
            if epoch_losses:
                epoch_avg = sum(epoch_losses) / len(epoch_losses)
                if verbose:
                    tprint(f"  Epoch {epoch+1} train loss: {epoch_avg:.4f}")

            # Compute validation loss (using pre-tokenized data)
            if val_examples:
                val_loss = self._compute_validation_loss(
                    training_client, val_examples, tokenized, batch_size=batch_size
                )
                val_losses_history.append(val_loss)

                if verbose:
                    tprint(f"  Epoch {epoch+1} val loss: {val_loss:.4f}")

                # Check for improvement
                if not np.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    if verbose:
                        tprint(f"    New best validation loss!")
                else:
                    epochs_without_improvement += 1
                    if verbose:
                        tprint(f"    No improvement for {epochs_without_improvement} epoch(s)")

                # Early stopping check
                if epochs_without_improvement >= early_stop_patience:
                    if verbose:
                        tprint(f"\n  Early stopping triggered after {epoch+1} epochs")
                        tprint(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    early_stopped = True
                    break

        elapsed = time.time() - start_time

        # Save trained adapter for both download and inference
        # Include seed_suffix when running seed control experiments
        adapter_name = f"{heuristic.lower()}_lora{seed_suffix}"
        if verbose:
            tprint(f"\n  Saving adapter as '{adapter_name}'...")

        adapter_path = None
        state_path = None
        try:
            # Save full training state for load_state() compatibility
            state_name = f"{adapter_name}_state"
            state_result = training_client.save_state(name=state_name).result()
            state_path = state_result.path
            if verbose:
                tprint(f"  Training state saved to: {state_path}")

            # save_weights_for_sampler() creates a downloadable /sampler_weights/ checkpoint
            # This is required for get_checkpoint_archive_url_from_tinker_path() to work
            save_result = training_client.save_weights_for_sampler(name=adapter_name).result()
            adapter_path = save_result.path
            self.adapter_paths[heuristic] = adapter_path
            if verbose:
                tprint(f"  Sampler weights saved to: {adapter_path}")

            # Also create sampling client for LoRANudgeTest (ephemeral, for inference)
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=adapter_name,
                retry_config=build_tinker_sampling_retry_config(),
            )
        except Exception as e:
            tprint(f"  Warning: Failed to save adapter: {e}")
            # Fallback to just the sampling client
            try:
                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=adapter_name,
                    retry_config=build_tinker_sampling_retry_config(),
                )
                # Note: This path won't be downloadable, but at least inference works
                adapter_path = f"tinker://{self.model_name}/{adapter_name}"
                self.adapter_paths[heuristic] = adapter_path
            except Exception as e2:
                tprint(f"  Warning: Failed to save sampling client: {e2}")

        # Download adapter weights for gradient orthogonality analysis
        weights_path = None
        if adapter_path:
            if verbose:
                tprint(f"  Downloading adapter weights for gradient analysis...")
            # Use model-specific directory and include seed_suffix in weights filename
            weights_dir = get_weights_dir(self.model_name)
            weights_dir.mkdir(parents=True, exist_ok=True)
            output_path = weights_dir / f"{heuristic.lower()}{seed_suffix}_adapter_weights.npz"
            if download_adapter_weights(adapter_path, output_path, verbose):
                weights_path = output_path
            else:
                if verbose:
                    tprint(f"  Will fall back to placeholder mode for gradient analysis")

        # Return metrics
        metrics = TrainingMetrics(
            heuristic=heuristic,
            num_examples=len(train_examples),
            num_val_examples=len(val_examples),
            num_epochs=epoch + 1,  # Actual epochs run
            total_steps=total_steps,
            final_loss=losses[-1] if losses else 0.0,
            final_val_loss=val_losses_history[-1] if val_losses_history else float('nan'),
            best_val_loss=best_val_loss if val_examples else float('nan'),
            best_epoch=best_epoch if val_examples else 0,
            losses=losses,
            val_losses=val_losses_history,
            early_stopped=early_stopped,
            elapsed_seconds=elapsed,
            model_name=self.model_name,
            lora_rank=LORA_RANK,
            learning_rate=learning_rate,
            recommended_learning_rate=recommended_learning_rate,
            learning_rate_source=learning_rate_source,
            adapter_path=adapter_path,
            state_path=state_path
        )

        if verbose:
            tprint(f"\n  Training complete in {elapsed:.1f}s")
            tprint(f"  Final loss: {metrics.final_loss:.4f}")
            tprint(f"  Average loss: {sum(losses)/len(losses):.4f}" if losses else "")
            if val_examples:
                tprint(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
                tprint(f"  Early stopped: {early_stopped}")

        return metrics


def save_training_log(
    metrics_list: List[TrainingMetrics],
    output_dir: Path,
    model_name: Optional[str] = None
):
    """Save training metrics to files."""
    import math
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use model from first metrics entry or parameter
    model_used = model_name or (metrics_list[0].model_name if metrics_list else DEFAULT_MODEL_NAME)
    first_metrics = metrics_list[0] if metrics_list else None

    # Save summary JSON
    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model_used,
        "lora_rank": LORA_RANK,
        "learning_rate": first_metrics.learning_rate if first_metrics else None,
        "effective_learning_rate": first_metrics.learning_rate if first_metrics else None,
        "recommended_learning_rate": (
            first_metrics.recommended_learning_rate if first_metrics else None
        ),
        "learning_rate_source": first_metrics.learning_rate_source if first_metrics else None,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "output_dir": str(output_dir),
        "heuristics": {}
    }

    for m in metrics_list:
        summary["heuristics"][m.heuristic] = {
            "num_examples": m.num_examples,
            "num_val_examples": m.num_val_examples,
            "num_epochs": m.num_epochs,
            "total_steps": m.total_steps,
            "final_loss": m.final_loss,
            "avg_loss": sum(m.losses) / len(m.losses) if m.losses else 0,
            "final_val_loss": m.final_val_loss if not math.isnan(m.final_val_loss) else None,
            "best_val_loss": m.best_val_loss if not math.isnan(m.best_val_loss) else None,
            "best_epoch": m.best_epoch,
            "early_stopped": m.early_stopped,
            "elapsed_seconds": m.elapsed_seconds,
            "learning_rate": m.learning_rate,
            "recommended_learning_rate": m.recommended_learning_rate,
            "learning_rate_source": m.learning_rate_source,
            "adapter_path": m.adapter_path,
            "state_path": m.state_path
        }

    json_path = output_dir / "training_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    tprint(f"Saved training summary to {json_path}")

    # Save loss curves as CSV (train and val)
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["heuristic", "step", "loss"])
        for m in metrics_list:
            for i, loss in enumerate(m.losses):
                writer.writerow([m.heuristic, i + 1, loss])
    tprint(f"Saved training loss curves to {csv_path}")

    # Save validation loss curves
    val_csv_path = output_dir / "validation_log.csv"
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["heuristic", "epoch", "val_loss"])
        for m in metrics_list:
            for epoch, val_loss in enumerate(m.val_losses):
                if not math.isnan(val_loss):
                    writer.writerow([m.heuristic, epoch + 1, val_loss])
    tprint(f"Saved validation loss curves to {val_csv_path}")


def main():
    """Train all configured heuristic LoRAs."""
    import argparse
    import math
    import random
    import numpy as np

    parser = argparse.ArgumentParser(description="Train heuristic-specific LoRA adapters")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model to use (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--heuristics", type=str, nargs="+", default=["RC", "DD", "OT", "STYLE"],
                        help="Adapters to train (default: RC DD OT STYLE)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE,
                        help=f"Early stopping patience (default: {EARLY_STOP_PATIENCE})")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--parallel", action="store_true",
                        help="Train all selected heuristics in parallel")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size for API calls (default: 1, no batching)")
    parser.add_argument("--tokenize-workers", type=int, default=4,
                        help="Number of threads for pre-tokenization (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate. Defaults to tinker_cookbook.hyperparam_utils.get_lr(model)")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "image"],
                        help="Modality for output path differentiation (default: text)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility and orthogonality controls (default: 42)")
    parser.add_argument("--seed-control", action="store_true",
                        help="Train same heuristic with different seeds for orthogonality validation")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Effective patience (very large if disabled)
    patience = 999 if args.no_early_stop else args.early_stop_patience

    # Seed suffix for adapter naming (used when comparing same-heuristic with different seeds)
    seed_suffix = f"_seed{args.seed}" if args.seed_control else ""

    _validate_qwen3_vl_transformers_support(args.model)
    recommended_learning_rate = _resolve_recommended_learning_rate(args.model)
    effective_learning_rate = (
        args.learning_rate if args.learning_rate is not None else recommended_learning_rate
    )
    learning_rate_source = "override" if args.learning_rate is not None else "cookbook"

    tprint("=" * 60)
    tprint("LoRA Heuristic Training")
    tprint("=" * 60)
    tprint(f"Model: {args.model}")
    tprint(f"LoRA Rank: {LORA_RANK}")
    tprint(f"Recommended learning rate: {recommended_learning_rate}")
    tprint(f"Effective learning rate: {effective_learning_rate} ({learning_rate_source})")
    tprint(f"Epochs: {args.epochs}")
    tprint(f"Early stop patience: {patience if not args.no_early_stop else 'DISABLED'}")
    tprint(f"Batch size: {args.batch_size}")
    tprint(f"Tokenize workers: {args.tokenize_workers}")
    tprint(f"Heuristics: {args.heuristics}")
    tprint(f"Parallel training: {args.parallel}")
    tprint(f"Random seed: {args.seed}")
    if args.seed_control:
        tprint(f"Seed control mode: ON (adapters named with seed suffix)")
    tprint()

    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        tprint("ERROR: TINKER_API_KEY not set")
        tprint("Usage: TINKER_API_KEY='...' python Scripts/TrainHeuristicLoRAs.py")
        sys.exit(1)

    # Train each heuristic
    all_metrics = []
    start_time = time.time()

    def train_single_heuristic(heuristic: str, model_name: str, seed_suffix_param: str = "") -> Optional[TrainingMetrics]:
        """Train a single heuristic (for parallel execution)."""
        try:
            # Each thread needs its own trainer instance
            trainer = LoRATrainer(api_key)
            trainer.set_model(model_name)

            # Load training data (train split)
            tprint(f"\nLoading {heuristic} training data...")
            train_examples = load_training_data(heuristic, split="train")
            val_examples = load_training_data(heuristic, split="val")

            # Fallback: if no split column, use all data
            if len(train_examples) == 0:
                tprint(f"  No 'train' split found for {heuristic}, loading all data...")
                train_examples = load_training_data(heuristic, split="all")
                val_examples = []

            tprint(f"  {heuristic} - Train: {len(train_examples)}, Val: {len(val_examples)}")

            if len(train_examples) == 0:
                tprint(f"  WARNING: No training data for {heuristic}, skipping")
                return None

            # Train with validation
            metrics = trainer.train_heuristic(
                heuristic=heuristic,
                train_examples=train_examples,
                val_examples=val_examples,
                num_epochs=args.epochs,
                early_stop_patience=patience,
                learning_rate=effective_learning_rate,
                recommended_learning_rate=recommended_learning_rate,
                learning_rate_source=learning_rate_source,
                batch_size=args.batch_size,
                tokenize_workers=args.tokenize_workers,
                verbose=True,
                seed_suffix=seed_suffix_param
            )
            return metrics

        except Exception as e:
            tprint(f"\nERROR training {heuristic}: {e}")
            import traceback
            traceback.print_exc()
            return None

    if args.parallel and len(args.heuristics) > 1:
        # Parallel training
        tprint(f"\nTraining {len(args.heuristics)} heuristics in parallel...")
        with ThreadPoolExecutor(max_workers=len(args.heuristics)) as executor:
            futures = {
                executor.submit(train_single_heuristic, h, args.model, seed_suffix): h
                for h in args.heuristics
            }
            for future in as_completed(futures):
                heuristic = futures[future]
                try:
                    metrics = future.result()
                    if metrics:
                        all_metrics.append(metrics)
                except Exception as e:
                    tprint(f"ERROR: {heuristic} training failed: {e}")
    else:
        # Sequential training
        tprint("Initializing trainer...")
        trainer = LoRATrainer(api_key)
        trainer.set_model(args.model)

        for heuristic in args.heuristics:
            try:
                # Load training data (train split)
                tprint(f"\nLoading {heuristic} training data...")
                train_examples = load_training_data(heuristic, split="train")
                val_examples = load_training_data(heuristic, split="val")

                # Fallback: if no split column, use all data
                if len(train_examples) == 0:
                    tprint("  No 'train' split found, loading all data...")
                    train_examples = load_training_data(heuristic, split="all")
                    val_examples = []

                tprint(f"  Train examples: {len(train_examples)}")
                tprint(f"  Val examples: {len(val_examples)}")

                if len(train_examples) == 0:
                    tprint(f"  WARNING: No training data for {heuristic}, skipping")
                    continue

                # Train with validation
                metrics = trainer.train_heuristic(
                    heuristic=heuristic,
                    train_examples=train_examples,
                    val_examples=val_examples,
                    num_epochs=args.epochs,
                    early_stop_patience=patience,
                    learning_rate=effective_learning_rate,
                    recommended_learning_rate=recommended_learning_rate,
                    learning_rate_source=learning_rate_source,
                    batch_size=args.batch_size,
                    tokenize_workers=args.tokenize_workers,
                    verbose=True,
                    seed_suffix=seed_suffix
                )
                all_metrics.append(metrics)

            except Exception as e:
                tprint(f"\nERROR training {heuristic}: {e}")
                import traceback
                traceback.print_exc()

    total_time = time.time() - start_time

    # Save results (use model-specific subdirectory)
    tprint("\n" + "=" * 60)
    tprint("Saving Results")
    tprint("=" * 60)
    output_dir = get_training_output_dir(
        args.model,
        modality=args.modality,
        seed_control=args.seed_control,
        seed=args.seed,
    )
    save_training_log(all_metrics, output_dir, model_name=args.model)

    # Print summary
    tprint("\n" + "=" * 60)
    tprint("Training Summary")
    tprint("=" * 60)
    tprint(f"Total time: {total_time:.1f}s")
    tprint()

    for m in all_metrics:
        avg_loss = sum(m.losses) / len(m.losses) if m.losses else 0
        tprint(f"{m.heuristic}:")
        tprint(f"  Train examples: {m.num_examples}")
        tprint(f"  Val examples: {m.num_val_examples}")
        tprint(f"  Epochs: {m.num_epochs}")
        tprint(f"  Steps: {m.total_steps}")
        tprint(f"  Learning rate: {m.learning_rate} ({m.learning_rate_source})")
        tprint(f"  Recommended LR: {m.recommended_learning_rate}")
        tprint(f"  Final train loss: {m.final_loss:.4f}")
        tprint(f"  Avg train loss: {avg_loss:.4f}")
        if not math.isnan(m.best_val_loss):
            tprint(f"  Best val loss: {m.best_val_loss:.4f} (epoch {m.best_epoch})")
            tprint(f"  Early stopped: {m.early_stopped}")
        tprint(f"  Time: {m.elapsed_seconds:.1f}s")
        tprint(f"  Adapter: {m.adapter_path or 'NOT SAVED'}")
        tprint(f"  State: {m.state_path or 'NOT SAVED'}")
        tprint()

    tprint("=" * 60)
    tprint("Done!")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
