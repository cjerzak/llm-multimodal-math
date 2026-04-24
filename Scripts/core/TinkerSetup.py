#!/usr/bin/env python3
"""
TinkerSetup.py

Validates Tinker API connection and basic functionality.
Run this script first to confirm the environment is properly configured.
"""

import os
import sys
from pathlib import Path

# Add Scripts to path for imports when run directly
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from core.Logging import tprint
from core.TinkerStartup import (
    create_tinker_service_client,
    format_tinker_startup_config,
    load_tinker_startup_config,
)


def validate_api_key():
    """Check that TINKER_API_KEY is set."""
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        tprint("ERROR: TINKER_API_KEY environment variable not set")
        tprint("Please set it in your shell or .zshrc")
        return False
    tprint(f"✓ TINKER_API_KEY found (starts with: {api_key[:10]}...)")
    return True


def test_import():
    """Test that tinker can be imported."""
    try:
        import tinker
        tprint(f"✓ tinker package imported successfully (version: {getattr(tinker, '__version__', 'unknown')})")
        return True
    except ImportError as e:
        tprint(f"ERROR: Failed to import tinker: {e}")
        return False


def test_service_client():
    """Test ServiceClient creation."""
    try:
        import tinker
        service_client = create_tinker_service_client(
            tinker_module=tinker,
            config=load_tinker_startup_config(),
        )
        tprint("✓ ServiceClient created successfully")
        return service_client
    except Exception as e:
        tprint(f"ERROR: Failed to create ServiceClient: {e}")
        return None


def test_lora_client(service_client):
    """Test LoRA training client creation."""
    if service_client is None:
        tprint("SKIP: Cannot test LoRA client without ServiceClient")
        return False

    try:
        # Use smaller model for quick testing
        training_client = service_client.create_lora_training_client(
            base_model="meta-llama/Llama-3.2-3B",
            rank=32
        )
        tprint("✓ LoRA training client created successfully (Llama-3.2-3B, rank=32)")
        return training_client
    except Exception as e:
        tprint(f"ERROR: Failed to create LoRA training client: {e}")
        return None


def main():
    """Run all validation tests."""
    tprint("=" * 60)
    tprint("Tinker API Validation")
    tprint("=" * 60)
    tprint()

    # Step 1: Check API key
    tprint("[1/4] Checking API key...")
    if not validate_api_key():
        sys.exit(1)
    tprint()

    # Step 2: Test import
    tprint("[2/4] Testing tinker import...")
    if not test_import():
        sys.exit(1)
    tprint()

    # Step 3: Test ServiceClient
    tprint("[3/4] Testing ServiceClient creation...")
    tprint(f"      Startup settings: {format_tinker_startup_config(load_tinker_startup_config())}")
    service_client = test_service_client()
    tprint()

    # Step 4: Test LoRA client (optional - may take time/cost)
    tprint("[4/4] Testing LoRA training client creation...")
    tprint("      (This may take a moment and use API credits)")
    training_client = test_lora_client(service_client)
    tprint()

    # Summary
    tprint("=" * 60)
    if training_client:
        tprint("SUCCESS: All Tinker API tests passed!")
        tprint("You are ready to proceed with the experiments.")
    elif service_client:
        tprint("PARTIAL SUCCESS: ServiceClient works but LoRA client failed.")
        tprint("Check your API quota and model availability.")
    else:
        tprint("FAILED: Tinker API validation failed.")
        tprint("Check your API key and network connection.")
    tprint("=" * 60)


if __name__ == "__main__":
    main()
