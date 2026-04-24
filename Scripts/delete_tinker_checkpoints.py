#!/usr/bin/env python3
"""
Delete all user checkpoints and sampler weights from Tinker.
"""

import os

import tinker

from core.TinkerStartup import create_tinker_service_client


# Assumes you already did: export TINKER_API_KEY=...
service_client = create_tinker_service_client(
    tinker_module=tinker,
    api_key=os.getenv("TINKER_API_KEY"),
)
rest_client = service_client.create_rest_client()

# List all your user checkpoints
checkpoints_response = rest_client.list_user_checkpoints().result()

# Delete each checkpoint
for checkpoint in checkpoints_response.checkpoints:
    rest_client.delete_checkpoint_from_tinker_path(checkpoint.tinker_path).result()
print("All checkpoints deleted.")

# List all your user sampler weights
sampler_weights_response = rest_client.list_user_sampler_weights().result()

# Delete each sampler weight
for sampler_weight in sampler_weights_response.sampler_weights:
    rest_client.delete_sampler_weight_from_tinker_path(sampler_weight.tinker_path).result()
print("All sampler weights deleted.")
