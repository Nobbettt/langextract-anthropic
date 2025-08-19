#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple test for the Anthropic provider plugin."""

import os

import langextract as lx

# Import the provider to trigger registration with LangExtract
# Note: This manual import is only needed when running without installation.
# After `pip install -e .`, the entry point system handles this automatically.
from langextract_anthropic import AnthropicLanguageModel  # noqa: F401


def main():
    """Test the Anthropic provider."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key:
        print("Set AZURE_OPENAI_API_KEY to test")
        return

    if not azure_endpoint:
        print("Set AZURE_OPENAI_ENDPOINT to test")
        return
    if not api_version:
        print("Set AZURE_OPENAI_API_VERSION to test")
        return

    config = lx.factory.ModelConfig(
        model_id="azureopenai-gpt-4",
        provider="AzureOpenAILanguageModel",
        provider_kwargs={
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
        },
    )
    model = lx.factory.create_model(config)

    print(f"✓ Created {model.__class__.__name__}")

    # Test inference
    prompts = ["Say hello"]
    results = list(model.infer(prompts))

    if results and results[0]:
        print(f"✓ Inference worked: {results[0][0].output[:50]}...")
    else:
        print("✗ No response")


if __name__ == "__main__":
    main()
