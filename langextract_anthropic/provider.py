"""Provider implementation for Anthropic Claude."""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Iterator, Sequence
from typing import Any, Final

import langextract as lx  # type: ignore[import-untyped]

from langextract_anthropic.schema import AnthropicSchema

# Anthropic Messages API supported parameters
# Based on: https://docs.anthropic.com/en/api/messages
_ANTHROPIC_CONFIG_KEYS: Final[set[str]] = {
    'max_tokens',  # Maximum number of tokens to generate (required)
    'temperature',  # Controls randomness (0-1)
    'top_p',  # Nucleus sampling (0-1)
    'top_k',  # Top-k sampling (0 or positive integer)
    'stop_sequences',  # List of strings to stop generation
    'metadata',  # Metadata object for request tracking
    'system',  # System message (alternative to system role in messages)
    'stream',  # Whether to stream responses (unsupported in this implementation)
    'tools',  # Tool use definitions (unsupported in basic implementation)
    'tool_choice',  # Tool choice options (unsupported in basic implementation)
    'service_tier',  # Priority of request processing ("auto", "standard_only")
    'thinking',  # Extended thinking configuration (unsupported in basic implementation)
}


@lx.providers.registry.register(r'^anthropic', priority=10)
class AnthropicLanguageModel(lx.inference.BaseLanguageModel):
    """Language model inference using Anthropic's Claude API with structured output.

    This provider handles model IDs matching: ['^anthropic']
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic language model.

        Args:
            model_id: The Anthropic model ID to use (e.g., 'anthropic-claude-3-5-sonnet-latest').
            api_key: API key for Anthropic service.
            model_name: Claude model name. If None, extracted from model_id.
            temperature: Sampling temperature (0-1).
            max_workers: Maximum number of parallel API calls.
            **kwargs: Additional parameters passed to the Anthropic API.
        """
        # Lazy import: Anthropic package required
        try:
            # pylint: disable=import-outside-toplevel
            from anthropic import Anthropic
        except ImportError as e:
            raise lx.exceptions.InferenceConfigError(
                'Anthropic provider requires anthropic package. '
                'Install with: pip install anthropic>=0.34.0'
            ) from e

        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.temperature = temperature
        self.max_workers = max_workers
        self._response_schema: dict[str, Any] | None = None
        self._enable_structured_output: bool = False

        # Extract model name from model_id if not provided
        if model_name:
            self.model_name = model_name
        else:
            # Extract model name by removing 'anthropic-' prefix
            if isinstance(model_id, str) and model_id.startswith('anthropic-'):
                self.model_name = model_id[len('anthropic-') :]
            else:
                # Default to Claude 3.5 Sonnet if no specific model
                self.model_name = model_id or 'claude-3-5-sonnet-latest'

        # Validate required parameters
        if not self.api_key:
            raise lx.exceptions.InferenceConfigError(
                'Anthropic API key not provided. Set ANTHROPIC_API_KEY '
                'environment variable or pass api_key parameter.'
            )

        # Reject unsupported parameters early if provided at construction
        unsupported = {'stream', 'tools', 'tool_choice', 'thinking'}
        for key in unsupported.intersection(set((kwargs or {}).keys())):
            raise lx.exceptions.InferenceConfigError(
                f"Unsupported parameter provided: {key}. This provider does not support it yet."
            )

        # Initialize the Anthropic client
        self._client = Anthropic(
            api_key=self.api_key,
        )

        # Filter extra kwargs to only include valid Anthropic API parameters
        self._extra_kwargs = {
            k: v for k, v in (kwargs or {}).items() if k in _ANTHROPIC_CONFIG_KEYS
        }

    @classmethod
    def get_schema_class(cls) -> type[AnthropicSchema]:
        """Tell LangExtract about our schema support."""
        return AnthropicSchema

    def apply_schema(self, schema_instance: object | None) -> None:
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance is None:
            self._response_schema = None
            self._enable_structured_output = False
            return
        if isinstance(schema_instance, AnthropicSchema):
            cfg = schema_instance.to_provider_config()
            # Support both keys for compatibility with older schema impls
            self._response_schema = cfg.get('response_schema')
            self._enable_structured_output = bool(
                cfg.get('enable_structured_output') or cfg.get('structured_output')
            )

    def _process_single_prompt(
        self, prompt: str, config: dict[str, Any]
    ) -> lx.inference.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            # Apply stored kwargs that weren't already set in config
            for key, value in self._extra_kwargs.items():
                if key not in config and value is not None:
                    config[key] = value

            # Build messages. When structured output is enabled, include an
            # explicit system instruction for JSON formatting.
            system_message = None
            if self._enable_structured_output:
                system_message = (
                    'You are a helpful assistant that outputs JSON. '
                    'Return only a valid JSON object (no code fences or explanatory text).'
                )

            messages: list[dict[str, str]] = [{'role': 'user', 'content': prompt}]

            api_params: dict[str, Any] = {
                'model': self.model_name,
                'messages': messages,
                'max_tokens': config.get(
                    'max_tokens', 1024
                ),  # Anthropic requires max_tokens
            }

            # Add system message if structured output is enabled
            if system_message:
                api_params['system'] = system_message

            # Only set temperature if explicitly provided
            temp = config.get('temperature', self.temperature)
            if temp is not None:
                api_params['temperature'] = temp

            # Apply standard configuration parameters
            if (v := config.get('top_p')) is not None:
                api_params['top_p'] = v
            if (v := config.get('top_k')) is not None:
                api_params['top_k'] = v

            # Apply Anthropic-specific parameters from whitelist
            # Reject unsupported params if present
            for key in _ANTHROPIC_CONFIG_KEYS:
                if (
                    key in {'stream', 'tools', 'tool_choice', 'thinking'}
                    and key in config
                ):
                    raise lx.exceptions.InferenceConfigError(
                        f"Unsupported parameter provided: {key}. This provider does not support it yet."
                    )
                if key == 'stop_sequences' and (v := config.get(key)) is not None:
                    api_params['stop_sequences'] = v
                elif key == 'metadata' and (v := config.get(key)) is not None:
                    api_params['metadata'] = v
                elif key == 'system' and (v := config.get(key)) is not None:
                    # Override system message if provided in config
                    api_params['system'] = v
                elif key == 'service_tier' and (v := config.get(key)) is not None:
                    api_params['service_tier'] = v

            response = self._client.messages.create(**api_params)

            # Extract the response text from Anthropic's response format
            if response.content and len(response.content) > 0:
                # Anthropic returns a list of content blocks, get the text from the first one
                content_block = response.content[0]
                if hasattr(content_block, 'text'):
                    output_text = content_block.text
                else:
                    output_text = str(content_block)
            else:
                output_text = ""

            return lx.inference.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise lx.exceptions.InferenceRuntimeError(
                f'Anthropic API error: {str(e)}', original=e
            ) from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
        """Runs inference on a list of prompts via Anthropic's API.

        Args:
            batch_prompts: A list of string prompts.
            **kwargs: Additional generation params (temperature, top_p, max_tokens, etc.)

        Yields:
            Lists of ScoredOutputs.
        """
        config: dict[str, Any] = {}

        # Handle standard parameters explicitly
        temp = kwargs.get('temperature', self.temperature)
        if temp is not None:
            config['temperature'] = temp
        if 'max_tokens' in kwargs:
            config['max_tokens'] = kwargs['max_tokens']
        if 'top_p' in kwargs:
            config['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            config['top_k'] = kwargs['top_k']

        # Handle all other whitelisted Anthropic parameters
        handled_keys = {'temperature', 'max_tokens', 'top_p', 'top_k'}
        for key, value in kwargs.items():
            if (
                key not in handled_keys
                and key in _ANTHROPIC_CONFIG_KEYS
                and value is not None
            ):
                # Reject unsupported params at runtime
                if key in {'stream', 'tools', 'tool_choice', 'thinking'}:
                    raise lx.exceptions.InferenceConfigError(
                        f"Unsupported parameter provided: {key}. This provider does not support it yet."
                    )
                config[key] = value

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[lx.inference.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise RuntimeError(f'Parallel inference error: {str(e)}') from e

                for result in results:
                    if result is None:
                        raise RuntimeError('Failed to process one or more prompts')
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]
