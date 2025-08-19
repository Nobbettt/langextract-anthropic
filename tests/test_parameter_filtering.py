#!/usr/bin/env python3
"""Parameter filtering tests without requiring API credentials.

Validates that:
- Valid parameters are accepted and stored
- Invalid/unsupported parameters are filtered or rejected
- Parameters are correctly passed to the API call structure
"""

import os
from unittest.mock import MagicMock, patch

import langextract as lx
import pytest

from langextract_anthropic import AnthropicLanguageModel
from langextract_anthropic.provider import _ANTHROPIC_CONFIG_KEYS


@pytest.mark.unit
def test_parameter_filtering():
    """Valid params are kept, invalid are dropped; API call receives only allowed values."""
    test_kwargs = {
        # Valid parameters (should be kept)
        'temperature': 0.7,
        'max_tokens': 1000,
        'top_p': 0.9,
        'top_k': 50,
        'stop_sequences': ["\n", "STOP"],
        'metadata': {"user": "test-user"},
        # Invalid parameters (should be dropped)
        'invalid_param': 'should_be_ignored',
        'not_supported': 42,
        'random_key': [1, 2, 3],
    }

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Initialize provider
        provider = AnthropicLanguageModel(api_key='test-key', **test_kwargs)

        # Check filtering in _extra_kwargs
        expected_valid_count = (
            5  # Only valid params from test_kwargs (temperature is special)
        )
        assert len(provider._extra_kwargs) == expected_valid_count

        # Verify valid parameters are stored (temperature is stored separately)
        assert provider.temperature == 0.7  # temperature is stored as instance variable
        assert provider._extra_kwargs['max_tokens'] == 1000
        assert provider._extra_kwargs['top_p'] == 0.9
        assert provider._extra_kwargs['top_k'] == 50
        assert provider._extra_kwargs['stop_sequences'] == ["\n", "STOP"]
        assert provider._extra_kwargs['metadata'] == {"user": "test-user"}

        # Verify invalid parameters are not stored
        assert 'invalid_param' not in provider._extra_kwargs
        assert 'not_supported' not in provider._extra_kwargs
        assert 'random_key' not in provider._extra_kwargs


@pytest.mark.unit
def test_unsupported_parameters_rejected():
    """Unsupported parameters should raise errors immediately during init."""
    unsupported_params = {
        'stream': True,
        'tools': [{"type": "function"}],
        'tool_choice': "auto",
    }

    with patch('anthropic.Anthropic'):
        for param_name, param_value in unsupported_params.items():
            with pytest.raises(
                lx.exceptions.InferenceConfigError,
                match=f"Unsupported parameter provided: {param_name}",
            ):
                AnthropicLanguageModel(api_key='test-key', **{param_name: param_value})


@pytest.mark.unit
def test_api_call_parameter_passing():
    """Test that parameters are correctly passed to Anthropic API call."""

    # Mock the Anthropic client and response
    mock_content_block = MagicMock()
    mock_content_block.text = "Test response"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = mock_client

        # Initialize provider with parameters
        provider = AnthropicLanguageModel(
            api_key='test-key',
            temperature=0.8,
            max_tokens=512,
            top_p=0.95,
            top_k=40,
        )

        # Make a test inference call
        prompts = ["Test prompt"]
        list(
            provider.infer(
                prompts,
                stop_sequences=["STOP", "END"],
                metadata={"request_id": "test-123"},
            )
        )

        # Verify API call was made with correct parameters
        assert mock_client.messages.create.called
        call_args = mock_client.messages.create.call_args.kwargs

        # Check basic parameters
        assert call_args['model'] == 'claude-3-5-sonnet-latest'  # default
        assert call_args['max_tokens'] == 512
        assert call_args['temperature'] == 0.8
        assert call_args['top_p'] == 0.95
        assert call_args['top_k'] == 40

        # Check runtime parameters
        assert call_args['stop_sequences'] == ["STOP", "END"]
        assert call_args['metadata'] == {"request_id": "test-123"}

        # Check messages structure
        assert 'messages' in call_args
        assert len(call_args['messages']) == 1
        assert call_args['messages'][0]['role'] == 'user'
        assert call_args['messages'][0]['content'] == 'Test prompt'


@pytest.mark.unit
def test_system_message_parameter():
    """Test that system message parameter is handled correctly."""

    mock_content_block = MagicMock()
    mock_content_block.text = "Test response"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = mock_client

        provider = AnthropicLanguageModel(api_key='test-key')

        # Test with system message parameter
        prompts = ["Test prompt"]
        list(provider.infer(prompts, system="You are a helpful assistant for testing."))

        # Verify system message was passed
        call_args = mock_client.messages.create.call_args.kwargs
        assert call_args['system'] == "You are a helpful assistant for testing."


@pytest.mark.unit
def test_service_tier_parameter():
    """Test that service_tier parameter is handled correctly."""

    mock_content_block = MagicMock()
    mock_content_block.text = "Test response"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = mock_client

        provider = AnthropicLanguageModel(api_key='test-key')

        # Test with service_tier parameter
        prompts = ["Test prompt"]
        list(provider.infer(prompts, service_tier="standard_only"))

        # Verify service_tier was passed
        call_args = mock_client.messages.create.call_args.kwargs
        assert call_args['service_tier'] == "standard_only"


@pytest.mark.unit
def test_config_keys_completeness():
    """Ensure _ANTHROPIC_CONFIG_KEYS contains expected parameters."""
    expected_keys = {
        'max_tokens',
        'temperature',
        'top_p',
        'top_k',
        'stop_sequences',
        'metadata',
        'system',
        'stream',  # Unsupported but listed
        'tools',  # Unsupported but listed
        'tool_choice',  # Unsupported but listed
        'service_tier',  # New parameter
        'thinking',  # Unsupported but listed
    }

    assert _ANTHROPIC_CONFIG_KEYS == expected_keys


@pytest.mark.unit
def test_runtime_parameter_rejection():
    """Test that unsupported parameters are rejected at runtime."""

    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicLanguageModel(api_key='test-key')

        # Test rejection of unsupported parameters during infer()
        unsupported_runtime_params = ['stream', 'tools', 'tool_choice', 'thinking']

        for param in unsupported_runtime_params:
            with pytest.raises(
                lx.exceptions.InferenceConfigError,
                match=f"Unsupported parameter provided: {param}",
            ):
                list(provider.infer(["test prompt"], **{param: True}))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
