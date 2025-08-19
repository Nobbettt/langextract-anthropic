"""Unit tests for Anthropic provider."""

import langextract as lx
import pytest

from langextract_anthropic import AnthropicLanguageModel
from langextract_anthropic.schema import AnthropicSchema


@pytest.mark.unit
class TestAnthropicProvider:
    """Test Anthropic provider functionality."""

    def test_provider_initialization(self, mock_anthropic_client):
        """Test provider can be initialized with valid credentials."""
        provider = AnthropicLanguageModel(
            model_id="anthropic-claude-3-5-sonnet-latest",
            api_key="test-key",
        )

        assert provider.model_id == "anthropic-claude-3-5-sonnet-latest"
        assert provider.model_name == "claude-3-5-sonnet-latest"
        assert provider.api_key == "test-key"

    def test_model_name_extraction(self, mock_anthropic_client):
        """Test model name is correctly extracted from model ID."""
        test_cases = [
            ("anthropic-claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"),
            ("anthropic-claude-3-opus-latest", "claude-3-opus-latest"),
            ("claude-3-haiku-latest", "claude-3-haiku-latest"),  # Direct model name
            (None, "claude-3-5-sonnet-latest"),  # Default
        ]

        for model_id, expected_name in test_cases:
            provider = AnthropicLanguageModel(
                model_id=model_id,
                api_key="test-key",
            )
            assert provider.model_name == expected_name

    def test_provider_initialization_missing_api_key(self):
        """Test provider fails without API key."""
        with pytest.raises(lx.exceptions.InferenceConfigError, match="API key not provided"):
            AnthropicLanguageModel(
                model_id="anthropic-claude-3-5-sonnet-latest",
            )

    def test_unsupported_parameters_rejected(self, mock_anthropic_client):
        """Test that unsupported parameters are rejected at initialization."""
        unsupported_params = ["stream", "tools", "tool_choice", "thinking"]

        for param in unsupported_params:
            with pytest.raises(
                lx.exceptions.InferenceConfigError,
                match=f"Unsupported parameter provided: {param}"
            ):
                AnthropicLanguageModel(
                    api_key="test-key",
                    **{param: True}
                )

    def test_schema_class_support(self, mock_anthropic_client):
        """Test provider returns correct schema class."""
        provider = AnthropicLanguageModel(api_key="test-key")
        assert provider.get_schema_class() == AnthropicSchema

    def test_apply_schema(self, mock_anthropic_client):
        """Test schema application."""
        provider = AnthropicLanguageModel(api_key="test-key")

        # Test with None schema
        provider.apply_schema(None)
        assert provider._response_schema is None
        assert provider._enable_structured_output is False

        # Test with Anthropic schema
        schema = AnthropicSchema({"type": "object"})
        provider.apply_schema(schema)
        assert provider._response_schema is not None
        assert provider._enable_structured_output is True

    def test_parameter_filtering(self, mock_anthropic_client):
        """Test that only whitelisted parameters are stored."""
        valid_params = {
            "max_tokens": 1000,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "stop_sequences": ["STOP"],
            "metadata": {"user": "test"}
        }
        invalid_params = {
            "invalid_param": "value",
            "another_invalid": 123
        }

        provider = AnthropicLanguageModel(
            api_key="test-key",
            **valid_params,
            **invalid_params
        )

        # Only valid parameters should be stored (temperature is stored separately)
        expected_extra_kwargs_count = len(valid_params) - 1  # temperature is not in _extra_kwargs
        assert len(provider._extra_kwargs) == expected_extra_kwargs_count

        # Temperature is stored as instance variable
        assert provider.temperature == valid_params["temperature"]

        # Other valid parameters are stored in _extra_kwargs
        for key, value in valid_params.items():
            if key == "temperature":
                continue  # Skip temperature, already checked above
            assert provider._extra_kwargs[key] == value

        # Invalid parameters should not be stored
        for key in invalid_params:
            assert key not in provider._extra_kwargs

    def test_max_workers_setting(self, mock_anthropic_client):
        """Test max_workers parameter is set correctly."""
        provider = AnthropicLanguageModel(
            api_key="test-key",
            max_workers=5
        )
        assert provider.max_workers == 5

        # Test default value
        provider_default = AnthropicLanguageModel(api_key="test-key")
        assert provider_default.max_workers == 10
