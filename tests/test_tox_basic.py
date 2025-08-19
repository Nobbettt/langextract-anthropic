"""Basic test to verify tox setup works correctly."""

import pytest


@pytest.mark.unit
def test_hello_world():
    """Test basic functionality."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_import_provider():
    """Test that we can import the Anthropic provider."""
    from langextract_anthropic.provider import AnthropicLanguageModel

    # Test that the class exists and has expected attributes
    assert hasattr(AnthropicLanguageModel, 'infer')
    assert hasattr(AnthropicLanguageModel, 'get_schema_class')


@pytest.mark.unit
def test_provider_registration():
    """Provider registration and routing via registry.resolve()."""
    import langextract as lx
    from langextract.providers import registry

    lx.providers.load_plugins_once()
    provider_class = registry.resolve('anthropic-test')
    assert provider_class.__name__ == 'AnthropicLanguageModel'
