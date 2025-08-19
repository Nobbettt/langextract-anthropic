"""Pytest configuration and fixtures for langextract-anthropic tests.

Notes:
- `tests/test_anthropic_parameters.py` is a script-style integration validator that is
  executed directly (e.g., via `python tests/test_anthropic_parameters.py`) when you
  have real Anthropic credentials. It is not a pytest test module and should be
  excluded from pytest collection to avoid fixture resolution errors.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Exclude the script-style integration validator from pytest collection. This
# prevents errors like "fixture 'param_name' not found" during unit test runs.
# It remains executable directly via `python tests/test_anthropic_parameters.py`.
collect_ignore = ["test_anthropic_parameters.py"]


@pytest.fixture
def mock_anthropic_credentials():
    """Mock Anthropic credentials for testing."""
    with patch.dict(
        os.environ,
        {
            'ANTHROPIC_API_KEY': 'test-api-key',
        },
    ):
        yield


@pytest.fixture
def mock_anthropic_client():
    """Mock the Anthropic client to avoid real API calls."""
    with patch('anthropic.Anthropic') as mock_client_class:
        mock_client = mock_client_class.return_value

        # Mock successful response with content blocks
        mock_content_block = MagicMock()
        mock_content_block.text = '{"extractions": []}'

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client.messages.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_extraction_examples():
    """Sample extraction examples for testing."""
    import langextract as lx

    return [
        lx.data.ExampleData(
            text="John Smith works at Microsoft in Seattle.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="person",
                    extraction_text="John Smith",
                    attributes={"role": "employee"},
                ),
                lx.data.Extraction(
                    extraction_class="organization",
                    extraction_text="Microsoft",
                    attributes={"type": "company"},
                ),
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text="Seattle",
                    attributes={"type": "city"},
                ),
            ],
        )
    ]
