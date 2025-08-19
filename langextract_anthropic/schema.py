"""Schema implementation for Anthropic provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import langextract as lx  # type: ignore[import-untyped]


class AnthropicSchema(lx.schema.BaseSchema):
    """Schema implementation for Anthropic structured output."""

    def __init__(self, schema_dict: dict[str, Any]) -> None:
        """Initialize the schema with a dictionary."""
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Return the schema dictionary."""
        return self._schema_dict

    @classmethod
    def from_examples(
        cls, examples_data: Sequence[Any], attribute_suffix: str = "_attributes"
    ) -> AnthropicSchema:
        """Build schema from example extractions.

        Args:
            examples_data: Sequence of ExampleData objects.
            attribute_suffix: Suffix for attribute fields.

        Returns:
            A configured AnthropicSchema instance.
        """
        extraction_types: dict[str, set[str]] = {}
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())

        schema_dict: dict[str, Any] = {
            "type": "object",
            "properties": {
                "extractions": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["extractions"],
        }

        return cls(schema_dict)

    def to_provider_config(self) -> dict[str, Any]:
        """Convert to provider-specific configuration.

        Returns:
            Dictionary of provider-specific configuration.
        """
        # Map to provider kwargs. We expose enable_structured_output to
        # align with provider plugin guidelines.
        return {
            "response_schema": self._schema_dict,
            "enable_structured_output": True,
        }

    @property
    def supports_strict_mode(self) -> bool:
        """Whether this schema guarantees valid structured output.

        Returns:
            True if the provider enforces valid JSON output (no code fences).
        """
        # Anthropic Claude models support structured output through
        # proper prompting and JSON formatting, which returns
        # well-formed JSON without markdown fencing.
        return True
