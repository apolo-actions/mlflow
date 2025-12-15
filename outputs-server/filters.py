"""Filter module for MLflow registered model filtering."""

from typing import Any

from apolo_app_types.dynamic_outputs import BaseModelFilter


class ModelFilter(BaseModelFilter):
    """Filter for MLflow registered models.

    Supports filtering by any field with various operators:
    - eq: Exact match (case-insensitive)
    - ne: Not equal (case-insensitive)
    - like: Contains substring (case-insensitive)
    - in: Value exists in list field (for tags, checks tag keys)

    Filter syntax: field:operator:value,field2:operator2:value2

    Examples:
        - name:eq:my-model
        - name:like:llama
        - description:like:text-generation
        - tags:in:production
    """

    def _get_field_value(self, model: dict[str, Any], field: str) -> Any:
        """Get field value from model dict.

        Args:
            model: MLflow registered model dict
            field: Field name to retrieve

        Returns:
            Field value or None if not found
        """
        return model.get(field)

    def _matches_in_operator(self, value: Any, filter_value: str) -> bool:
        """Handle IN operator for MLflow tags.

        MLflow tags are stored as list of {key: str, value: str} dicts.

        Args:
            value: Field value (expected to be a list)
            filter_value: Value to search for in the list

        Returns:
            True if filter_value is found in value
        """
        if isinstance(value, list):
            for item in value:
                # MLflow tags: list of {key, value} dicts
                if isinstance(item, dict) and "key" in item:
                    if filter_value.lower() == item["key"].lower():
                        return True
                # Simple string list
                elif isinstance(item, str):
                    if filter_value.lower() == item.lower():
                        return True
            return False
        return False
