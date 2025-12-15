"""Filter module for MLflow registered model filtering."""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Supported filter operators."""

    EQ = "eq"
    NE = "ne"
    LIKE = "like"
    IN = "in"


class FilterCondition(BaseModel):
    """A single filter condition."""

    field: str
    operator: FilterOperator
    value: str


class ModelFilter:
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

    def __init__(self, filter_string: str | None) -> None:
        """Initialize filter from filter string.

        Args:
            filter_string: Filter string in format field:op:value,field:op:value
        """
        self.conditions: list[FilterCondition] = []
        self._raw_filter = filter_string

        if filter_string:
            self._parse(filter_string)

    def _parse(self, filter_string: str) -> None:
        """Parse filter string into conditions.

        Args:
            filter_string: Filter string to parse
        """
        for part in filter_string.split(","):
            part = part.strip()
            if not part:
                continue

            parts = part.split(":")
            if len(parts) == 3:
                field, op, value = parts
                try:
                    operator = FilterOperator(op.lower())
                    self.conditions.append(
                        FilterCondition(field=field.lower(), operator=operator, value=value)
                    )
                except ValueError:
                    logger.warning(f"Unknown filter operator: {op}")
            else:
                logger.warning(f"Invalid filter format: {part}. Expected field:operator:value")

    def apply(self, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply all filter conditions to a list of models.

        Args:
            models: List of MLflow registered model dicts to filter

        Returns:
            Filtered list of models matching all conditions (AND logic)
        """
        if not self.conditions:
            return models

        result = models
        for condition in self.conditions:
            result = [m for m in result if self._matches(m, condition)]

        logger.debug(
            f"Filter applied: {len(models)} -> {len(result)} models",
            extra={"conditions": len(self.conditions)},
        )
        return result

    def _matches(self, model: dict[str, Any], condition: FilterCondition) -> bool:
        """Check if a model matches a single filter condition.

        Args:
            model: MLflow registered model dict to check
            condition: Filter condition to apply

        Returns:
            True if model matches the condition
        """
        value = self._get_field_value(model, condition.field)

        if value is None:
            return condition.operator == FilterOperator.NE

        match condition.operator:
            case FilterOperator.EQ:
                return self._compare_equal(value, condition.value)
            case FilterOperator.NE:
                return not self._compare_equal(value, condition.value)
            case FilterOperator.LIKE:
                return condition.value.lower() in str(value).lower()
            case FilterOperator.IN:
                if isinstance(value, list):
                    # For MLflow tags, check if any tag key matches
                    for item in value:
                        if isinstance(item, dict) and "key" in item:
                            if condition.value.lower() == item["key"].lower():
                                return True
                        elif isinstance(item, str):
                            if condition.value.lower() == item.lower():
                                return True
                    return False
                return False

        return False

    def _compare_equal(self, value: Any, filter_value: str) -> bool:
        """Compare value for equality, handling different types.

        Args:
            value: Model field value
            filter_value: Filter value (string)

        Returns:
            True if values are equal
        """
        if isinstance(value, bool):
            return str(value).lower() == filter_value.lower()
        if isinstance(value, int | float):
            try:
                return value == type(value)(filter_value)
            except (ValueError, TypeError):
                return False
        return str(value).lower() == filter_value.lower()

    def _get_field_value(self, model: dict[str, Any], field: str) -> Any:
        """Get field value from model dict.

        Args:
            model: MLflow registered model dict
            field: Field name to retrieve

        Returns:
            Field value or None if not found
        """
        return model.get(field)

    def has_conditions(self) -> bool:
        """Check if filter has any conditions to apply.

        Returns:
            True if there are conditions
        """
        return bool(self.conditions)

    def __repr__(self) -> str:
        """String representation of filter."""
        return f"ModelFilter(conditions={len(self.conditions)})"
