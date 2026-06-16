from typing import Any, Dict


def build_validation_error_message(errors: Dict[str, Any]) -> str:
    """Build a readable message for validation errors."""
    lines = [f"{key}: {value}" for key, value in errors.items()]
    return "; ".join(lines)
