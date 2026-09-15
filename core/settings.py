"""Validated non-secret runtime settings, independent of service credentials."""

import math
from pathlib import Path
import tomllib


DEFAULTS = {
    "music": {
        "operation_timeout_seconds": 40.0,
        "prepare_concurrency": 3,
        "pending_limit": 32,
        "identity_review_timeout_seconds": 8.0,
        "clarification_timeout_seconds": 120.0,
    },
    "logging": {"level": "INFO", "sensitive_content": True},
}


def load_settings(path: Path) -> dict:
    """Use defaults for omitted settings; reject typos and invalid values."""
    try:
        with path.open("rb") as source:
            supplied = tomllib.load(source)
    except FileNotFoundError:
        supplied = {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc
    result = {section: values.copy() for section, values in DEFAULTS.items()}
    for section, values in supplied.items():
        if section not in DEFAULTS or not isinstance(values, dict):
            raise ValueError(f"{path}: unknown or invalid section {section!r}")
        for key, value in values.items():
            label = f"{path}: {section}.{key}"
            if key not in DEFAULTS[section]:
                raise ValueError(f"{label}: unknown setting")
            default = DEFAULTS[section][key]
            if section == "music":
                valid_type = type(value) is int if type(default) is int else type(value) in (int, float)
                if not valid_type or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{label}: expected a positive finite {'integer' if type(default) is int else 'number'}")
            elif key == "level":
                if not isinstance(value, str) or value.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                    raise ValueError(f"{label}: expected DEBUG, INFO, WARNING, ERROR, or CRITICAL")
                value = value.upper()
            elif type(value) is not bool:
                raise ValueError(f"{label}: expected true or false")
            result[section][key] = value
    return result
