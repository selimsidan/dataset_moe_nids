"""Loads the single YAML config that drives a full run, applies `--set
a.b.c=value` overrides, and resolves `${run_name}` placeholders.

Ported unchanged from moe_nids/training/config.py.
"""
from __future__ import annotations

import copy
import re

import yaml

from data import paths as _paths

_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides or []:
        key, value = override.split("=", 1)
        _set_dotted(cfg, key, _coerce(value))
    cfg.setdefault("OUTPUT_DIR", _paths.OUTPUT_DIR)
    _resolve_placeholders(cfg, cfg)
    return cfg


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    return value


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _get_dotted(cfg: dict, dotted_key: str):
    node = cfg
    for part in dotted_key.split("."):
        node = node[part]
    return node


def _resolve_placeholders(node, root: dict):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                node[k] = _resolve_string(v, root)
            else:
                _resolve_placeholders(v, root)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                node[i] = _resolve_string(v, root)
            else:
                _resolve_placeholders(v, root)


def _resolve_string(value: str, root: dict) -> str:
    def _sub(match: re.Match) -> str:
        return str(_get_dotted(root, match.group(1)))

    return _PLACEHOLDER.sub(_sub, value)


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
