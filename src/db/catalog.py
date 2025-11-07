"""Catalog wrapper to unify access to models and libs tables."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import list_models as _list_models, get_model_dict2
from .libs import list_libs as _list_libs, get_lib_dict2


def list_models(task: Optional[str] = None) -> list[Dict[str, Any]]:
    return _list_models(task=task)


def list_libs(task: Optional[str] = None) -> list[Dict[str, Any]]:
    return _list_libs(task=task)


def get_model_dict(model_id: str) -> Optional[Dict[str, Any]]:
    return get_model_dict2(model_id)


def get_lib_dict(lib_id: str) -> Optional[Dict[str, Any]]:
    return get_lib_dict2(lib_id)
