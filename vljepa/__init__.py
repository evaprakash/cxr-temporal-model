"""Pair-conditioned VL-JEPA for CheXTemporal 5-way progression."""

from .model import VLJEPA
from .prompts import QUERY_TEMPLATE, TARGET_TEMPLATE, class_target_texts, query_text

__all__ = [
    "VLJEPA",
    "QUERY_TEMPLATE",
    "TARGET_TEMPLATE",
    "query_text",
    "class_target_texts",
]
