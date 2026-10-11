"""Option-2 VL-JEPA text templates.

Query (predictor condition)
    ``What is the progression of {finding}?``
    Finding stays lowercase (``pleural effusion``).

Targets (Y-encoder)
    ``{Finding} is {class}.``
    Finding is sentence-cased (``Pleural effusion is stable.``).
    One sentence per ``CLS_ORDER`` class. Guess c is scored only against sentence c.
"""

from __future__ import annotations

from typing import List, Sequence

from . import _root  # noqa: F401

from progression_phrases import CLS_ORDER

QUERY_TEMPLATE = "What is the progression of {finding}?"
TARGET_TEMPLATE = "{finding} is {cls}."


def cap_finding(finding: str) -> str:
    finding = str(finding).strip()
    if not finding:
        return finding
    return finding[:1].upper() + finding[1:]


def query_text(finding: str) -> str:
    return QUERY_TEMPLATE.format(finding=str(finding).strip().lower())


def target_text(finding: str, cls: str) -> str:
    return TARGET_TEMPLATE.format(finding=cap_finding(finding), cls=cls)


def class_target_texts(
    finding: str,
    classes: Sequence[str] | None = None,
) -> List[str]:
    cls_list = CLS_ORDER if classes is None else list(classes)
    return [target_text(finding, cls) for cls in cls_list]
