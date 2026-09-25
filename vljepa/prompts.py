"""Option-2 VL-JEPA text templates.

Query (predictor condition)
    ``What is the progression of {Finding}?``

Target / negatives (Y-encoder / InfoNCE)
    ``{Finding} is {class}.``
    Positive = gold class; negatives = the other four ``CLS_ORDER`` classes.

Finding capitalization matches the existing JEPA templated condition
(``Edema is worsening.``).
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
    return QUERY_TEMPLATE.format(finding=cap_finding(finding))


def target_text(finding: str, cls: str) -> str:
    return TARGET_TEMPLATE.format(finding=cap_finding(finding), cls=cls)


def class_target_texts(
    finding: str,
    classes: Sequence[str] | None = None,
) -> List[str]:
    cls_list = CLS_ORDER if classes is None else list(classes)
    return [target_text(finding, cls) for cls in cls_list]
