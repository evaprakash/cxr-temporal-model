"""Shared progression-class vocabulary.

Pulled out of ``progression_classify.py`` so the dataset (``dataset_combined_jepa``)
can build templated training conditions of the form ``"{finding} is {phrase}"``
using the same class names that the eval-time prompt ensembling uses.
Kept dependency-free so importers don't trigger circular imports.

Vocabulary
----------
``CLS_ORDER``
    Canonical 5-class progression order. Used everywhere as the column
    order of confusion matrices, the row order of per-class scores, etc.

``PROGRESSION_PHRASES``
    CLIP-style synonym bank, one list per class. Used by
    ``progression_classify.build_class_prompts`` to construct the
    multi-phrase prompt bank that the eval scripts score against.

``SILVER_TO_CLS``
    Maps the raw progression strings in ``silver_findings.parquet``
    (``Stable`` / ``New`` / ``Worse`` / ``Improved`` / ``Resolved``) onto
    ``CLS_ORDER``. Used by the dataset when building templated condition
    text from silver findings rows.

``GT_TO_CLS``
    Maps the raw progression strings in ``gold_progression_pairs.parquet``
    onto ``CLS_ORDER``. Gold uses past-tense / adjective forms
    (``improved`` / ``worse``); we re-key those to present-tense forms
    that match the ``"{disease} is {phrase}"`` template.
"""

from typing import Dict, List


CLS_ORDER: List[str] = ["improving", "stable", "worsening", "new", "resolved"]


PROGRESSION_PHRASES: Dict[str, List[str]] = {
    "improving": [
        "better", "decreased", "decreasing", "improved",
        "improving", "reduced", "smaller",
    ],
    "stable": [
        "constant", "stable", "unchanged",
    ],
    "worsening": [
        "bigger", "developing", "enlarged", "enlarging", "greater",
        "growing", "increased", "increasing", "larger",
        "progressing", "progressive", "worse", "worsened", "worsening",
    ],
    "new": [
        "new", "newly developed", "newly appeared", "newly seen",
        "appeared", "emerged",
    ],
    "resolved": [
        "resolved", "resolving", "cleared", "disappeared",
        "no longer present", "no longer seen", "completely resolved",
    ],
}


SILVER_TO_CLS: Dict[str, str] = {
    "Stable":   "stable",
    "New":      "new",
    "Worse":    "worsening",
    "Improved": "improving",
    "Resolved": "resolved",
}


GT_TO_CLS: Dict[str, str] = {
    "improved": "improving",
    "worse":    "worsening",
    "stable":   "stable",
    "new":      "new",
    "resolved": "resolved",
}
