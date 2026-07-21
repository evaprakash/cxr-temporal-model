"""Canonical CXR anatomy region vocabulary + finding→region map.

Used by the anatomy-embedding add-on: each name is encoded once with
BioViL-T's text encoder to warm-start a learnable 128-D token, then the
token attends over patch grids at train time.

Region names are short radiology phrases (not finding names). The map
below assigns each CheXpert-style silver finding to one or more regions
so multi-finding pairs contribute multiple contrastive rows.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


# ~24 region phrases — enough spatial diversity without exploding the
# token table. Warm-started via BioViL-T text encode of these strings.
ANATOMY_REGIONS: List[str] = [
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "right lung apex",
    "left lung apex",
    "right lung base",
    "left lung base",
    "right hilum",
    "left hilum",
    "cardiac silhouette",
    "cardiomediastinal silhouette",
    "mediastinum",
    "aortic arch",
    "trachea",
    "right costophrenic angle",
    "left costophrenic angle",
    "right pleural space",
    "left pleural space",
    "right hemidiaphragm",
    "left hemidiaphragm",
    "lingula",
]

REGION_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(ANATOMY_REGIONS)}

# Each silver finding → one or more anatomy regions. Bilateral / diffuse
# findings list both sides so same-image left/right negatives exist.
FINDING_TO_REGIONS: Dict[str, List[str]] = {
    "atelectasis": [
        "right lower lung zone",
        "left lower lung zone",
        "right lung base",
        "left lung base",
    ],
    "cardiomegaly": ["cardiac silhouette"],
    "consolidation": [
        "right mid lung zone",
        "left mid lung zone",
        "right lower lung zone",
        "left lower lung zone",
    ],
    "edema": [
        "right mid lung zone",
        "left mid lung zone",
        "right lower lung zone",
        "left lower lung zone",
        "cardiac silhouette",
    ],
    "enlarged cardiomediastinum": [
        "cardiomediastinal silhouette",
        "mediastinum",
    ],
    "lung lesion": [
        "right mid lung zone",
        "left mid lung zone",
        "right upper lung zone",
        "left upper lung zone",
    ],
    "lung opacity": [
        "right mid lung zone",
        "left mid lung zone",
        "right lower lung zone",
        "left lower lung zone",
    ],
    "pleural effusion": [
        "right pleural space",
        "left pleural space",
        "right costophrenic angle",
        "left costophrenic angle",
        "right lung base",
        "left lung base",
    ],
    "pleural other": [
        "right pleural space",
        "left pleural space",
    ],
    "pneumonia": [
        "right mid lung zone",
        "left mid lung zone",
        "right lower lung zone",
        "left lower lung zone",
    ],
    "pneumothorax": [
        "right pleural space",
        "left pleural space",
        "right lung apex",
        "left lung apex",
    ],
}


def findings_to_region_ids(findings: Sequence[str]) -> List[int]:
    """Unique sorted region indices for a list of finding names."""
    ids = set()
    for f in findings:
        key = str(f).strip().lower()
        for region in FINDING_TO_REGIONS.get(key, []):
            if region in REGION_TO_IDX:
                ids.add(REGION_TO_IDX[region])
    return sorted(ids)
