"""Marlowe cycle-6 scratch layout for VL-JEPA.

``/scratch/m000081-pm06`` is deleted 2026-10-01. Defaults:

    /scratch/m000081/eprakash/                          SCRATCH_BASE
      all_data/                                         silver CXRs
      hf/Llama-3.2-1B                                   Llama weights
      logs/                                             slurm stdout
      temporal/final/
        CheXTemporal/
        final_gold_{mimic,chexpert,rexgradient}_images/
        tempcxr/
        cxr-temporal-model/                             PROJECT_DIR

Override with ``SCRATCH_BASE``, ``PROJECT_DIR``, ``JEPA_IMAGE_ROOTS_DIR``,
``CHEXTEMPORAL_DIR``, ``VLJEPA_HF_HOME``, ``VLJEPA_LLAMA_LOCAL``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

SCRATCH_BASE_DEFAULT = "/scratch/m000081/eprakash"
PROJECT_DIR_DEFAULT = (
    "/scratch/m000081/eprakash/temporal/final/cxr-temporal-model"
)
GOLD_DATASETS = ("mimic", "chexpert", "rexgradient")


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scratch_base() -> str:
    return os.environ.get("SCRATCH_BASE", SCRATCH_BASE_DEFAULT)


def project_dir() -> str:
    return os.environ.get("PROJECT_DIR", PROJECT_DIR_DEFAULT)


def chextemporal_dir() -> str:
    env = os.environ.get("CHEXTEMPORAL_DIR")
    if env:
        return env
    for cand in (
        os.path.join(repo_root(), "CheXTemporal"),
        os.path.join(scratch_base(), "temporal", "final", "CheXTemporal"),
        os.path.join(project_dir(), "CheXTemporal"),
    ):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(repo_root(), "CheXTemporal")


def image_roots_dir() -> str:
    env = os.environ.get("JEPA_IMAGE_ROOTS_DIR")
    if env:
        return env
    local = os.path.join(repo_root(), "all_data")
    if os.path.isdir(local):
        return os.path.abspath(local)
    return os.path.join(scratch_base(), "all_data")


def image_roots() -> Dict[str, str]:
    base = image_roots_dir()
    return {
        "mimic": os.path.join(base, "mimic"),
        "chexpert": os.path.join(base, "chexpert", "train"),
        "rexgradient": os.path.join(base, "rexgradient", "deid_png"),
    }


def hf_home() -> str:
    return os.environ.get("VLJEPA_HF_HOME") or os.path.join(scratch_base(), "hf")


def llama_dir() -> str:
    explicit = os.environ.get("VLJEPA_LLAMA_LOCAL")
    if explicit:
        return explicit
    return os.path.join(hf_home(), "Llama-3.2-1B")


def himl_src() -> str:
    return os.path.join(
        repo_root(),
        "tempcxr",
        "modules",
        "hi-ml",
        "hi-ml-multimodal",
        "src",
    )


def gold_image_dir(dataset: str, extra_bases: Optional[List[str]] = None) -> Optional[str]:
    name = f"final_gold_{dataset}_images"
    bases = [
        repo_root(),
        os.path.join(scratch_base(), "temporal", "final"),
        project_dir(),
        chextemporal_dir(),
        os.path.dirname(chextemporal_dir()),
    ]
    if extra_bases:
        bases.extend(extra_bases)
    seen = set()
    for base in bases:
        if not base or base in seen:
            continue
        seen.add(base)
        cand = os.path.join(base, name)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def gold_image_dirs() -> Dict[str, str]:
    found = {}
    for d in GOLD_DATASETS:
        path = gold_image_dir(d)
        if path:
            found[d] = path
    return found


def silver_findings_parquet() -> str:
    return os.path.join(chextemporal_dir(), "silver_findings.parquet")


def gold_pairs_parquet() -> str:
    return os.path.join(chextemporal_dir(), "gold_progression_pairs.parquet")


def _first_file(
    root: str,
    suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".dcm"),
    max_dirs: int = 4000,
) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n += 1
        for name in filenames:
            if name.lower().endswith(suffixes):
                return os.path.join(dirpath, name)
        if n >= max_dirs:
            return None
    return None


def inventory(require_sample: bool = True) -> List[Tuple[str, str, bool, str]]:
    """Rows of (name, path, ok, detail).

    ``require_sample=False`` only checks that directories exist (fast slurm abort).
    """
    rows: List[Tuple[str, str, bool, str]] = []

    def add(name: str, path: str, ok: bool, detail: str = "") -> None:
        rows.append((name, path, ok, detail))

    proj = repo_root()
    add("repo", proj, os.path.isdir(os.path.join(proj, "vljepa")), "")

    himl = himl_src()
    add(
        "hi-ml",
        himl,
        os.path.isdir(os.path.join(himl, "health_multimodal")),
        "health_multimodal",
    )

    cxr = chextemporal_dir()
    silver = silver_findings_parquet()
    gold_pq = gold_pairs_parquet()
    add("CheXTemporal", cxr, os.path.isdir(cxr), "")
    add("silver_findings.parquet", silver, os.path.isfile(silver), "")
    add("gold_progression_pairs.parquet", gold_pq, os.path.isfile(gold_pq), "")

    for key, path in image_roots().items():
        exists = os.path.isdir(path)
        sample = _first_file(path) if exists and require_sample else None
        ok = exists and (sample is not None if require_sample else True)
        add(
            f"all_data/{key}",
            path,
            ok,
            sample or ("ok" if exists else "missing"),
        )

    for d in GOLD_DATASETS:
        path = gold_image_dir(d) or os.path.join(proj, f"final_gold_{d}_images")
        exists = os.path.isdir(path)
        sample = _first_file(path) if exists and require_sample else None
        ok = exists and (sample is not None if require_sample else True)
        add(
            f"gold/{d}",
            path,
            ok,
            sample or ("ok" if exists else "missing or empty"),
        )

    llama = llama_dir()
    add("Llama-3.2-1B", llama, os.path.isfile(os.path.join(llama, "config.json")), "config.json")
    return rows
