"""3-way progression classification on MS-CXR-T.

MS-CXR-T (Bannur et al.) provides held-out (prior, current) pairs from
MIMIC-CXR with a 3-way per-finding label: ``improving`` / ``stable`` /
``worsening``. There are no ``new`` or ``resolved`` cases in this
benchmark, so we restrict the model's argmax / argmin to those three
canonical classes.

The expected CSV schema is

    patient_id, study_id_prev, study_id_curr,
    img_path_prev, img_path_curr, disease_name, comparison

with absolute MIMIC-CXR JPG paths in ``img_path_prev`` /
``img_path_curr`` (so we don't apply any prefix-stripping logic).
We reuse the multi-phrase prompt bank, the ``score_one_pair`` core
scoring routine, and the dual cosine / Smooth L1 reporting from
``progression_classify.py``.

Modes
-----
``--demo``   Print 3-way scoring for one CSV row.
``--eval``   Compute overall + per-class + per-finding accuracy.

Usage
-----
    python eval_mscxrt.py --demo
    python eval_mscxrt.py --eval
    python eval_mscxrt.py --eval --csv /path/to/mscxrt_labels_new.csv
"""

import os
import torch

from csv_progression_eval import (
    build_parser,
    load_csv_pairs,
    run_csv_demo,
    run_csv_eval,
)
from infer_jepa import load_jepa_model


BENCHMARK_NAME = "MS-CXR-T"
DEFAULT_CSV = os.environ.get("MSCXRT_CSV", "mscxrt_labels_new.csv")

# Map raw ``comparison`` strings to canonical CLS_ORDER classes.
# MS-CXR-T uses the present-tense forms by default, but we also accept
# a few common synonyms (no-change, improved, worsened) so a CSV
# regenerated with a slightly different vocab still works.
LABEL_MAP = {
    "improving": "improving",
    "improved":  "improving",
    "stable":    "stable",
    "no change": "stable",
    "no-change": "stable",
    "unchanged": "stable",
    "worsening": "worsening",
    "worsened":  "worsening",
}

# MS-CXR-T has no ``new`` or ``resolved`` cases — restrict to 3 classes.
VALID_CLASSES = ["improving", "stable", "worsening"]


def main():
    parser = build_parser(DEFAULT_CSV, BENCHMARK_NAME)
    args = parser.parse_args()

    try:
        _ = args.prompt_template.format("test_disease", "test_phrase")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{disease}}, {{phrase}}). Got {args.prompt_template!r}: {e}"
        )

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)
    df = load_csv_pairs(args.csv, LABEL_MAP, VALID_CLASSES)
    if len(df) == 0:
        raise RuntimeError(
            f"No usable {BENCHMARK_NAME} rows after label filtering."
        )

    if args.demo:
        run_csv_demo(args, model, df, VALID_CLASSES, device)
    else:
        run_csv_eval(args, model, df, VALID_CLASSES, device,
                     benchmark_name=BENCHMARK_NAME)


if __name__ == "__main__":
    main()
