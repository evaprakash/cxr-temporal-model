"""3-way progression classification on Chest ImaGenome (CIG).

The CIG temporal-comparison labels are 6-way in the raw release
(improved / worsened / no change / new / resolved / unchanged-finding,
plus some "n/a" rows). For an apples-to-apples comparison with our
present-tense prompt bank we keep only the three labels that map cleanly
to {improving, stable, worsening} and drop everything else:

    improved   -> improving
    worsened   -> worsening
    no change  -> stable

Anything else (``new``, ``resolved``, ``n/a``, etc.) is filtered out
before evaluation. Like MS-CXR-T, we restrict the model's argmax /
argmin to these three classes.

The expected CSV schema is

    patient_id, study_id_prev, study_id_curr,
    img_path_prev, img_path_curr, disease_name, comparison

with absolute MIMIC-CXR JPG paths in ``img_path_prev`` /
``img_path_curr``.

Modes
-----
``--demo``   Print 3-way scoring for one CSV row.
``--eval``   Compute overall + per-class + per-finding accuracy.

Usage
-----
    python eval_cig.py --demo
    python eval_cig.py --eval
    python eval_cig.py --eval --csv /path/to/cig_gold_labels_new.csv
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


BENCHMARK_NAME = "CIG"
DEFAULT_CSV = os.environ.get("CIG_CSV", "cig_gold_labels_new.csv")

# Per the user's instruction: keep only ``improved``, ``worsened``,
# ``no change``. Anything else (including ``new`` / ``resolved`` /
# ``n/a``) is dropped at filter time. The CIG export sometimes lower-
# cases differently; ``load_csv_pairs`` already does ``.strip().lower()``
# so we just need a couple of underscored / hyphenated variants here.
LABEL_MAP = {
    "improved":   "improving",
    "worsened":   "worsening",
    "no change":  "stable",
    "no-change":  "stable",
    "no_change":  "stable",
}

# CIG, after the above filter, has no ``new`` or ``resolved`` cases —
# restrict to 3 classes.
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
            f"No usable {BENCHMARK_NAME} rows after label filtering. "
            f"Expected at least one row with comparison in "
            f"{set(LABEL_MAP.keys())}."
        )

    if args.demo:
        run_csv_demo(args, model, df, VALID_CLASSES, device)
    else:
        run_csv_eval(args, model, df, VALID_CLASSES, device,
                     benchmark_name=BENCHMARK_NAME)


if __name__ == "__main__":
    main()
