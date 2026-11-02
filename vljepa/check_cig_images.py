#!/usr/bin/env python3
"""Check and copy the Chest ImaGenome gold images under a MIMIC-CXR-JPG copy.

Same as check_mscxrt_images.py (standard library only, same flags), with the
Chest ImaGenome gold CSV and a ./cig copy folder as defaults.

    python3 vljepa/check_cig_images.py \\
        --root /scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0

    # copy every found image to <dest>/pXX/pXXXXXXXX/sXXXXXXXX/<id>.jpg
    python3 vljepa/check_cig_images.py \\
        --root /scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0 \\
        --copy --copy-dir <dest>
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from check_mscxrt_images import main  # noqa: E402

DEFAULTS = {
    "--csv": os.environ.get("CIG_CSV", os.path.join(HERE, "..", "cig_gold_labels_new.csv")),
    "--copy-dir": "cig",
}

if __name__ == "__main__":
    given = {a.split("=", 1)[0] for a in sys.argv[1:]}
    for flag, value in DEFAULTS.items():
        if flag not in given:
            sys.argv += [flag, value]
    main()
