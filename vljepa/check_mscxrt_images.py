#!/usr/bin/env python3
"""Check which MS-CXR-T images exist under a MIMIC-CXR-JPG copy.

Standard library only, so it runs on any machine with python3 (no conda env).
Image paths are matched from ``files/pXX/...`` onward, so the CSV's original
prefix does not matter.

    python3 vljepa/check_mscxrt_images.py \\
        --root /scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0

    # only the files prepare reported missing on ccw, and list the ones found
    # here so they can be tarred and copied over
    python3 vljepa/check_mscxrt_images.py \\
        --root /scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0 \\
        --only mscxrt_clean_missing_urls.txt --found-list found.txt
    tar -czf mscxrt_missing.tgz -C <root> -T found.txt

    # copy every found image to ./mscxrt/pXX/pXXXXXXXX/sXXXXXXXX/<id>.jpg
    python3 vljepa/check_mscxrt_images.py --root <root> --copy
"""

import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter


def progress(items, label):
    """Yield items while drawing a one-line bar on stderr."""
    n = len(items)
    start = last = time.time()
    for i, item in enumerate(items, 1):
        yield item
        now = time.time()
        if i == n or now - last > 0.2:
            last = now
            done = int(30 * i / max(n, 1))
            rate = i / max(now - start, 1e-9)
            eta = (n - i) / rate if rate else 0
            sys.stderr.write(f"\r{label} [{'#' * done}{'.' * (30 - done)}] {i}/{n} "
                             f"{rate:.0f}/s eta {eta:.0f}s ")
            sys.stderr.flush()
    sys.stderr.write("\n")


def rel_files_path(path):
    path = path.strip()
    i = path.find("files/p")
    return path[i:] if i >= 0 else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default=os.path.join(here, "..", "mscxrt_labels_new.csv"))
    p.add_argument("--root", default="/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0",
                   help="folder that contains files/p10 ... files/p19")
    p.add_argument("--only", help="check just these paths or URLs (one per line), "
                                  "e.g. mscxrt_clean_missing_urls.txt from prepare")
    p.add_argument("--found-list", help="write the found paths here, relative to --root (for tar -T)")
    p.add_argument("--missing-list", help="write the still-missing paths here")
    p.add_argument("--copy", action="store_true",
                   help="copy the found images into --copy-dir, keeping the layout after files/")
    p.add_argument("--copy-dir", default="mscxrt",
                   help="where --copy puts images (default: ./mscxrt in the current directory)")
    args = p.parse_args()

    if args.only:
        with open(args.only) as f:
            wanted = {rel_files_path(line) for line in f if line.strip()}
        source = args.only
    else:
        wanted = set()
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                wanted.add(rel_files_path(row["img_path_prev"]))
                wanted.add(rel_files_path(row["img_path_curr"]))
        source = args.csv
    if None in wanted:
        wanted.discard(None)
        print("[check] some paths had no files/ part and were skipped")
    if not os.path.isdir(os.path.join(args.root, "files")):
        raise SystemExit(f"[check] no files/ folder under {args.root}")

    found, missing = [], []
    for rel in progress(sorted(wanted), "checking"):
        (found if os.path.isfile(os.path.join(args.root, rel)) else missing).append(rel)

    print(f"[check] {source}: {len(wanted)} distinct images")
    print(f"[check] under {args.root}: found {len(found)}, missing {len(missing)}")
    if missing:
        by_group = Counter(m.split("/")[1] for m in missing)
        print("  missing by folder: " + ", ".join(f"{k} {v}" for k, v in sorted(by_group.items())))
        for m in missing[:5]:
            print(f"  e.g. {m}")
    if args.found_list:
        with open(args.found_list, "w") as f:
            f.write("".join(r + "\n" for r in found))
        print(f"[check] wrote {len(found)} found paths to {args.found_list}")
    if args.missing_list:
        with open(args.missing_list, "w") as f:
            f.write("".join(r + "\n" for r in missing))
        print(f"[check] wrote {len(missing)} missing paths to {args.missing_list}")
    if args.copy:
        dest_root = os.path.abspath(args.copy_dir)
        copied = skipped = 0
        for rel in progress(found, "copying "):
            src = os.path.join(args.root, rel)
            dst = os.path.join(dest_root, rel[len("files/"):])
            if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        print(f"[check] copied {copied} images to {dest_root} "
              f"({skipped} already there){'; missing ones were not copied' if missing else ''}")


if __name__ == "__main__":
    main()
