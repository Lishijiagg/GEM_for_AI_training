#!/usr/bin/env python3
"""
merge_dfba.py
Combine the parts of a stage-2 dFBA run that was split with --n-chunks.

  python3 merge_dfba.py results_dfba/<model name> [results_dfba/<model 2> ...]
  python3 merge_dfba.py results_dfba/*/

For each folder: checks that every part finished (dfba_summary_chunkIIIofNNN.tsv
present for all N parts), then writes dfba_summary.tsv, dfba_net_change.tsv,
dfba_initial_conc.tsv, dfba.log and run_info.txt as for an unsplit run.
The per-part files are moved to a subfolder chunks/.
Folders of unsplit runs (no chunk files) are left unchanged.
"""
import glob
import os
import re
import shutil
import sys

import pandas as pd

PAT = re.compile(r"_chunk(\d{3})of(\d{3})")


def merge(d):
    parts = sorted(glob.glob(os.path.join(d, "run_info_chunk*of*.txt")))
    if not parts:
        print(f"{d}: no parts found, nothing to merge")
        return True
    n = int(PAT.search(parts[0]).group(2))
    done = {int(PAT.search(p).group(1))
            for p in glob.glob(os.path.join(d, "dfba_summary_chunk*of*.tsv"))}
    missing = sorted(set(range(1, n + 1)) - done)
    if missing:
        print(f"{d}: {len(done)} of {n} parts finished; missing parts {missing}; not merged")
        return False

    def cat(prefix, sort_cols, fill=False, fmt=None):
        files = sorted(glob.glob(os.path.join(d, f"{prefix}_chunk*of*.tsv")))
        df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
        if fill:
            df = df.fillna(0)
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df.to_csv(os.path.join(d, f"{prefix}.tsv"), sep="\t", index=False, float_format=fmt)
        return df

    s = cat("dfba_summary", ["condition", "conc_set", "method"])
    cat("dfba_net_change", ["condition", "conc_set", "method"], fill=True, fmt="%.6g")
    ini = cat("dfba_initial_conc", ["condition", "conc_set"], fill=True, fmt="%.6g")

    info = dict(line.rstrip("\n").split("\t", 1)
                for line in open(parts[0]) if "\t" in line)
    expected = int(info.get("total_conc_sets", len(ini)))
    if len(ini) != expected:
        print(f"{d}: WARNING {len(ini)} concentration sets merged, {expected} expected")
    with open(os.path.join(d, "run_info.txt"), "w") as f:
        for k, v in info.items():
            if k in ("chunk", "conc_sets_in_chunk"):
                continue
            f.write(f"{k}\t{v}\n")
        f.write(f"merged_parts\t{n}\nmerged_conc_sets\t{len(ini)}\n")
    with open(os.path.join(d, "dfba.log"), "w") as out:
        for p in sorted(glob.glob(os.path.join(d, "dfba_chunk*of*.log"))):
            out.write(f"# {os.path.basename(p)}\n")
            out.write(open(p).read())

    cd = os.path.join(d, "chunks")
    os.makedirs(cd, exist_ok=True)
    for p in glob.glob(os.path.join(d, "*_chunk*of*.*")):
        shutil.move(p, os.path.join(cd, os.path.basename(p)))
    print(f"{d}: merged {n} parts, {len(ini)} concentration sets, {len(s)} runs")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok = all([merge(d.rstrip("/")) for d in sys.argv[1:]])
    sys.exit(0 if ok else 1)
