"""agg_matrix.py — aggregate the 75-run training matrix.

Reads every physx/models/matrix/matrix_<domain>_<kind>_s<seed>.stats.json and
prints per (domain, kind) mean train rel-MAE, plus pooled stats.
"""

import collections
import glob
import json
import os

FILES = sorted(glob.glob(os.path.join("physx", "models", "matrix", "matrix_*.json")))

rows = []
for f in FILES:
    d = json.load(open(f))
    base = os.path.basename(f).replace(".stats.json", "")
    parts = base.split("_")
    dom, kind, seed = parts[1], parts[2], int(parts[3][1:])
    rows.append({
        "domain": dom, "kind": kind, "seed": seed,
        "train_rel_mae": d.get("train_rel_mae"),
        "traj_norm": d.get("traj_norm"),
        "hard_phys": d.get("hard_phys"),
    })

agg = collections.defaultdict(list)
for r in rows:
    agg[(r["domain"], r["kind"])].append(r["train_rel_mae"])

print(f"{'domain':12s} {'kind':10s} n  mean(rel_mae)  min    max")
for k in sorted(agg):
    v = agg[k]
    print(f"{k[0]:12s} {k[1]:10s} {len(v)}  {sum(v)/len(v):.4f}   {min(v):.4f} {max(v):.4f}")

# pooled physics vs no-physics (transformer only), per domain
print()
for dom in sorted(set(r["domain"] for r in rows)):
    phys = [r["train_rel_mae"] for r in rows if r["domain"] == dom and r["kind"] == "phys"]
    nophys = [r["train_rel_mae"] for r in rows if r["domain"] == dom and r["kind"] == "nophys"]
    if phys and nophys:
        print(f"{dom:12s} phys {sum(phys)/len(phys):.4f} (n={len(phys)}) vs nophys {sum(nophys)/len(nophys):.4f} (n={len(nophys)})")

print(f"\ntotal runs: {len(rows)}")
kinds = collections.Counter(r['kind'] for r in rows)
print("kinds:", dict(kinds))
norms = collections.Counter(r['traj_norm'] for r in rows)
print("traj_norms:", dict(norms))
