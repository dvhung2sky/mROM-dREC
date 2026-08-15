#!/usr/bin/env python3
"""Filter the generated turbine database to the operating envelope used in the paper.

The published example keeps the samples with a hub-height mean wind speed at or below
20.2 m/s, 1248 of the 1597 generated. Above rated wind the pitch controller sheds thrust
and aerodynamic damping rises, so the fore-aft fluctuation shrinks against the mean and
those samples dominate the per-sample failures; the envelope is stated rather than hidden.

    python3 cap_vhub.py wind_turbine_database.npz wt_v20_full.npz
"""
from __future__ import annotations

import sys

import numpy as np

V_MAX = 20.2


def main(src: str, dst: str) -> None:
    d = np.load(src, allow_pickle=True)
    names = [str(x) for x in d["param_names"]]
    keep = d["params"][:, names.index("V_hub")] <= V_MAX
    n = int(keep.sum())
    out = {}
    for k in d.files:
        a = d[k]
        out[k] = a[keep] if a.ndim >= 1 and a.shape[0] == len(keep) else a
    np.savez_compressed(dst, **out)
    print(f"{n} of {len(keep)} samples with V_hub <= {V_MAX} m/s -> {dst}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["wind_turbine_database.npz", "wt_v20_full.npz"]))
