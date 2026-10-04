"""Regenerate Table 2 (TABLE.csv) from the per-seed results.

The tunnel and turbine rows are recomputed from the results CSV that each run wrote
(03_tunnel/arm*/..._results.csv, 04_turbine/arm*/..._results.csv) and checked against
RESULTS.csv; the oscillator and frame rows come from RESULTS.csv, whose per-seed values
are in 01_sdof/arm*/seed*/ and 02_frame_v3/arm*/seed*/.

    python3 make_table2.py          # run inside release_webapp/
"""
import csv, statistics as st

ROWS = [("SDOF (eta 0.60-0.96, N=1500)", "pooled"), ("3-story frame v3", "pooled"),
        ("Tunnel moving-train (N=500)", "unweighted"), ("Turbine V_hub<=20.2", "unweighted")]
SEEDS = [42, 0, 1, 2, 3]
FILES = {"Tunnel moving-train (N=500)": ("03_tunnel/arm{a}/tunnel_half_arm{a}_s{s}_results.csv", 11),
         "Turbine V_hub<=20.2": ("04_turbine/arm{a}/wt_arm{a}_v20_full{suf}_results.csv", 5)}

res = {}
for r in csv.DictReader(open("RESULTS.csv")):
    res[(r["example"], r["metric"], int(r["seed"]))] = {k: float(r[k]) for k in ("arm1", "arm2", "arm3")}

def from_csv(ex, s):
    pat, n = FILES[ex]
    out = {}
    for a in (2, 3):
        f = pat.format(a=a, s=s, suf="" if s == 42 else f"_s{s}")
        rows = {r["label"]: r for r in csv.DictReader(open(f))}
        mean = lambda r: st.mean(float(r[f"r2_s{i}"]) for i in range(1, n + 1))
        out[f"arm{a}"] = mean(rows["trained"])
        out["arm1"] = mean(rows["u_red_ref"])          # Arm 1 = the reduced model, no learning
    return out

with open("TABLE.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["example", "metric", "n_seeds"] + [f"{a}_{m}" for a in ("arm1", "arm2", "arm3") for m in ("mean", "sd")]
               + ["A3_minus_A2_mean", "A3_minus_A2_sd", "A3_minus_A1_mean", "A3_minus_A1_sd"])
    for ex, metric in ROWS:
        per = []
        for s in SEEDS:
            v = from_csv(ex, s) if ex in FILES else res[(ex, metric, s)]
            if (ex, metric, s) in res:                   # cross-check the CSVs against RESULTS.csv
                assert all(abs(v[k] - res[(ex, metric, s)][k]) < 5e-4 for k in v), (ex, s, v, res[(ex, metric, s)])
            per.append(v)
        col = lambda f: [f(v) for v in per]
        stats = []
        for f in (lambda v: v["arm1"], lambda v: v["arm2"], lambda v: v["arm3"],
                  lambda v: v["arm3"] - v["arm2"], lambda v: v["arm3"] - v["arm1"]):
            stats += [f"{st.mean(col(f)):.4f}", f"{st.stdev(col(f)):.4f}"]
        w.writerow([ex, metric, len(per)] + stats)
        print(f"{ex:32s} Arm1 {stats[0]}±{stats[1]}  Arm2 {stats[2]}±{stats[3]}  Arm3 {stats[4]}±{stats[5]}")
