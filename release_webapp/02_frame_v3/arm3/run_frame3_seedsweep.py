"""Seed sweep for the frame v3 three-arm result.

Reproduces run_frame3_split.py exactly (same notebook cells, same CPU-affordable
config override: L=150, STEP=150, HIDDEN=128, LAYERS=1, BATCH=64, EPOCHS=10) but with
SEED as a parameter, so the reported 0.8671 can be checked against other draws.

SEED drives BOTH the 70/15/15 split and the torch init, so the spread here is the
combined split + training variance -- which is what a reader of the table actually
cares about.  Arm 1 needs no model, so each run also reports how favourable its own
test split was, letting the two sources be separated after the fact.

Writes seed_sweep/frame3_seed<SEED>_arm<ARM>.json -- never touches the existing
frame3_split_arm{2,3}.json.

Usage: python3 run_frame3_seedsweep.py <seed> [arm=3] [epochs=10]
"""
import json
import os
import re
import sys

import numpy as np

SEED   = int(sys.argv[1])
ARM    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10

# keep parallel runs from oversubscribing the 8 P-cores
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))

OUTDIR = "seed_sweep"
os.makedirs(OUTDIR, exist_ok=True)

nb = json.load(open("frame3_v3_colab.ipynb"))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
G = {}

# --- cell 0: setup, on CPU, with the seed swapped in ---
src0 = (code[0].replace('assert torch.cuda.is_available()', 'assert True')
               .replace('device = torch.device("cuda")', 'device = torch.device("cpu")')
               .replace('print("GPU:", torch.cuda.get_device_name(0))',
                        'print("running on CPU")'))
src0 = re.sub(r"^SEED = .*$", f"SEED = {SEED}", src0, count=1, flags=re.M)
exec(src0, G)
assert G['SEED'] == SEED, G['SEED']
print(f"=== SEED={SEED}  ARM={ARM}  EPOCHS={EPOCHS} ===")

G['DB'] = 'frame3_v3_database.npz'
exec(code[2], G)          # consolidate + validity screen
exec(code[3], G)          # 70/15/15 split
exec(code[4], G)          # metrics + Arm 1
exec(code[5], G)          # model definitions (defines the config constants)

# --- same CPU-affordable override as run_frame3_split.py, then re-exec cell 5 ---
src5 = code[5]
# L_IN/L_OUT share one line and L_ALL is derived from them -- rewrite that line whole.
src5 = src5.replace("L_IN = L_OUT = 200", "L_IN = L_OUT = 150")
for k, v in dict(STEP=150, HIDDEN=128, LAYERS=1, BATCH=64, EPOCHS=EPOCHS).items():
    src5 = re.sub(rf"^{k}(\s*)= .*$", f"{k}\\1= {v}", src5, count=1, flags=re.M)
exec(src5, G)
assert G['L_ALL'] == G['L_IN'] + G['L_OUT'], (G['L_IN'], G['L_OUT'], G['L_ALL'])
print(f"config: L_IN={G['L_IN']} L_OUT={G['L_OUT']} STEP={G['STEP']} "
      f"HIDDEN={G['HIDDEN']} LAYERS={G['LAYERS']} EPOCHS={G['EPOCHS']}")

exec(code[6], G)          # channels
exec(code[7], G)          # train + rollout

u_nl, u_red = G['u_nl'], G['u_red']
idx_te, idx_va = G['idx_te'], G['idx_va']

mods = G['run_arm'](ARM, f"S{SEED}-ARM{ARM}")
pred = G['rollout'](mods, idx_te, ARM)
m = G['metrics'](u_nl[idx_te], pred, f"Arm {ARM}")
G['show'](m)

out = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
       for k, v in m.items()}
out['seed'] = SEED
out['arm'] = ARM
out['epochs'] = EPOCHS
out['n_train'], out['n_val'], out['n_test'] = len(G['idx_tr']), len(idx_va), len(idx_te)

# Arm 1 on this seed's own test split -- free (no model), and tells us how
# favourable this particular draw was.
arm1 = G['metrics'](u_nl[idx_te], u_red[idx_te], "Arm 1")
out['arm1_R2'] = float(arm1['R2'])
out['arm1_peak_err_pct'] = float(arm1['peak_err_pct'])
out['arm1_RMSE_mm'] = float(arm1['RMSE_mm'])

if ARM == 3:
    # split-invariant quality: fraction of the Arm-1 residual energy removed
    ss1 = float(((u_nl[idx_te] - u_red[idx_te]) ** 2).sum())
    ss3 = float(((u_nl[idx_te] - pred) ** 2).sum())
    out['resid_energy_removed'] = 1.0 - ss3 / ss1

    # alpha fitted on VAL, applied to TEST (never fitted on the reported split)
    pv = G['rollout'](mods, idx_va, 3)
    dv = pv - u_red[idx_va]
    dte = pred - u_red[idx_te]
    grid = np.linspace(0, 1.2, 25)
    tv = u_nl[idx_va]
    denom = ((tv - tv.mean()) ** 2).sum() + 1e-12
    r2v = [1 - ((tv - (u_red[idx_va] + a * dv)) ** 2).sum() / denom for a in grid]
    a_star = float(grid[int(np.argmax(r2v))])
    mA = G['metrics'](u_nl[idx_te], u_red[idx_te] + a_star * dte,
                      f"Arm 3 alpha={a_star:.2f}")
    G['show'](mA)
    out['alpha_star'] = a_star
    out['R2_alpha'] = float(mA['R2'])
    out['peak_alpha'] = float(mA['peak_err_pct'])

path = os.path.join(OUTDIR, f"frame3_seed{SEED}_arm{ARM}.json")
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}")
print(f"SUMMARY seed={SEED} arm1={out['arm1_R2']:+.4f} arm{ARM}={out['R2']:+.4f} "
      + (f"removed={out['resid_energy_removed']:.4f} alpha={out.get('alpha_star')} "
         f"R2a={out.get('R2_alpha'):+.4f}" if ARM == 3 else ""))
