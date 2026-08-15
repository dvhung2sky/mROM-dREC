"""
Merge the wind-turbine shards into wind_turbine_database.npz, stored in the SAME
layout the tunnel scripts expect -- (N, S, T) with a q_sensor load channel -- so
Arm 1/2/3 reuse red_to_nl_tunnel*.py with a one-line database swap.
"""
import glob
import numpy as np

import gen_wind_turbine_database as G

Z_HUB = G.Z_HUB
shards = sorted(glob.glob("wt_shards/wt_*.npz"))
assert shards, "no shards found"

acc = {}
for f in shards:
    d = np.load(f, allow_pickle=True)
    for k in d.files:
        if k in ("z_sensor", "param_names"):
            acc[k] = d[k]
        else:
            acc.setdefault(k, []).append(d[k])
for k in list(acc):
    if isinstance(acc[k], list):
        acc[k] = np.concatenate(acc[k], axis=0)

order = np.argsort(acc["seeds"])                      # deterministic sample order
for k, v in acc.items():
    if k not in ("z_sensor", "param_names") and len(v) == len(order):
        acc[k] = v[order]

# Drop samples the integrator could not carry.  Widening Omega into the 3P band makes a
# few samples resonate until the soil fully plasticises and Newmark runs away (1e6 m); the
# generator now flags them rather than writing them out as data.
keep = (acc["diverged"] < 0) & (acc["nonconv"] == 0)
n_all = len(keep)
print(f"divergence screen: keeping {keep.sum()}/{n_all}  "
      f"(diverged {(acc['diverged'] >= 0).sum()}, "
      f"non-converged-only {((acc['diverged'] < 0) & (acc['nonconv'] > 0)).sum()})")
for k in list(acc):
    if k not in ("z_sensor", "param_names") and len(acc[k]) == n_all:
        acc[k] = acc[k][keep]

N, T, S = acc["u_nl"].shape
z_s = acc["z_sensor"]
names = list(map(str, acc["param_names"]))
alpha = acc["params"][:, names.index("alpha_sh")]

# local wind speed at each sensor = the physical load driver (power-law shear)
shear = (np.clip(z_s, 1e-3, None)[None, :] / Z_HUB) ** alpha[:, None]      # (N, S)
q_sensor = acc["V_hub_t"][:, :, None] * shear[:, None, :]                  # (N, T, S)

def nts(a):                                            # (N,T,S) -> (N,S,T)
    return np.transpose(a, (0, 2, 1)).astype(np.float32)

dt_out = G.DT * G.DECIMATE
out = dict(
    u_nl=nts(acc["u_nl"]), u_lin=nts(acc["u_lin"]), u_red=nts(acc["u_red"]),
    q_sensor=nts(q_sensor),
    u_hub=acc["u_hub"].astype(np.float32),
    V_hub_t=acc["V_hub_t"].astype(np.float32),
    params=acc["params"], param_names=acc["param_names"],
    omegas=acc["omegas"], z_sensor=z_s,
    t=(np.arange(T) * dt_out).astype(np.float64), dt=np.float64(dt_out),
    seeds=acc["seeds"],
    ypl_max=acc["ypl_max"], th_pl=acc["th_pl"],
    n_soil_yield=acc["n_soil_yield"], n_hinge_yield=acc["n_hinge_yield"],
    nonconv=acc["nonconv"], diverged=acc["diverged"],
)
_out_db = __import__("os").environ.get("OUT_DB", "wind_turbine_database.npz")
np.savez_compressed(_out_db, **out)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


print(f"N={N}  T={T}  S={S}  dt_out={dt_out}s  record={T*dt_out:.0f}s")
print(f"z_sensor = {np.round(z_s,1)} m")
print(f"global R2  u_lin -> u_nl : {r2(out['u_nl'], out['u_lin']):+.4f}")
print(f"global R2  u_red -> u_nl : {r2(out['u_nl'], out['u_red']):+.4f}")
ps = [r2(out["u_nl"][i], out["u_red"][i]) for i in range(N)]
print(f"per-sample R2(u_red): mean={np.mean(ps):+.4f}  p10={np.percentile(ps,10):+.4f}  "
      f"min={np.min(ps):+.4f}")
print(f"soil yielded in {100*(acc['n_soil_yield']>0).mean():.0f}% of samples, "
      f"hinge in {100*(acc['n_hinge_yield']>0).mean():.0f}%")
print(f"saved {_out_db}")
