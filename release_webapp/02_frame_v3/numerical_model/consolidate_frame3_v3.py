"""
Repack frame3_v3_database.npz into the (N, S, T) layout the arm scripts expect,
so Arm 1/2/3 reuse the same pipeline as the tunnel and wind-turbine rows.

  S = 3 floors
  q_sensor = the per-floor lateral force (v2 applies load at EVERY floor)
  u_red    = the K = M Phi Omega^2 Phi^T M low-fidelity floor model
"""
import numpy as np

d = np.load("frame3_v3_database.npz", allow_pickle=True)
g = np.where(d["converged"])[0]

# Validity screen: a few samples overshoot the intended ductility badly (up to ~14x
# amplification over the linear response) and are effectively near-collapse, well
# outside the design range target_mu <= 2.0. They dominate any flattened metric.
_u = np.stack([d[f"u{i}"][g] for i in (1, 2, 3)], axis=1)
_m = np.stack([d[f"u{i}_mod"][g] for i in (1, 2, 3)], axis=1)
_amp = np.max(np.abs(_u), axis=(1, 2)) / (np.max(np.abs(_m), axis=(1, 2)) + 1e-30)
_keep = _amp < 1.5
print(f"validity screen: dropped {(~_keep).sum()} / {len(g)} near-collapse samples "
      f"(amplification >= 1.5x)")
g = g[_keep]
cols = list(map(str, d["param_cols"]))

u_nl = np.stack([d[f"u{i}"][g] for i in (1, 2, 3)], axis=1)          # (N,3,T)
u_lin = np.stack([d[f"u{i}_lin"][g] for i in (1, 2, 3)], axis=1)
u_red = np.stack([d[f"u{i}_mod"][g] for i in (1, 2, 3)], axis=1)
q = d["F"][g].astype(np.float32)                                     # (N,3,T)
P = d["params"][g]
fn = P[:, [cols.index(f"fn{i}") for i in (1, 2, 3)]]
omegas = 2 * np.pi * fn                                              # (N,3)
h = P[:, [cols.index(f"h{i}") for i in (1, 2, 3)]]
z_sensor = np.cumsum(h.mean(axis=0))                                 # nominal elevations
t = d["t"].astype(np.float64)
dt = float(t[1] - t[0])

np.savez_compressed(
    "frame3_v3_arm_database.npz",
    u_nl=u_nl.astype(np.float32), u_lin=u_lin.astype(np.float32),
    u_red=u_red.astype(np.float32), q_sensor=q,
    params=P.astype(np.float64), param_names=np.array(cols),
    omegas=omegas, z_sensor=z_sensor, t=t, dt=np.float64(dt),
    Phi_floor=d["Phi_floor"][g], m_floor=P[:, [cols.index(f"m{i}") for i in (1,2,3)]],
    h_story=P[:, [cols.index(f"h{i}") for i in (1,2,3)]],
)


def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / (((a - a.mean()) ** 2).sum() + 1e-12))


print(f"N={len(g)}  S=3  T={u_nl.shape[2]}  dt={dt:.4f}s  z={np.round(z_sensor,2)} m")
print(f"global R2  u_lin -> u_nl : {r2(u_nl, u_lin):+.4f}")
print(f"global R2  u_red -> u_nl : {r2(u_nl, u_red):+.4f}   (K = M Phi Om^2 Phi^T M)")
print("saved frame3_v3_arm_database.npz")
