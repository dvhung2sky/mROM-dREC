"""
Non-linear SDOF Database Generator — 1000 samples
==================================================
Standalone generator using a pure-numpy leapfrog solver with bilinear
kinematic-hardening (no OpenSeesPy required).

Schema saved to nonlinear_sdof_database.npz
────────────────────────────────────────────
  params    (N, 5)   [m, k, c, alpha, fy]
  F         (N, T)   force histories
  u         (N, T)   non-linear displacement
  u_lin     (N, T)   linear displacement
  t         (T,)     time vector
  fs        scalar   sampling frequency
  converged (N,)     bool

Parameter ranges
────────────────
  fₙ  ∈ [1, 8]  Hz
  m   ∈ [0.5, 5] kg
  xi  ∈ [0.02, 0.15]
  alpha ∈ [0.01, 0.15]   hardening ratio
  eta ∈ [0.1, 0.8]       yield ratio (u_y = eta * |u_lin_peak|)
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES = int(__import__("os").environ.get("N_SAMPLES", 1000))
FS        = 100.0
DT        = 1.0 / FS
N_T       = 500
t_arr     = np.arange(N_T) * DT   # 0.00 … 4.99 s
SUBSTEPS  = 4
import os as _os
ETA_LO = float(_os.environ.get("ETA_LO", 0.10))
ETA_HI = float(_os.environ.get("ETA_HI", 0.80))
OUT_DB = _os.environ.get("OUT_DB", "nonlinear_sdof_database.npz")
RNG_SEED  = 2025
rng       = np.random.default_rng(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Force generators (identical flavours to generate_sdof_database.py)
# ─────────────────────────────────────────────────────────────────────────────

def _peak_norm(signal, F_peak):
    mx = np.max(np.abs(signal))
    return signal / mx * F_peak if mx > 1e-12 else signal


def force_multisine(t, rng, F_peak):
    N_comp = 40
    freqs  = rng.uniform(0.5, 45.0, N_comp)
    amps   = rng.exponential(scale=1.0, size=N_comp)
    phases = rng.uniform(0.0, 2 * np.pi, N_comp)
    F = np.sum([a * np.sin(2 * np.pi * f * t + phi)
                for a, f, phi in zip(amps, freqs, phases)], axis=0)
    return _peak_norm(F, F_peak)


def force_chirp_noise(t, rng, F_peak):
    from scipy.signal import chirp as sp_chirp
    f0, f1    = rng.uniform(0.5, 5.0), rng.uniform(20.0, 45.0)
    chirp_sig = sp_chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear')
    white  = rng.standard_normal(len(t))
    freqs  = np.fft.rfftfreq(len(t), d=DT)
    spec   = np.fft.rfft(white)
    f_lo   = rng.uniform(2.0, 8.0)
    f_hi   = rng.uniform(20.0, 45.0)
    spec  *= (freqs >= f_lo) & (freqs <= f_hi)
    noise  = np.fft.irfft(spec, n=len(t))
    w_c = rng.uniform(0.4, 0.8)
    F = w_c * chirp_sig / (np.std(chirp_sig) + 1e-12) + \
        (1 - w_c) * noise / (np.std(noise) + 1e-12)
    return _peak_norm(F, F_peak)


def force_burst_train(t, rng, F_peak):
    F = np.zeros_like(t)
    for _ in range(int(rng.integers(3, 7))):
        t_c   = rng.uniform(0.2 * t[-1], 0.9 * t[-1])
        sigma = rng.uniform(0.1, 0.6)
        freq  = rng.uniform(1.0, 40.0)
        amp   = rng.uniform(0.3, 1.0)
        env   = np.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        F    += amp * env * np.sin(2 * np.pi * freq * t)
    return _peak_norm(F, F_peak)


_FORCE_GENS = [force_multisine, force_chirp_noise, force_burst_train]


def generate_force(t, rng):
    F_peak = rng.uniform(5.0, 30.0)
    return rng.choice(_FORCE_GENS)(t, rng, F_peak)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy leapfrog (velocity-Störmer-Verlet) solver
# ─────────────────────────────────────────────────────────────────────────────

def sdof_solver(m, k, c, F_arr, dt, substeps, bilinear=True, fy=None, alpha=None):
    """
    Leapfrog SDOF solver with optional bilinear kinematic hardening.
    Returns displacement array (N_T,) or None on divergence.
    """
    N_t    = len(F_arr)
    dt_sub = dt / substeps
    uy     = (fy / k) if bilinear else np.inf

    u, v, up, fs = 0.0, 0.0, 0.0, 0.0
    a = (F_arr[0] - c * v - fs) / m

    u_out   = np.zeros(N_t, dtype=np.float32)
    F_prev  = float(F_arr[0])

    for t_idx in range(1, N_t):
        F_now = float(F_arr[t_idx])
        for ss in range(substeps):
            F_cur = F_prev + (F_now - F_prev) * (ss + 1) / substeps
            vh = v + 0.5 * dt_sub * a
            un = u + dt_sub * vh

            # Return-mapping for bilinear kinematic hardening
            if bilinear:
                uel = un - up
                if uel > uy:
                    fs = k * uy + alpha * k * (uel - uy)
                    up += (1.0 - alpha) * (uel - uy)
                elif uel < -uy:
                    fs = -k * uy + alpha * k * (uel + uy)
                    up += (1.0 - alpha) * (uel + uy)
                else:
                    fs = k * uel
            else:
                fs = k * un

            a_new = (F_cur - c * vh - fs) / m
            v = vh + 0.5 * dt_sub * a_new
            a = a_new
            u = un

        F_prev = F_now
        u_out[t_idx] = u

    if not (np.isfinite(u_out).all() and np.abs(u_out).max() < 1e4):
        return None
    return u_out


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Non-linear SDOF Database Generator  (pure-numpy, 1000 samples)")
print(f"  fs = {FS} Hz,  N_t = N_T,  T = {(N_T-1)*DT:.2f} s,  substeps = {SUBSTEPS}")
print("=" * 65)

params_list    = []
F_list         = []
u_nl_list      = []
u_lin_list     = []
converged_list = []

n_failed = 0
t_start  = time.time()
attempt  = 0

while len(params_list) < N_SAMPLES:
    attempt += 1

    # Sample structural parameters
    fn  = rng.uniform(1.0, 8.0)
    m   = rng.uniform(0.5, 5.0)
    k   = m * (2.0 * np.pi * fn) ** 2
    xi  = rng.uniform(0.02, 0.15)
    c   = 2.0 * xi * np.sqrt(k * m)

    # Sample nonlinear parameters
    alpha = rng.uniform(0.01, 0.15)
    eta   = rng.uniform(ETA_LO, ETA_HI)   # yield disp / linear peak; higher = milder

    # Generate force
    F = generate_force(t_arr, rng).astype(np.float64)

    # Linear response
    u_lin = sdof_solver(m, k, c, F, DT, SUBSTEPS, bilinear=False)
    if u_lin is None:
        n_failed += 1
        continue

    # Yield displacement from linear peak
    u_lin_peak = float(np.max(np.abs(u_lin)))
    uy  = max(eta * u_lin_peak, 1e-6)
    fy  = k * uy

    # Nonlinear response
    u_nl = sdof_solver(m, k, c, F, DT, SUBSTEPS, bilinear=True, fy=fy, alpha=alpha)

    converged = u_nl is not None
    if not converged:
        u_nl = np.full(N_T, np.nan, dtype=np.float32)
        n_failed += 1

    params_list.append([m, k, c, alpha, fy])
    F_list.append(F.astype(np.float32))
    u_nl_list.append(u_nl)
    u_lin_list.append(u_lin.astype(np.float32))
    converged_list.append(converged)

    n_done = len(params_list)
    if n_done % 50 == 0 or n_done == N_SAMPLES:
        elapsed   = time.time() - t_start
        rate      = n_done / elapsed if elapsed > 0 else 1
        remaining = (N_SAMPLES - n_done) / rate
        filled    = int(30 * n_done / N_SAMPLES)
        bar       = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {n_done}/{N_SAMPLES}  "
            f"{elapsed:.1f}s elapsed  ~{remaining:.0f}s left  "
            f"(failed: {n_failed})"
        )
        sys.stdout.flush()

elapsed_total = time.time() - t_start
n_converged   = sum(converged_list)
print(f"\n\n  Done — {n_converged}/{N_SAMPLES} converged in {elapsed_total:.1f}s "
      f"({n_failed} failures out of {attempt} attempts)")

# ─────────────────────────────────────────────────────────────────────────────
# Save database
# ─────────────────────────────────────────────────────────────────────────────
params_arr    = np.array(params_list,    dtype=np.float32)   # (N, 5)
F_arr_out     = np.array(F_list,         dtype=np.float32)   # (N, T)
u_nl_arr      = np.array(u_nl_list,      dtype=np.float32)   # (N, T)
u_lin_arr     = np.array(u_lin_list,     dtype=np.float32)   # (N, T)
converged_arr = np.array(converged_list, dtype=bool)         # (N,)

np.savez_compressed(
    OUT_DB,
    params    = params_arr,
    F         = F_arr_out,
    u         = u_nl_arr,
    u_lin     = u_lin_arr,
    t         = t_arr.astype(np.float32),
    fs        = np.float32(FS),
    converged = converged_arr,
)
print("  Saved → nonlinear_sdof_database.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
mask         = converged_arr
u_nl_peaks   = np.max(np.abs(u_nl_arr[mask]),  axis=1)
u_lin_peaks  = np.max(np.abs(u_lin_arr[mask]), axis=1)
ratio        = u_nl_peaks / (u_lin_peaks + 1e-12)

print("\n" + "=" * 65)
print("  Database summary")
print("=" * 65)
print(f"  Samples         : {N_SAMPLES}")
print(f"  Converged       : {n_converged}")
print(f"  Signal length   : {N_T} pts @ {FS} Hz = {(N_T-1)*DT:.2f} s")
print(f"  m   range       : {params_arr[:,0].min():.3f} – {params_arr[:,0].max():.3f} kg")
print(f"  k   range       : {params_arr[:,1].min():.1f} – {params_arr[:,1].max():.1f} N/m")
print(f"  c   range       : {params_arr[:,2].min():.4f} – {params_arr[:,2].max():.4f} N·s/m")
print(f"  α   range       : {params_arr[:,3].min():.4f} – {params_arr[:,3].max():.4f}")
print(f"  f_y range       : {params_arr[:,4].min():.6f} – {params_arr[:,4].max():.4f} N")
fn_arr = np.sqrt(params_arr[:, 1] / params_arr[:, 0]) / (2 * np.pi)
xi_arr = params_arr[:, 2] / (2 * np.sqrt(params_arr[:, 1] * params_arr[:, 0]))
print(f"  fₙ  range       : {fn_arr.min():.2f} – {fn_arr.max():.2f} Hz")
print(f"  ξ   range       : {xi_arr.min():.3f} – {xi_arr.max():.3f}")
print(f"\n  For converged samples:")
print(f"  peak |u_nl|     : {u_nl_peaks.min():.6f} – {u_nl_peaks.max():.6f} m")
print(f"  peak |u_lin|    : {u_lin_peaks.min():.6f} – {u_lin_peaks.max():.6f} m")
print(f"  ratio nl/lin    : {ratio.min():.3f} – {ratio.max():.3f}  (median {np.median(ratio):.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# Preview plot — 6 random converged samples
# ─────────────────────────────────────────────────────────────────────────────
converged_idx = np.where(converged_arr)[0]
plot_idx = rng.choice(converged_idx, size=min(6, len(converged_idx)), replace=False)
plot_idx.sort()

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Non-linear SDOF (Bilinear Elasto-Plastic) — 6 Random Samples  [N=1000]",
             fontsize=13, fontweight='bold')
gs = GridSpec(6, 3, figure=fig, hspace=0.08, wspace=0.35)

for row, i in enumerate(plot_idx):
    F_i     = F_arr_out[i]
    u_nl_i  = u_nl_arr[i]
    u_lin_i = u_lin_arr[i]
    m_i, k_i, c_i, alpha_i, fy_i = params_arr[i]
    fn_i  = np.sqrt(k_i / m_i) / (2 * np.pi)
    xi_i  = c_i / (2 * np.sqrt(k_i * m_i))
    uy_i  = fy_i / k_i

    ax_f = fig.add_subplot(gs[row, 0])
    ax_f.plot(t_arr, F_i, color='steelblue', linewidth=0.7)
    ax_f.axhline(0, color='k', linewidth=0.4)
    ax_f.set_ylabel('F [N]', fontsize=7)
    ax_f.tick_params(labelsize=6)
    if row == 0:
        ax_f.set_title('Force F(t)', fontsize=9)
    if row < 5:
        ax_f.set_xticklabels([])

    ax_u = fig.add_subplot(gs[row, 1])
    ax_u.plot(t_arr, u_lin_i, color='gray',      linewidth=0.6, alpha=0.7, label='linear')
    ax_u.plot(t_arr, u_nl_i,  color='firebrick', linewidth=0.7, label='non-linear')
    ax_u.axhline(0,    color='k',      linewidth=0.4)
    ax_u.axhline( uy_i, color='orange', linewidth=0.5, linestyle='--', alpha=0.6)
    ax_u.axhline(-uy_i, color='orange', linewidth=0.5, linestyle='--', alpha=0.6)
    ax_u.set_ylabel('u [m]', fontsize=7)
    ax_u.tick_params(labelsize=6)
    label = (f"fₙ={fn_i:.2f}Hz  ξ={xi_i:.3f}\n"
             f"α={alpha_i:.3f}  u_y={uy_i:.5f}")
    ax_u.text(0.98, 0.96, label, transform=ax_u.transAxes,
              fontsize=6, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='lightyellow', alpha=0.85))
    if row == 0:
        ax_u.set_title('Displacement (gray=linear, red=non-linear)', fontsize=9)
        ax_u.legend(fontsize=6, loc='upper left')
    if row < 5:
        ax_u.set_xticklabels([])

    ax_h = fig.add_subplot(gs[row, 2])
    v_nl = np.gradient(u_nl_i.astype(np.float64), DT)
    a_nl = np.gradient(v_nl, DT)
    fs_i = F_i.astype(np.float64) - m_i * a_nl - c_i * v_nl
    ax_h.plot(u_nl_i, fs_i, color='darkgreen', linewidth=0.4, alpha=0.7)
    ax_h.axhline(0,    color='k',      linewidth=0.3)
    ax_h.axvline(0,    color='k',      linewidth=0.3)
    ax_h.axhline( fy_i, color='orange', linewidth=0.5, linestyle='--', alpha=0.6)
    ax_h.axhline(-fy_i, color='orange', linewidth=0.5, linestyle='--', alpha=0.6)
    ax_h.set_ylabel('f_s [N]', fontsize=7)
    if row == 5:
        ax_h.set_xlabel('u [m]', fontsize=7)
    ax_h.tick_params(labelsize=6)
    if row == 0:
        ax_h.set_title('Hysteresis Loop (f_s vs u)', fontsize=9)
    if row < 5:
        ax_h.set_xticklabels([])

fig.text(0.35, 0.02, 'Time [s]', ha='center', fontsize=10)
# plt.savefig('nonlinear_sdof_database_preview.png', dpi=140, bbox_inches='tight')
print("\n  Preview figure → nonlinear_sdof_database_preview.png")
plt.show()
