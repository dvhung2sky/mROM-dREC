"""
Multi-Fidelity GRU for Non-linear SDOF  (v2 — drift-aware)
============================================================
Core idea:
  u_nonlinear(t) = u_linear(t) + Δu(t)

  The linear solution u_lin is available cheaply (from the linear SDOF database).
  The GRU learns ONLY the residual  Δu = u_nl - u_lin.
  Final prediction = u_lin + GRU(Δu).

v2 fixes over v1:
  1. Per-sample normalisation of Δu (not global z-score — residuals vary hugely)
  2. Tukey flat-top taper in inference (preserves DC / plastic drift)
  3. Sequential hidden-state carryover in inference (drift is cumulative)
  4. 8-dim static: [m, k, c, fn, ξ, α, fy, η]  (η = ductility ratio)
  5. OneCycleLR for fast 50-epoch convergence
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Setup
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

L_out = 100
L_in  = 100
L_all = 200
sliding_step = 15

HIDDEN       = 160      # 128 → 160: more capacity for richer 9-channel input
N_GRU_LAYERS = 2

PEAK_ALPHA    = 0.5
CTX_NOISE_STD = 0.05         # low noise — residual signal is delicate
SPEC_ALPHA    = 0.1
SPEC_WARMUP   = 10

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load non-linear SDOF database
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
data      = np.load(_os.environ.get("DB", "nonlinear_sdof_database.npz"))
params    = data["params"].astype(np.float64)     # (N, 5) : [m, k, c, alpha, fy]
F_raw     = data["F"].astype(np.float64)          # (N, T)
u_nl_raw  = data["u"].astype(np.float64)          # (N, T) non-linear response
u_lin_raw = data["u_lin"].astype(np.float64)      # (N, T) linear response
t         = data["t"].astype(np.float64)          # (T,)
converged = data["converged"]                      # (N,)
N_full, T = F_raw.shape
dt_data   = float(t[1] - t[0])

# Filter out non-converged samples
good = np.where(converged)[0]
params    = params[good]
F_raw     = F_raw[good]
u_nl_raw  = u_nl_raw[good]
u_lin_raw = u_lin_raw[good]
N         = len(good)

print(f"Loaded {N}/{N_full} converged samples  |  seq_len = {T}  |  dt = {dt_data:.4f} s")
print(f"L_all={L_all}  L_in={L_in}  L_out={L_out}  sliding_step={sliding_step}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Train / val split  (50/50)
# ─────────────────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
idx     = rng.permutation(N)
idx     = rng.permutation(N)
_ntr, _nva = int(0.70 * N), int(0.15 * N)
idx_tr  = idx[:_ntr]
idx_val = idx[_ntr:_ntr + _nva]      # model selection only
idx_te  = idx[_ntr + _nva:]          # final metrics only
print(f"70/15/15 -> train {len(idx_tr)}  val {len(idx_val)}  test {len(idx_te)}")
print(f"Train : {len(idx_tr)}   Val : {len(idx_val)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Normalisation
# ─────────────────────────────────────────────────────────────────────────────
m_arr     = params[:, 0]
k_arr     = params[:, 1]
c_arr     = params[:, 2]
alpha_arr = params[:, 3]
fy_arr    = params[:, 4]

# 3a. Static physics: [m, k, c] → log → z-score
phys_mkc       = np.column_stack([m_arr, k_arr, c_arr])
log_phys       = np.log(phys_mkc + 1e-8)
phys_mean      = log_phys[idx_tr].mean(0)
phys_std       = log_phys[idx_tr].std(0) + 1e-8
phys_norm_base = ((log_phys - phys_mean) / phys_std).astype(np.float32)

# Derived: fn, ξ
fn_arr  = np.sqrt(k_arr / m_arr) / (2 * np.pi)
xi_arr  = c_arr / (2 * np.sqrt(k_arr * m_arr))
derived = np.column_stack([fn_arr, xi_arr])
log_der = np.log(derived + 1e-8)
der_mean = log_der[idx_tr].mean(0)
der_std  = log_der[idx_tr].std(0) + 1e-8
derived_norm = ((log_der - der_mean) / der_std).astype(np.float32)

# Non-linear params: [alpha, fy] → log → z-score
nl_params = np.column_stack([alpha_arr, fy_arr])
log_nl    = np.log(nl_params + 1e-8)
nl_mean   = log_nl[idx_tr].mean(0)
nl_std    = log_nl[idx_tr].std(0) + 1e-8
nl_norm   = ((log_nl - nl_mean) / nl_std).astype(np.float32)

# Ductility ratio η = u_y / u_lin_peak
u_lin_peaks = np.max(np.abs(u_lin_raw), axis=1)
uy_arr      = fy_arr / k_arr
eta_arr     = uy_arr / (u_lin_peaks + 1e-12)
log_eta     = np.log(eta_arr + 1e-8)
eta_mean    = log_eta[idx_tr].mean()
eta_std     = log_eta[idx_tr].std() + 1e-8
eta_norm    = ((log_eta - eta_mean) / eta_std).astype(np.float32)

# Combined static (base 8-dim); a 9th feature (net_drift_bias) is appended
# below once duct_pos/duct_neg are available.
phys_norm = np.concatenate([phys_norm_base, derived_norm, nl_norm,
                            eta_norm[:, None]], axis=1)
print(f"  η range : {eta_arr.min():.4f} – {eta_arr.max():.4f}  "
      f"(mean={eta_arr.mean():.3f})")

# 3b. Force : per-sample z-score
F_mu    = F_raw.mean(1, keepdims=True)
F_sigma = F_raw.std(1, keepdims=True) + 1e-8
F_norm  = ((F_raw - F_mu) / F_sigma).astype(np.float32)

# 3c. Target: residual Δu normalised by u_lin's per-sample std
#     Previous version used Δu's own stats → DATA LEAKAGE at inference
#     (Δu is unknown when u_nl is what we're predicting).
#     u_lin's std is known at inference. Δu magnitude scales with u_lin
#     for yielding systems, so this is physically motivated.
delta_u     = u_nl_raw - u_lin_raw                                   # (N, T)
ulin_sigma  = u_lin_raw.std(1, keepdims=True) + 1e-10                # (N, 1) ← known at inference
delta_norm  = (delta_u / ulin_sigma).astype(np.float32)              # (N, T) dimensionless
ulin_norm   = (u_lin_raw / ulin_sigma).astype(np.float32)            # (N, T) same scale → commensurable

# Plastic-drift physics feature (SIGN-AWARE): split cumulative max into
# positive-side and negative-side excursions.  Previous single ductility
# channel was unsigned cummax|u_lin|/u_y — carries magnitude but destroys
# direction.  The asymmetry (duct_pos - duct_neg) tells the model which
# side yielded first/most → correct plastic drift direction.
pos_part   = np.maximum(u_lin_raw, 0.0)                              # (N, T) ≥ 0
neg_part   = np.maximum(-u_lin_raw, 0.0)                             # (N, T) ≥ 0
cummax_pos = np.maximum.accumulate(pos_part, axis=1)                 # (N, T)
cummax_neg = np.maximum.accumulate(neg_part, axis=1)                 # (N, T)
duct_pos   = np.log1p(np.clip(cummax_pos / (uy_arr[:, None] + 1e-12),
                               0, 100.0)).astype(np.float32)
duct_neg   = np.log1p(np.clip(cummax_neg / (uy_arr[:, None] + 1e-12),
                               0, 100.0)).astype(np.float32)
dp_std     = duct_pos[idx_tr].std() + 1e-8
dn_std     = duct_neg[idx_tr].std() + 1e-8
duct_pos   = duct_pos / dp_std
duct_neg   = duct_neg / dn_std

print(f"Δu/σ_u_lin ratio  |  range: "
      f"[{delta_norm.min():.3f}, {delta_norm.max():.3f}]  "
      f"std={delta_norm.std():.3f}")
print(f"duct_pos |  range: [{duct_pos.min():.3f}, {duct_pos.max():.3f}]  "
      f"duct_neg |  range: [{duct_neg.min():.3f}, {duct_neg.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3d. Cheap EPP approximation — DRIFT-DIRECTION-AWARE physics prior
# ─────────────────────────────────────────────────────────────────────────────
# The envelope features (duct_pos/duct_neg, argmax|u_lin|) sometimes predict
# the WRONG drift direction for lightly-damped systems: a strong positive
# pulse can rebound hard enough to yield negatively and establish a NEGATIVE
# drift, even though u_lin peaks positively.  The fix is a simple pure-EPP
# (α=0) solver that explicitly tracks plastic offset. Pure EPP ignores
# kinematic hardening but is guaranteed to predict the correct drift SIGN.
def sdof_bilinear_solver_batch(params, F_arr, dt, substeps=4):
    """Bilinear kinematic-hardening SDOF solver with substepped central-difference.
    Returns (u_approx, up_traj) where up_traj[i, t] is the plastic-offset
    trajectory — the running cumulative plastic strain, which DIRECTLY
    governs drift. Feeding this into the GRU is richer than just the
    terminal value: at every time step the network sees the current
    plastic-drift estimate."""
    N_, T_ = F_arr.shape
    u_out  = np.zeros((N_, T_), dtype=np.float32)
    up_out = np.zeros((N_, T_), dtype=np.float32)
    dt_sub = dt / substeps
    for i in range(N_):
        m_, k_, c_, alpha_, fy_ = params[i]
        uy_ = fy_ / k_
        pos, v, u_p = 0.0, 0.0, 0.0
        a = F_arr[i, 0] / m_
        hist   = np.zeros(T_, dtype=np.float32)
        uphist = np.zeros(T_, dtype=np.float32)
        F_prev = F_arr[i, 0]
        for t_ in range(1, T_):
            F_now = F_arr[i, t_]
            for ss in range(substeps):
                F_cur = F_prev + (F_now - F_prev) * (ss + 1) / substeps
                v_half = v + 0.5 * dt_sub * a
                pos_new = pos + dt_sub * v_half
                u_el = pos_new - u_p
                if u_el > uy_:
                    fs = k_ * uy_ + alpha_ * k_ * (u_el - uy_)
                    u_p += (1.0 - alpha_) * (u_el - uy_)
                elif u_el < -uy_:
                    fs = -k_ * uy_ + alpha_ * k_ * (u_el + uy_)
                    u_p += (1.0 - alpha_) * (u_el + uy_)
                else:
                    fs = k_ * u_el
                a_new = (F_cur - c_ * v_half - fs) / m_
                v = v_half + 0.5 * dt_sub * a_new
                a, pos = a_new, pos_new
            F_prev = F_now
            hist[t_]   = pos
            uphist[t_] = u_p
        if not np.isfinite(hist).all() or np.abs(hist).max() > 10.0:
            hist[:]   = 0.0
            uphist[:] = 0.0
        u_out[i]  = hist
        up_out[i] = uphist
    return u_out, up_out


print("Running cheap bilinear-EPP solver (drift-aware physics prior) …")
t_epp = time.perf_counter()
u_approx, up_approx = sdof_bilinear_solver_batch(params, F_raw, dt_data, substeps=4)
print(f"  Bilinear solver time : {time.perf_counter() - t_epp:.1f}s  "
      f"u_approx range=[{u_approx.min():.4f}, {u_approx.max():.4f}]  "
      f"u_p range=[{up_approx.min():.4f}, {up_approx.max():.4f}]")

# delta_approx: what the cheap solver predicts as the residual Δu.
# up_approx: plastic-offset trajectory (running drift estimate).
# Both normalised by u_lin's per-sample σ for unit consistency.
delta_approx_norm = ((u_approx  - u_lin_raw) / ulin_sigma).astype(np.float32)
up_approx_norm    = ( up_approx              / ulin_sigma).astype(np.float32)
print(f"delta_approx / σ  |  std={delta_approx_norm.std():.3f}")
print(f"up_approx    / σ  |  std={up_approx_norm.std():.3f}  "
      f"range=[{up_approx_norm.min():.3f}, {up_approx_norm.max():.3f}]")

# 9th static feature = net drift bias (from u_approx terminal, sign-correct).
u_app_scale   = (ulin_sigma.squeeze() + 1e-10)
net_drift_bias = (u_approx[:, -1] / u_app_scale).astype(np.float32)
ndb_std        = net_drift_bias[idx_tr].std() + 1e-8
net_drift_bias = (net_drift_bias / ndb_std).astype(np.float32)
phys_norm      = np.concatenate([phys_norm, net_drift_bias[:, None]], axis=1)  # (N, 9)
N_STATIC       = 9
print(f"net_drift_bias    |  range: "
      f"[{net_drift_bias.min():.3f}, {net_drift_bias.max():.3f}]")
print(f"Static features : [m, k, c, fn, ξ, α, fy, η, net_drift]  ({N_STATIC}-dim)")

# Quick check: how big is the residual vs the linear solution?
lin_rms = np.sqrt(np.mean(u_lin_raw[idx_tr]**2))
res_rms = np.sqrt(np.mean(delta_u[idx_tr]**2))
print(f"RMS u_lin={lin_rms:.6f}  RMS Δu={res_rms:.6f}  ratio={res_rms/(lin_rms+1e-12):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build sliding-window dataset
# ─────────────────────────────────────────────────────────────────────────────
NET_DRIFT_IDX = N_STATIC - 1     # index of net_drift_bias in phys_norm

def make_windows(sample_indices, flip=False):
    """
    Time-series input per window  (L_all, 9):
      [F_norm, u_prev_delta, u_lin_norm, delta_approx, up_approx,
       duct_pos, duct_neg, sin_phase, cos_phase]

    up_approx = plastic-offset trajectory from the cheap bilinear solver —
    a running estimate of cumulative plastic drift at each time step.

    If flip=True: bilinear EPP sign-symmetry negates F, u_lin, u_nl, u_approx,
    u_p. duct_pos ↔ duct_neg swap, net_drift_bias negates.
    """
    Xt_list, Xs_list, Y_list = [], [], []
    t_local = np.arange(L_all, dtype=np.float32) * dt_data
    s       = -1.0 if flip else 1.0

    for si in sample_indices:
        F_pad      = np.concatenate([np.zeros(L_in, dtype=np.float32), s * F_norm[si]])
        delta_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), s * delta_norm[si]])
        ulin_pad   = np.concatenate([np.zeros(L_in, dtype=np.float32), s * ulin_norm[si]])
        dapx_pad   = np.concatenate([np.zeros(L_in, dtype=np.float32), s * delta_approx_norm[si]])
        upapx_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), s * up_approx_norm[si]])
        if flip:
            dpos_pad = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_neg[si]])
            dneg_pad = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_pos[si]])
        else:
            dpos_pad = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_pos[si]])
            dneg_pad = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_neg[si]])
        p = phys_norm[si].copy()
        if flip:
            p[NET_DRIFT_IDX] = -p[NET_DRIFT_IDX]
        p = p[None, :]

        phase  = (2 * np.pi * fn_arr[si] * t_local).astype(np.float32)
        sin_ph = np.sin(phase)
        cos_ph = np.cos(phase)

        for start in range(0, T, sliding_step):
            end = start + L_all
            if end > len(F_pad):
                break

            F_win     = F_pad[start:end]
            delta_win = delta_pad[start:end]
            ulin_win  = ulin_pad[start:end]
            dapx_win  = dapx_pad[start:end]
            upapx_win = upapx_pad[start:end]
            dpos_win  = dpos_pad[start:end]
            dneg_win  = dneg_pad[start:end]

            u_prev = np.concatenate([delta_win[:L_in],
                                     np.zeros(L_out, dtype=np.float32)])

            x_time = np.column_stack([F_win, u_prev, ulin_win,
                                      sin_ph, cos_ph])
            y_win  = delta_win[L_in:]

            Xt_list.append(x_time)
            Xs_list.append(p)
            Y_list.append(y_win)

    return (np.array(Xt_list, dtype=np.float32),
            np.array(Xs_list, dtype=np.float32),
            np.array(Y_list,  dtype=np.float32))


# Sign-flip augmentation: bilinear EPP is sign-symmetric, so doubling training
# data with mirrored samples is free and teaches the network this symmetry.
print("Building training windows (original) …")
Xa, Sa, Ya = make_windows(idx_tr, flip=False)
print("Building training windows (sign-flipped augmentation) …")
Xb, Sb, Yb = make_windows(idx_tr, flip=True)
X_tr = np.concatenate([Xa, Xb], axis=0)
S_tr = np.concatenate([Sa, Sb], axis=0)
Y_tr = np.concatenate([Ya, Yb], axis=0)
print("Building validation windows …")
X_va, S_va, Y_va = make_windows(idx_val, flip=False)

print(f"Train windows : {len(X_tr)}   Val windows : {len(X_va)}")
print(f"X_time shape : {X_tr.shape}   X_static shape : {S_tr.shape}   Y shape : {Y_tr.shape}")

N_TIME_CH = 5   # NO EPP: [F, u_prev_delta, u_lin, sin_phase, cos_phase]
                # (delta_approx, up_approx, duct_pos, duct_neg removed)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, X_time, X_static, Y):
        self.X_time   = torch.from_numpy(X_time)
        self.X_static = torch.from_numpy(X_static)
        self.Y        = torch.from_numpy(Y)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        return self.X_time[i], self.X_static[i], self.Y[i]

BATCH    = 64
dl_train = DataLoader(WindowDataset(X_tr, S_tr, Y_tr),
                      batch_size=BATCH, shuffle=True,  drop_last=False)
dl_val   = DataLoader(WindowDataset(X_va, S_va, Y_va),
                      batch_size=BATCH, shuffle=False, drop_last=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Model
# ─────────────────────────────────────────────────────────────────────────────
class FiLMStaticEncoder(nn.Module):
    """FiLM modulation with 8-dim static features."""
    def __init__(self, n_static=8, n_time=5, dropout=0.1):
        super().__init__()
        hidden = max(n_static * 2, 64)
        self.film_mlp = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_time * 2),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, L_all),
            nn.Tanh(),
            nn.Linear(L_all, L_all),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_time, x_static):
        s = x_static.squeeze(1)
        film = self.film_mlp(s).unsqueeze(1)
        gamma, beta = film.chunk(2, dim=-1)
        x_film = (1.0 + gamma) * x_time + beta
        static_ch = self.dropout(self.static_proj(s))
        static_ch = static_ch.unsqueeze(2)
        return torch.cat([x_film, static_ch], dim=2)


class RepresentationLearning(nn.Module):
    """GRU backbone."""
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        last   = out[:, -L_out:, :]
        return last, h


class PredictionHead(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, 1, bias=False)
        nn.init.xavier_uniform_(self.shared[0].weight, gain=0.1)
        nn.init.zeros_(self.head.weight)

    def forward(self, feat):
        return self.head(self.shared(feat)).squeeze(-1)


class ForceSkip(nn.Module):
    """Quasi-static skip: alpha(params) * F_norm → residual prediction."""
    def __init__(self, n_static=8):
        super().__init__()
        self.gain = nn.Sequential(
            nn.Linear(n_static, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, F_out, s_flat):
        alpha = self.gain(s_flat)
        return alpha * F_out


GRU_INPUT_SIZE = N_TIME_CH + 1   # 8 time + 1 static channel = 9

film_agg       = FiLMStaticEncoder(n_static=N_STATIC, n_time=N_TIME_CH, dropout=0.1).to(device)
representation = RepresentationLearning(input_size=GRU_INPUT_SIZE, hidden_size=HIDDEN,
                                         num_layers=N_GRU_LAYERS, dropout=0.15).to(device)
pred_head      = PredictionHead(hidden_size=HIDDEN).to(device)
force_skip     = ForceSkip(n_static=N_STATIC).to(device)

all_params_list = (list(film_agg.parameters()) +
                   list(representation.parameters()) +
                   list(pred_head.parameters()) +
                   list(force_skip.parameters()))
n_params = sum(p.numel() for p in all_params_list if p.requires_grad)
print(f"\nFiLMStaticEncoder      : {sum(p.numel() for p in film_agg.parameters() if p.requires_grad):,}")
print(f"RepresentationLearning : {sum(p.numel() for p in representation.parameters() if p.requires_grad):,}  (GRU hidden={HIDDEN}, layers={N_GRU_LAYERS})")
print(f"PredictionHead         : {sum(p.numel() for p in pred_head.parameters() if p.requires_grad):,}")
print(f"ForceSkip              : {sum(p.numel() for p in force_skip.parameters() if p.requires_grad):,}")
print(f"Total parameters       : {n_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Loss, optimiser, scheduler
# ─────────────────────────────────────────────────────────────────────────────
N_EPOCHS = 15
LR       = 2e-3
PATIENCE = 50    # effectively disabled for 50 epochs

criterion = nn.MSELoss()


def spectral_envelope_loss(pred, true, smooth_k=5):
    P = torch.fft.rfft(pred, dim=-1).abs()
    T_spec = torch.fft.rfft(true, dim=-1).abs()
    P = F.avg_pool1d(P.unsqueeze(1), kernel_size=smooth_k,
                     stride=1, padding=smooth_k // 2).squeeze(1)
    T_spec = F.avg_pool1d(T_spec.unsqueeze(1), kernel_size=smooth_k,
                          stride=1, padding=smooth_k // 2).squeeze(1)
    P = P / (P.amax(dim=-1, keepdim=True) + 1e-8)
    T_spec = T_spec / (T_spec.amax(dim=-1, keepdim=True) + 1e-8)
    log_P = torch.log(P + 1e-4)
    log_T = torch.log(T_spec + 1e-4)
    return F.huber_loss(log_P, log_T, delta=1.0)


def peak_amplitude_loss(pred, true):
    tau = 0.1
    w_true = F.softmax(true.abs() / tau, dim=-1)
    w_pred = F.softmax(pred.abs() / tau, dim=-1)
    soft_peak = (w_pred * pred.abs()).sum(-1) - (w_true * true.abs()).sum(-1)
    max_err = (pred.abs().max(dim=-1).values - true.abs().max(dim=-1).values).abs()
    weight = (true.abs() + 1e-3).pow(1.5)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    weighted_mse = (weight * (pred - true)**2).sum(dim=-1).mean()
    return 0.4 * soft_peak.mean() + 0.4 * max_err.mean() + 0.2 * weighted_mse


optimiser = torch.optim.AdamW(all_params_list, lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=LR, epochs=N_EPOCHS,
    steps_per_epoch=len(dl_train),
    pct_start=0.3, anneal_strategy="cos")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + 1e-12)).item()


def set_mode(train):
    film_agg.train(train)
    representation.train(train)
    pred_head.train(train)
    force_skip.train(train)


def run_epoch(loader, train=True, cur_epoch=0):
    set_mode(train)
    total_mse, preds, trues = 0.0, [], []
    spec_w = min(1.0, cur_epoch / SPEC_WARMUP) * SPEC_ALPHA if cur_epoch > 0 else 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb_time, xb_static, yb in loader:
            xb_time   = xb_time.to(device)
            xb_static = xb_static.to(device)
            yb        = yb.to(device)

            if train:
                noise = torch.randn(xb_time.shape[0], L_in, device=device) * CTX_NOISE_STD
                xb_time = xb_time.clone()
                xb_time[:, :L_in, 1] = xb_time[:, :L_in, 1] + noise

            s_flat = xb_static.squeeze(1)
            F_out  = xb_time[:, -L_out:, 0]
            x      = film_agg(xb_time, xb_static)

            feat, _ = representation(x)
            pred = pred_head(feat) + force_skip(F_out, s_flat)

            mse_loss = criterion(pred, yb)
            pk_loss  = peak_amplitude_loss(pred, yb)
            if spec_w > 0:
                spec_loss = spectral_envelope_loss(pred, yb)
            else:
                spec_loss = torch.tensor(0.0, device=device)
            # Terminal-drift loss: penalise wrong mean of last 25% of window.
            # Directly attacks the failure mode where predicted drift has the
            # wrong sign (R² is dominated by DC / drift error).
            tail = L_out // 4
            drift_loss = F.mse_loss(pred[:, -tail:].mean(-1),
                                    yb[:, -tail:].mean(-1))
            loss = (mse_loss + PEAK_ALPHA * pk_loss + spec_w * spec_loss
                    + 0.3 * drift_loss)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params_list, 1.0)
                optimiser.step()
                scheduler.step()       # OneCycleLR steps per batch
            total_mse += mse_loss.item() * len(xb_time)
            preds.append(pred.detach().cpu())
            trues.append(yb.detach().cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return total_mse / len(loader.dataset), r2_score(trues, preds)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}
best_val, best_epoch, patience_ctr = float("inf"), 0, 0

t_train_wall = time.perf_counter()
t_train_cpu  = time.process_time()

print("\n" + "─" * 72)
print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  "
      f"{'Train R²':>9}  {'Val R²':>9}  {'LR':>9}")
print("─" * 72)

for epoch in range(1, N_EPOCHS + 1):
    tr_loss, tr_r2 = run_epoch(dl_train, train=True, cur_epoch=epoch)
    va_loss, va_r2 = run_epoch(dl_val,   train=False, cur_epoch=epoch)
    current_lr = optimiser.param_groups[0]["lr"]

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_r2"].append(tr_r2)
    history["val_r2"].append(va_r2)

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {tr_loss:>10.5f}  {va_loss:>10.5f}  "
              f"{tr_r2:>9.4f}  {va_r2:>9.4f}  {current_lr:>9.2e}")

    if va_loss < best_val:
        best_val, best_epoch, patience_ctr = va_loss, epoch, 0
        torch.save({
            "film_agg_state":       film_agg.state_dict(),
            "representation_state": representation.state_dict(),
            "pred_head_state":      pred_head.state_dict(),
            "force_skip_state":     force_skip.state_dict(),
            "epoch":                epoch,
            "val_loss":             va_loss,
            "val_r2":               va_r2,
            "phys_mean":            phys_mean,
            "phys_std":             phys_std,
            "ulin_sigma":           ulin_sigma,
            "dp_std":               dp_std,
            "dn_std":               dn_std,
        }, "best_nlsdof_multifidelity.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  "
                  f"(best val MSE = {best_val:.5f} @ epoch {best_epoch})")
            break

print("─" * 72)
print(f"Training  |  wall: {time.perf_counter()-t_train_wall:.1f}s  "
      f"|  CPU: {time.process_time()-t_train_cpu:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Inference — Tukey flat-top taper + sequential hidden-state carryover
# ─────────────────────────────────────────────────────────────────────────────
def make_tukey_taper(n, alpha=0.3):
    """Tukey (tapered cosine) window — flat top preserves DC / drift.
    alpha=0.3 means only 30% of edges are tapered, 70% is flat at 1.0."""
    w = np.ones(n, dtype=np.float32)
    n_taper = int(alpha * n / 2)
    if n_taper > 0:
        taper_up   = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        taper_down = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))[::-1]
        w[:n_taper] = taper_up
        w[-n_taper:] = taper_down
    return w


def infer_overlap_add(sample_indices):
    """
    Reconstruct full T-step non-linear displacement:
      u_nl_pred = u_lin + denorm(GRU_predicted_residual)

    Uses Tukey (flat-top) taper to preserve DC/drift content,
    and sequential processing with hidden state carryover.
    """
    ckpt = torch.load("best_nlsdof_multifidelity.pt", map_location=device, weights_only=False)
    def _clean(sd):
        return {k: v for k, v in sd.items() if not k.startswith("_")}
    film_agg.load_state_dict(_clean(ckpt["film_agg_state"]))
    representation.load_state_dict(_clean(ckpt["representation_state"]))
    pred_head.load_state_dict(_clean(ckpt["pred_head_state"]))
    force_skip.load_state_dict(_clean(ckpt["force_skip_state"]))
    set_mode(False)

    infer_step = L_out // 2          # 50% overlap with flat-top taper
    taper = make_tukey_taper(L_out, alpha=0.3)

    n = len(sample_indices)
    acc = np.zeros((n, T + L_in), dtype=np.float32)
    wgt = np.zeros((n, T + L_in), dtype=np.float32)
    est = np.zeros((n, T + L_in), dtype=np.float32)

    t_local = np.arange(L_all, dtype=np.float32) * dt_data

    with torch.no_grad():
        starts = list(range(0, T, infer_step))

        for start in starts:
            end = start + L_all
            if end > T + L_in:
                break

            batch_xt, batch_xs = [], []
            valid_idx = []

            for si_idx, si in enumerate(sample_indices):
                F_pad     = np.concatenate([np.zeros(L_in, dtype=np.float32), F_norm[si]])
                ulin_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), ulin_norm[si]])
                dapx_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), delta_approx_norm[si]])
                upapx_pad = np.concatenate([np.zeros(L_in, dtype=np.float32), up_approx_norm[si]])
                dpos_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_pos[si]])
                dneg_pad  = np.concatenate([np.zeros(L_in, dtype=np.float32), duct_neg[si]])

                F_win     = F_pad[start:end]
                ulin_win  = ulin_pad[start:end]
                dapx_win  = dapx_pad[start:end]
                upapx_win = upapx_pad[start:end]
                dpos_win  = dpos_pad[start:end]
                dneg_win  = dneg_pad[start:end]
                u_ctx     = est[si_idx, start:start + L_in]
                u_prev    = np.concatenate([u_ctx,
                                            np.zeros(L_out, dtype=np.float32)])

                phase  = 2 * np.pi * fn_arr[si] * t_local
                sin_ph = np.sin(phase).astype(np.float32)
                cos_ph = np.cos(phase).astype(np.float32)

                x_time = np.column_stack([F_win, u_prev, ulin_win,
                                          sin_ph, cos_ph])
                batch_xt.append(x_time)
                batch_xs.append(phys_norm[si][None, :])
                valid_idx.append(si_idx)

            if not batch_xt:
                continue

            xt = torch.from_numpy(np.array(batch_xt, dtype=np.float32)).to(device)
            xs = torch.from_numpy(np.array(batch_xs, dtype=np.float32)).to(device)
            s_flat = xs.squeeze(1)

            F_out   = xt[:, -L_out:, 0]
            x       = film_agg(xt, xs)
            feat, _ = representation(x)
            pred_w  = pred_head(feat) + force_skip(F_out, s_flat)
            out_np  = pred_w.cpu().numpy()

            out_s = start + L_in
            out_e = min(out_s + L_out, T + L_in)
            n_stp = out_e - out_s

            for b, si_idx in enumerate(valid_idx):
                acc[si_idx, out_s:out_e] += out_np[b, :n_stp] * taper[:n_stp]
                wgt[si_idx, out_s:out_e] += taper[:n_stp]
                safe_w = np.where(wgt[si_idx, out_s:out_e] > 0,
                                  wgt[si_idx, out_s:out_e], 1.0)
                raw_est = acc[si_idx, out_s:out_e] / safe_w
                est[si_idx, out_s:out_e] = np.clip(raw_est, -6.0, 6.0)

    safe_wgt = np.where(wgt > 0, wgt, 1.0)
    pred_delta_norm_pad = acc / safe_wgt
    pred_delta_norm = pred_delta_norm_pad[:, L_in:]                # (n, T)

    # Denormalise using u_lin's std (known at inference — no data leakage)
    pred_delta_u = pred_delta_norm * ulin_sigma[sample_indices]

    # ★ Multi-fidelity: final prediction = u_lin + predicted Δu
    pred_u = u_lin_raw[sample_indices] + pred_delta_u
    true_u = u_nl_raw[sample_indices]
    return pred_u, true_u


t_infer_wall = time.perf_counter()
t_infer_cpu  = time.process_time()
idx_val = idx_te        # from here on, all metrics are on the TEST set
pred_u, true_u = infer_overlap_add(idx_val)
print(f"Inference |  wall: {time.perf_counter()-t_infer_wall:.1f}s  "
      f"|  CPU: {time.process_time()-t_infer_cpu:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 10b.  Physics post-filter
# ─────────────────────────────────────────────────────────────────────────────
from scipy.signal import butter, sosfiltfilt

_FS = 1.0 / float(t[1] - t[0])

def physics_lowpass(pred_arr, sample_indices):
    out = pred_arr.copy()
    for i, si in enumerate(sample_indices):
        cutoff = min(max(6.0 * fn_arr[si], 25.0), 0.45 * _FS)
        sos = butter(4, cutoff, fs=_FS, btype="low", output="sos")
        out[i] = sosfiltfilt(sos, out[i]).astype(np.float32)
    return out

# [70/15/15 rerun] low-pass disabled:
# pred_u = physics_lowpass(pred_u, idx_val)

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Metrics in physical units
# ─────────────────────────────────────────────────────────────────────────────
rmse_per = np.sqrt(np.mean((pred_u - true_u) ** 2, axis=1))
peak_t   = np.max(np.abs(true_u), axis=1)
peak_p   = np.max(np.abs(pred_u), axis=1)
rel_err  = np.abs(peak_t - peak_p) / (peak_t + 1e-12) * 100.0
r2_glob  = r2_score(torch.from_numpy(true_u.ravel()),
                    torch.from_numpy(pred_u.ravel()))

# Baseline: u_lin alone
u_lin_val   = u_lin_raw[idx_val]
r2_lin      = r2_score(torch.from_numpy(true_u.ravel()),
                       torch.from_numpy(u_lin_val.ravel()))
rmse_lin    = np.sqrt(np.mean((u_lin_val - true_u) ** 2, axis=1))
peak_lin    = np.max(np.abs(u_lin_val), axis=1)
rel_err_lin = np.abs(peak_t - peak_lin) / (peak_t + 1e-12) * 100.0

print(f"\n{'─'*60}")
print("  Validation — physical units (multi-fidelity v2)")
print(f"{'─'*60}")
print(f"  Best epoch             : {best_epoch}")
print(f"  Val MSE (norm)         : {best_val:.5f}")
print(f"")
print(f"  {'Metric':<28}  {'Linear only':>12}  {'Multi-fidelity':>14}")
print(f"  {'─'*56}")
print(f"  {'Global R²':<28}  {r2_lin:>12.4f}  {r2_glob:>14.4f}")
print(f"  {'RMSE mean [mm]':<28}  {rmse_lin.mean()*1e3:>12.3f}  {rmse_per.mean()*1e3:>14.3f}")
print(f"  {'RMSE std  [mm]':<28}  {rmse_lin.std()*1e3:>12.3f}  {rmse_per.std()*1e3:>14.3f}")
print(f"  {'Peak error mean [%]':<28}  {rel_err_lin.mean():>12.2f}  {rel_err.mean():>14.2f}")
print(f"  {'Peak error median [%]':<28}  {np.median(rel_err_lin):>12.2f}  {np.median(rel_err):>14.2f}")
print(f"  {'Peak error 90th [%]':<28}  {np.percentile(rel_err_lin, 90):>12.2f}  {np.percentile(rel_err, 90):>14.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Training summary plots
# ─────────────────────────────────────────────────────────────────────────────
ep = np.arange(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Multi-Fidelity GRU v2 (L_all={L_all}, L_out={L_out}, {N_EPOCHS} epochs)  Training Summary",
             fontsize=13, fontweight="bold")

axes[0].semilogy(ep, history["train_loss"], label="Train MSE", color="steelblue")
axes[0].semilogy(ep, history["val_loss"],   label="Val MSE",   color="firebrick")
axes[0].axvline(best_epoch, color="k", lw=0.9, ls="--",
                label=f"Best epoch {best_epoch}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (norm. Δu)")
axes[0].set_title("Loss Curves"); axes[0].legend(fontsize=9)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].plot(ep, history["train_r2"], label="Train R²", color="steelblue")
axes[1].plot(ep, history["val_r2"],   label="Val R²",   color="firebrick")
axes[1].axvline(best_epoch, color="k", lw=0.9, ls="--")
axes[1].axhline(1.0, color="gray", ls=":", lw=0.8)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("R²")
axes[1].set_title("R² Score (residual Δu)"); axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

lim = max(peak_t.max(), peak_p.max()) * 1.05
axes[2].scatter(peak_t, peak_p, s=20, alpha=0.65, color="steelblue",
                edgecolors="white", linewidths=0.3, label="Multi-fidelity")
axes[2].scatter(peak_t, peak_lin, s=15, alpha=0.4, color="gray",
                edgecolors="white", linewidths=0.3, label="Linear only")
axes[2].plot([0, lim], [0, lim], "k--", lw=1.0, label="Perfect")
axes[2].set_xlabel("True peak |u|  [m]")
axes[2].set_ylabel("Predicted peak |u|  [m]")
axes[2].set_title("Peak Displacement (Val)")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim)
axes[2].set_aspect("equal")

plt.tight_layout()
# plt.savefig("mf_training_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Qualitative: predicted vs true u(t) for 6 validation samples
# ─────────────────────────────────────────────────────────────────────────────
plot_idx = np.random.default_rng(7).choice(len(idx_val), size=6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
fig.suptitle(
    f"Multi-Fidelity GRU v2 — Predicted vs True (6 Val Samples)",
    fontsize=12, fontweight="bold")

for ax, pi in zip(axes.flatten(), plot_idx):
    r2_i  = r2_score(torch.from_numpy(true_u[pi:pi+1]),
                     torch.from_numpy(pred_u[pi:pi+1]))
    r2_li = r2_score(torch.from_numpy(true_u[pi:pi+1]),
                     torch.from_numpy(u_lin_val[pi:pi+1]))
    ax.plot(t, true_u[pi],    color="steelblue", lw=1.4, label="True (non-linear)")
    ax.plot(t, u_lin_val[pi], color="gray",       lw=0.8, ls=":",  label="Linear baseline", alpha=0.7)
    ax.plot(t, pred_u[pi],    color="firebrick",  lw=1.0, ls="--", label="Multi-fidelity pred")
    ax.axhline(0, color="k", lw=0.4)
    ax.grid(True, alpha=0.3)
    m_i, k_i, c_i = params[idx_val[pi], :3]
    alpha_i, fy_i  = params[idx_val[pi], 3], params[idx_val[pi], 4]
    fn_i = np.sqrt(k_i / m_i) / (2 * np.pi)
    xi_i = c_i / (2 * np.sqrt(k_i * m_i))
    uy_i = fy_i / k_i
    ax.set_title(f"fₙ={fn_i:.2f}Hz  ξ={xi_i:.3f}  α={alpha_i:.3f}  "
                 f"u_y={uy_i:.5f}\nR²(MF)={r2_i:.3f}  R²(lin)={r2_li:.3f}",
                 fontsize=7.5)
    ax.set_ylabel("u  [m]", fontsize=8)

axes[0, 0].legend(fontsize=7)
for ax in axes[1]: ax.set_xlabel("Time [s]", fontsize=9)

plt.tight_layout()
# plt.savefig("mf_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 14.  PSD visualisation
# ─────────────────────────────────────────────────────────────────────────────
from scipy.signal import welch
from scipy.stats import gaussian_kde

print("\nRunning inference on training set for PSD …")
pred_u_tr, true_u_tr = infer_overlap_add(idx_tr)
# [70/15/15 rerun] low-pass disabled:
# pred_u_tr = physics_lowpass(pred_u_tr, idx_tr)

NPERSEG = max(T // 4, 64)

def compute_psds(signals):
    psds = []
    for sig in signals:
        f, p = welch(sig, fs=_FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        psds.append(p)
    return f, np.array(psds)

f_tr,  gt_psds_tr  = compute_psds(true_u_tr)
_,     pr_psds_tr  = compute_psds(pred_u_tr)
f_val, gt_psds_val = compute_psds(true_u)
_,     pr_psds_val = compute_psds(pred_u)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, f, gt_psds, pr_psds, split, n_s in [
    (axes[0], f_tr,  gt_psds_tr,  pr_psds_tr,  "Train",      len(idx_tr)),
    (axes[1], f_val, gt_psds_val, pr_psds_val, "Validation", len(idx_val)),
]:
    for p in gt_psds:
        ax.semilogy(f, p, color="black",    alpha=0.06, linewidth=0.5)
    for p in pr_psds:
        ax.semilogy(f, p, color="firebrick", alpha=0.06, linewidth=0.5)
    ax.semilogy(f, np.median(gt_psds, axis=0), "k-", linewidth=2, label="GT median")
    ax.semilogy(f, np.median(pr_psds, axis=0), "r-", linewidth=2, label="Predicted median")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD  (m²/Hz)")
    ax.set_title(f"{split}  (N={n_s})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

plt.suptitle("PSD — displacement u(t)  |  Multi-Fidelity GRU v2", fontsize=11,
             fontweight="bold")
plt.tight_layout()
# plt.savefig("mf_psd.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Peak displacement: scatter + KDE PDF ──────────────────────────────────
gt_peaks_tr    = np.max(np.abs(true_u_tr), axis=1)
pred_peaks_tr  = np.max(np.abs(pred_u_tr), axis=1)
gt_peaks_val   = np.max(np.abs(true_u),    axis=1)
pred_peaks_val = np.max(np.abs(pred_u),    axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for gt_pk, pr_pk, split, ls in [
    (gt_peaks_tr,  pred_peaks_tr,  "Train",      "-"),
    (gt_peaks_val, pred_peaks_val, "Validation", "--"),
]:
    x_hi    = max(gt_pk.max(), pr_pk.max()) * 1.15
    x_range = np.linspace(0, x_hi, 500)
    ax.plot(x_range, gaussian_kde(gt_pk)(x_range),
            color="black",    ls=ls, linewidth=2, label=f"GT ({split})")
    ax.plot(x_range, gaussian_kde(pr_pk)(x_range),
            color="firebrick", ls=ls, linewidth=2, label=f"Pred ({split})")
ax.set_xlabel("Peak |u|  [m]")
ax.set_ylabel("PDF")
ax.set_title("Peak Displacement — PDF")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
lim = max(gt_peaks_val.max(), pred_peaks_val.max()) * 1.08
ax2.scatter(gt_peaks_val, pred_peaks_val, s=20, alpha=0.65,
            color="steelblue", edgecolors="white", linewidths=0.3)
ax2.plot([0, lim], [0, lim], "k--", lw=1.0, label="Perfect (1:1)")
ax2.set_xlabel("True peak |u|  [m]")
ax2.set_ylabel("Predicted peak |u|  [m]")
ax2.set_title("Peak Displacement Scatter — Validation")
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_aspect("equal")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle("Peak Response  |  Multi-Fidelity GRU v2", fontsize=11,
             fontweight="bold")
plt.tight_layout()
# plt.savefig("mf_peak_pdf.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved: mf_training_summary.png | mf_predictions.png | mf_psd.png | mf_peak_pdf.png | best_nlsdof_multifidelity.pt")
