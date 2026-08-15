"""ARM 2 - purely data-driven baseline for the wind turbine: u_red -> u_nl.
Adapted from red_to_nl_tunnel_grf.py; 5 sensors, 17 continuous parameters.
"""

import time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Setup
# ─────────────────────────────────────────────────────────────────────────────
SEED = int(__import__("os").environ.get("SEED", 42))
TAG  = __import__("os").environ.get("TAG", "wt_arm2_hard")
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# Sliding-window config
L_out         = 100
L_in          = 100
L_all         = L_in + L_out
SLIDING_STEP  = int(__import__('os').environ.get('STEP', 50))
USE_FLIP      = __import__('os').environ.get('FLIP', '1') == '1'

# Sample counts (subset of full dataset)
# Split fractions (train, val, test) — uses the whole database
SPLIT = (0.70, 0.15, 0.15)

# Model / training config
HIDDEN        = 192
N_GRU_LAYERS  = 2
BIDIRECTIONAL = True
CTX_NOISE_STD = 0.01
FILM_DROPOUT  = 0.13
GRU_DROPOUT   = 0.15
WEIGHT_DECAY  = 4.0e-4

N_EPOCHS = 15
LR       = 2.0e-3
BATCH    = 32

# Loss weights (start from the 3-story tuned values, retuned for beam topology)
W_PEAK       = 0.15        # peak-magnitude MSE per sensor
W_TAIL       = 0.25        # tail-mean MSE per sensor (residual offset)
W_SMOOTH_Z   = 0.30        # ‖∇_z (pred − target)‖²  spatial-smoothness regulariser
W_SMOOTH_T   = 0.50        # ‖∇_t (pred − target)‖²  temporal-smoothness regulariser
                            # — suppresses broadband HF noise that R² can't see
                            # but PSD/time-domain plots expose as jitter.
W_SPEC       = 0.05        # weighted log-|FFT| loss; weight clamped so quiet
                            # bins push the PSD floor down without abandoning
                            # the loud bins that carry the actual signal.

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load tunnel-beam database
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = __import__("os").environ.get("DB_PATH", "wind_turbine_database.npz")
data = np.load(DB_PATH, allow_pickle=True)
u_red_raw   = data["u_red"]  .astype(np.float64)   # (N, 11, T)  reduced 11-DOF
u_lin_raw   = data["u_lin"]  .astype(np.float64)   # (N, 11, T)  full FE linear
u_nl_raw    = data["u_nl"]   .astype(np.float64)   # (N, 11, T)  full FE plastic (target)
q_raw       = data["q_sensor"].astype(np.float64)
omegas_raw  = data["omegas"] .astype(np.float64)   # (N, 11)
params_raw  = data["params"] .astype(np.float64)   # (N, 10)
param_names = list(map(str, data["param_names"]))
z_sensor    = data["z_sensor"].astype(np.float64)  # (11,)
t_axis      = data["t"]      .astype(np.float64)
dt_data     = float(data["dt"])

# Convention swap: tunnel NPZ stores (N, S, T) — move time axis to the middle.
def _to_NTS(a):  # (N, S, T) → (N, T, S)
    return np.transpose(a, (0, 2, 1))

u_red_raw   = _to_NTS(u_red_raw)        # (N, T, 11)
u_lin_raw   = _to_NTS(u_lin_raw)
u_nl_raw    = _to_NTS(u_nl_raw)
q_raw       = _to_NTS(q_raw)            # (N, T, 11)

# Optional cap on samples used (env N_MAX), applied BEFORE the split so SPLIT fractions
# refer to the capped set.  Truncation keeps the FIRST N_MAX samples, which are the same
# generator seeds as the smaller database, so 100% is a strict subset of 150% and 200% and
# the data-size trend is not confounded by different samples.
_NMAX = int(__import__("os").environ.get("N_MAX", 0))
if _NMAX:
    assert _NMAX <= u_nl_raw.shape[0], f"N_MAX={_NMAX} exceeds database N={u_nl_raw.shape[0]}"
    u_red_raw  = u_red_raw [:_NMAX]
    u_lin_raw  = u_lin_raw [:_NMAX]
    u_nl_raw   = u_nl_raw  [:_NMAX]
    q_raw      = q_raw     [:_NMAX]
    omegas_raw = omegas_raw[:_NMAX]
    params_raw = params_raw[:_NMAX]
    print(f"  N_MAX: using the first {_NMAX} samples")

N, T, S_STA = u_nl_raw.shape
assert S_STA == 5
print(f"Loaded N={N}  T={T}  S_STA={S_STA}  dt={dt_data:.4f}s   z_sensor={z_sensor.tolist()}")
print(f"Param names: {param_names}")

# Reference R²s before training (baseline scores the surrogate must beat)
def _r2(yt, yp):
    ss = ((yt - yp) ** 2).sum();  tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss / (tot + 1e-12))

def _per_sample_metrics(u_true, u_pred):
    """Per-sample R² and peak-relative ε, then aggregated.

    Global flattened R² is dominated by between-sample variance and hides
    per-sample peak errors (a 50%-off peak prints as +1.0000 when one freight
    sample dwarfs ten metro samples).  Per-sample stats are the honest view.
    """
    eps = []
    r2s = []
    for i in range(u_true.shape[0]):
        yt = u_true[i].ravel(); yp = u_pred[i].ravel()
        ss  = ((yt - yp) ** 2).sum()
        tot = ((yt - yt.mean()) ** 2).sum() + 1e-30
        r2s.append(1.0 - ss / tot)
        peak = np.max(np.abs(u_true[i]))
        eps.append(np.max(np.abs(u_true[i] - u_pred[i])) / (peak + 1e-30))
    r2s = np.asarray(r2s); eps = np.asarray(eps)
    return r2s, eps

r2_red_glb = _r2(u_nl_raw, u_red_raw)
r2_lin_glb = _r2(u_nl_raw, u_lin_raw)
r2_red_ps, eps_red = _per_sample_metrics(u_nl_raw, u_red_raw)
r2_lin_ps, eps_lin = _per_sample_metrics(u_nl_raw, u_lin_raw)
print(f"Reference  u_red → u_nl  : global R² = {r2_red_glb:+.6f}   "
      f"per-sample R² mean = {r2_red_ps.mean():+.4f}   "
      f"ε_peak mean = {eps_red.mean()*100:5.2f}%   max = {eps_red.max()*100:5.2f}%")
print(f"Reference  u_lin → u_nl  : global R² = {r2_lin_glb:+.6f}   "
      f"per-sample R² mean = {r2_lin_ps.mean():+.4f}   "
      f"ε_peak mean = {eps_lin.mean()*100:5.2f}%   max = {eps_lin.max()*100:5.2f}%")
print(f"  nonlinear samples (ε_peak(lin|nl) > 1%) : "
      f"{(eps_lin > 0.01).sum():d}/{len(eps_lin):d} "
      f"= {100.0*(eps_lin > 0.01).mean():.1f}%")

# Train / val / test split — 70/15/15.  VAL selects the checkpoint, TEST is reported, so
# the number quoted is never the one that was optimised against.  Both arms build the split
# identically from the same SEED, so they score the SAME test samples.
rng     = np.random.default_rng(SEED)
n_tr    = int(round(SPLIT[0] * N))
n_va    = int(round(SPLIT[1] * N))
assert abs(sum(SPLIT) - 1.0) < 1e-9, f"SPLIT must sum to 1, got {SPLIT}"
assert n_tr + n_va < N, f"SPLIT leaves no test samples out of N={N}"
idx     = rng.permutation(N)
idx_tr  = idx[:n_tr]
idx_val = idx[n_tr:n_tr + n_va]
idx_te  = idx[n_tr + n_va:]
print(f"Samples  —  train: {len(idx_tr)}   val: {len(idx_val)}   test: {len(idx_te)}   "
      f"(SPLIT={SPLIT} of {N} total)")

# First beam mode (used for phase bank + low-pass anchor)
fn1 = (omegas_raw[:, 0] / (2*np.pi)).astype(np.float64)        # (N,)
print(f"  f₁  range = [{fn1.min():.2f}, {fn1.max():.2f}] Hz")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Normalisation  —  per-sensor σ on the train split
# ─────────────────────────────────────────────────────────────────────────────
# Per-sample per-sensor σ for u_red (used to z-score the baseline channels)
sigma_red = u_red_raw.std(axis=1, keepdims=True) + 1e-12       # (N, 1, 11)

# Global per-sensor σ for the residual (train-only) — used for target normalisation
residual_raw = u_nl_raw.copy()          # ARM 2: predict u_nl directly
sigma_target = (residual_raw[idx_tr].std(axis=(0, 1), keepdims=True)
                + 1e-12).astype(np.float64)                    # (1, 1, 11)
print("  σ_target per sensor [mm]:",
      np.round(sigma_target.squeeze() * 1e3, 4).tolist())

target_norm = (residual_raw / sigma_target).astype(np.float32) # (N, T, 11)

# Channels (all (N, T, 11) unless noted)
u_red_norm     = (u_red_raw / sigma_red).astype(np.float32)
delta_lin_norm = ((u_lin_raw   - u_red_raw) / sigma_red).astype(np.float32)

# Live load: per-sample per-sensor z-score
# DC-PRESERVING load normalisation (Arm 2 only needs this, but it is harmless).  The
# per-sample per-sensor z-score that used to live here removed the mean wind speed, hence
# the mean thrust, hence the mean deflection the target is dominated by.  Divide by a global
# train-fitted per-sensor scale and do NOT centre, matching how the target is normalised.
q_sd  = q_raw[idx_tr].std(axis=(0, 1), keepdims=True) + 1e-12   # (1, 1, S)
q_norm = (q_raw / q_sd).astype(np.float32)                      # DC retained
print("  mean(q_norm) per sensor (0.0 would mean the DC was lost):",
      np.round(q_norm.mean(axis=(0, 1)), 3).tolist())

# Phase bank — single freq f₁ per sample
t_local = np.arange(L_all, dtype=np.float32) * dt_data         # (L_all,)
phase_sin = np.stack([np.sin(2*np.pi*float(fn1[i])*t_local) for i in range(N)]).astype(np.float32)
phase_cos = np.stack([np.cos(2*np.pi*float(fn1[i])*t_local) for i in range(N)]).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Static features  (~27-dim)
# ─────────────────────────────────────────────────────────────────────────────
def _log_z(x, ref_idx):
    """log-then-z-score, fitted on `ref_idx` rows."""
    lx = np.log(np.clip(x, 1e-30, None))
    mu = lx[ref_idx].mean(axis=0)
    sd = lx[ref_idx].std (axis=0) + 1e-8
    return ((lx - mu) / sd).astype(np.float32)

# GRF dataset: all 10 params are continuous and positive → log-z, no one-hot.
params_z     = _log_z(params_raw, idx_tr)                      # (N, 10)

# First 6 modal frequencies (Hz)  →  log-z
fn_modes_hz = (omegas_raw[:, :5] / (2*np.pi)).astype(np.float64)
fn_z        = _log_z(fn_modes_hz, idx_tr)                      # (N, 6)

# Peak |q| per sensor (proxy for spatial load amplitude)
qpeak = np.max(np.abs(q_raw), axis=1)                          # (N, 11)
qpeak_z = _log_z(qpeak + 1.0, idx_tr)                          # +1 to keep log positive

static_feat = np.concatenate([params_z, fn_z, qpeak_z], axis=1).astype(np.float32)
N_STATIC  = static_feat.shape[1]                                # 10+6+11 = 27
# Time channels: q + u_prev + sin + cos   (no u_red / u_lin)
N_TIME_CH = S_STA*2 + 2
print(f"Static : {N_STATIC}-dim   Time : {N_TIME_CH} channels   "
      f"(q + u_prev) x S + sin + cos   [ARM 2 - no low-fidelity channels]")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Streaming windowed Dataset  (no big up-front concat → low memory)
# ─────────────────────────────────────────────────────────────────────────────
def build_padded(i, flip=False, u_prev_full=None):
    """Per-sample padded arrays (T+L_in, ·)."""
    s = -1.0 if flip else 1.0
    zero3 = np.zeros((L_in, S_STA), np.float32)
    F_pad      = np.concatenate([zero3,  s * q_norm[i]],           axis=0)
    tgt_pad    = np.concatenate([zero3,  s * target_norm[i]],      axis=0)
    if u_prev_full is None:
        uprev_pad = tgt_pad.copy()
    else:
        uprev_pad = u_prev_full
    return F_pad, uprev_pad, tgt_pad


def window_starts():
    """Window starts that fully fit inside [0, T+L_in − L_all]."""
    starts = list(range(0, T, SLIDING_STEP))
    return [s for s in starts if s + L_all <= T + L_in]

WIN_STARTS = window_starts()
print(f"  window starts per sample: {len(WIN_STARTS)}  "
      f"(L_in={L_in}, L_out={L_out}, step={SLIDING_STEP})")


class TunnelWindowDataset(Dataset):
    """Sliding windows over a fixed list of sample indices, with optional sign-flip
    augmentation.  Builds time tensors on the fly to keep memory low."""
    def __init__(self, sample_indices, flip=False):
        self.indices = list(sample_indices)
        self.flip    = flip
        # (sample_index_in_self.indices, window_start)
        self.pairs = [(k, st) for k in range(len(self.indices)) for st in WIN_STARTS]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        k, start = self.pairs[idx]
        i = self.indices[k]
        (F_pad, uprev_pad, tgt_pad) = build_padded(i, flip=self.flip)
        end = start + L_all
        F_win     = F_pad     [start:end]          # (L_all, 11)
        tgt_win   = tgt_pad   [start:end]          # (L_all, 11)
        u_prev    = np.concatenate([uprev_pad[start:start+L_in],
                                    np.zeros((L_out, S_STA), np.float32)], axis=0)
        sin_full = phase_sin[i][:, None]           # (L_all, 1)
        cos_full = phase_cos[i][:, None]
        x_time = np.concatenate([F_win,
                                 u_prev, sin_full, cos_full], axis=1).astype(np.float32)
        y = tgt_win[L_in:].astype(np.float32)      # (L_out, 11)
        return (torch.from_numpy(x_time),
                torch.from_numpy(static_feat[i][None, :]),
                torch.from_numpy(y))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Model  (FiLM static encoder + GRU + per-sensor head + load skip)
# ─────────────────────────────────────────────────────────────────────────────
class FiLMStaticEncoder(nn.Module):
    def __init__(self, n_static, n_time, dropout=0.10):
        super().__init__()
        hidden = max(n_static * 2, 64)
        self.film_mlp = nn.Sequential(
            nn.Linear(n_static, hidden), nn.ReLU(),
            nn.Linear(hidden, n_time * 2),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, L_all), nn.Tanh(),
            nn.Linear(L_all, L_all),
        )
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x_time, x_static):
        s = x_static.squeeze(1)
        film = self.film_mlp(s).unsqueeze(1)
        gamma, beta = film.chunk(2, dim=-1)
        x_film = (1.0 + gamma) * x_time + beta
        sc = self.dropout(self.static_proj(s)).unsqueeze(2)
        return torch.cat([x_film, sc], dim=2)


class RepresentationLearning(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN, num_layers=N_GRU_LAYERS,
                 dropout=0.15, bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        self.out_size = hidden_size * (2 if bidirectional else 1)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name:      nn.init.zeros_(p)
    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return out[:, -L_out:, :], h


class JointHead(nn.Module):
    """Shared trunk + per-sensor heads.  Last layer zero-init → start at Δ = 0."""
    def __init__(self, hidden_size, n_out=S_STA):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden_size, 96), nn.GELU())
        self.per_sensor = nn.ModuleList([
            nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1, bias=False))
            for _ in range(n_out)
        ])
        nn.init.xavier_uniform_(self.trunk[0].weight, gain=0.1)
        for fh in self.per_sensor:
            nn.init.xavier_uniform_(fh[0].weight, gain=0.1)
            nn.init.zeros_(fh[-1].weight)
    def forward(self, feat):
        t = self.trunk(feat)
        outs = [h(t) for h in self.per_sensor]
        return torch.cat(outs, dim=-1)


class LoadSkip(nn.Module):
    """Per-sensor learned linear skip from local q to local residual.

    For each sensor s, the gain k_s(static) is a function of the static
    features (so e.g. softer soil → bigger skip).  This gives the network
    a direct path from the load to the response at the same z, freeing the
    GRU to focus on the temporal/spatial coupling that is harder to learn.
    """
    def __init__(self, n_static, n_sensor=S_STA):
        super().__init__()
        self.gain = nn.Sequential(nn.Linear(n_static, 64), nn.Tanh(),
                                   nn.Linear(64, n_sensor))
        # init small so the GRU dominates early
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)
    def forward(self, q_out, s_flat):
        """q_out: (B, L_out, 11)  static: (B, N_STATIC) → (B, L_out, 11)"""
        g = self.gain(s_flat).unsqueeze(1)              # (B, 1, 11)
        return g * q_out


def r2_score_t(yt, yp):
    ss = ((yt - yp) ** 2).sum()
    tot = ((yt - yt.mean()) ** 2).sum()
    return (1.0 - ss / (tot + 1e-12)).item()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Inference (overlap-add)
# ─────────────────────────────────────────────────────────────────────────────
def make_tukey_taper(n, alpha=0.3):
    w = np.ones(n, dtype=np.float32)
    nt = int(alpha * n / 2)
    if nt > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
        w[:nt] = ramp;  w[-nt:] = ramp[::-1]
    return w


def infer(film, rep, head, lskip, sample_indices, batch_size=32):
    film.eval(); rep.eval(); head.eval(); lskip.eval()
    infer_step = L_out // 2
    taper = make_tukey_taper(L_out, alpha=0.3)
    n = len(sample_indices)
    acc = np.zeros((n, T + L_in, S_STA), np.float32)
    wgt = np.zeros((n, T + L_in),         np.float32)
    est = np.zeros((n, T + L_in, S_STA), np.float32)

    with torch.no_grad():
        for start in range(0, T, infer_step):
            end = start + L_all
            if end > T + L_in:
                break
            batch_xt, batch_xs = [], []
            for bi, i in enumerate(sample_indices):
                z_pad     = np.zeros((L_in, S_STA), np.float32)
                F_pad     = np.concatenate([z_pad, q_norm          [i]], axis=0)
                u_ctx     = est[bi, start:start + L_in]
                u_prev    = np.concatenate([u_ctx, np.zeros((L_out, S_STA), np.float32)], axis=0)
                x_time = np.concatenate([
                    F_pad     [start:end],
                    u_prev,
                    phase_sin[i][:, None],
                    phase_cos[i][:, None],
                ], axis=1).astype(np.float32)
                batch_xt.append(x_time)
                batch_xs.append(static_feat[i][None, :])
            xt = torch.from_numpy(np.array(batch_xt, np.float32)).to(device)
            xs = torch.from_numpy(np.array(batch_xs, np.float32)).to(device)
            s_flat = xs.squeeze(1)
            q_out  = xt[:, -L_out:, :S_STA]                      # first 11 ch are q
            x = film(xt, xs)
            feat, _ = rep(x)
            pred_w  = head(feat) + lskip(q_out, s_flat)          # (B, L_out, 11)
            out_np = pred_w.cpu().numpy()
            out_s = start + L_in
            out_e = min(out_s + L_out, T + L_in)
            n_stp = out_e - out_s
            for bi in range(n):
                acc[bi, out_s:out_e] += out_np[bi, :n_stp] * taper[:n_stp, None]
                wgt[bi, out_s:out_e] += taper[:n_stp]
                sw = np.where(wgt[bi, out_s:out_e] > 0, wgt[bi, out_s:out_e], 1.0)
                est[bi, out_s:out_e] = np.clip(acc[bi, out_s:out_e] / sw[:, None], -8.0, 8.0)

    sw = np.where(wgt > 0, wgt, 1.0)
    pred_norm  = (acc / sw[:, :, None])[:, L_in:, :]             # (n, T, 11)
    pred_delta = pred_norm * sigma_target.astype(np.float32)     # un-normalise residual
    pred_u     = pred_delta                                      # ARM 2: no baseline added
    return pred_u.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Loss helpers
# ─────────────────────────────────────────────────────────────────────────────
def grad_z(x):
    """First spatial difference along the sensor axis. x: (..., S) → (..., S-1)."""
    return x[..., 1:] - x[..., :-1]


def grad_t(x):
    """First temporal difference along the time axis. x: (B, T, S) → (B, T-1, S)."""
    return x[:, 1:, :] - x[:, :-1, :]


def weighted_log_fft_loss(pred, target, eps_rel=0.05):
    """Per-bin weighted log-magnitude FFT loss along time, with clamped weights.

    Weight is 1 / (|GT_f| + eps_rel · max_f|GT|) per (batch, sensor), so the
    weight ratio between the quietest and loudest bin is bounded by 1/eps_rel
    (≈20× for eps_rel=0.05).  Without that clamp, |GT_f|→0 bins would receive
    infinite weight and the model would learn to match the noise floor while
    abandoning the signal peaks.
    """
    Pf = torch.fft.rfft(pred,   dim=1).abs()
    Tf = torch.fft.rfft(target, dim=1).abs()
    Tmax = Tf.amax(dim=1, keepdim=True)
    floor = eps_rel * Tmax + 1e-12
    w = 1.0 / (Tf + floor)
    w = w / w.mean(dim=1, keepdim=True)            # normalise per (batch, sensor)
    diff = (torch.log(Pf + 1e-10) - torch.log(Tf + 1e-10)) ** 2
    return (w * diff).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Training
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print("\n  building training datasets (original + sign-flipped) …")
    ds_a = TunnelWindowDataset(idx_tr, flip=False)
    ds_tr = (torch.utils.data.ConcatDataset([ds_a, TunnelWindowDataset(idx_tr, flip=True)])
             if USE_FLIP else ds_a)
    ds_va = TunnelWindowDataset(idx_val, flip=False)
    print(f"  train windows={len(ds_tr)}  val windows={len(ds_va)}")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  drop_last=False, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0)

    film  = FiLMStaticEncoder(n_static=N_STATIC, n_time=N_TIME_CH, dropout=FILM_DROPOUT).to(device)
    rep   = RepresentationLearning(input_size=N_TIME_CH + 1, dropout=GRU_DROPOUT).to(device)
    head  = JointHead(hidden_size=rep.out_size).to(device)
    lskip = LoadSkip(n_static=N_STATIC).to(device)
    plist = (list(film.parameters()) + list(rep.parameters())
             + list(head.parameters()) + list(lskip.parameters()))
    n_p = sum(p.numel() for p in plist)
    print(f"  params={n_p:,}  N_STATIC={N_STATIC}  N_TIME_CH={N_TIME_CH}")

    crit = nn.MSELoss()
    opt   = torch.optim.AdamW(plist, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=N_EPOCHS, steps_per_epoch=len(dl_tr),
        pct_start=0.3, anneal_strategy="cos")

    best_val = float("inf")
    best_state = None
    history = {"tr": [], "va": []}

    t0 = time.perf_counter()
    for epoch in range(1, N_EPOCHS + 1):
        film.train(); rep.train(); head.train(); lskip.train()
        total = 0.0; seen = 0
        for xt, xs, yb in dl_tr:
            xt = xt.to(device);  xs = xs.to(device);  yb = yb.to(device)

            # u_prev context channels live at indices [3*S_STA : 4*S_STA] (= [33:44])
            # because we now have 3 baseline channels (q, u_red, Δ_lin)
            # before the u_prev block.
            # channel layout here is q | u_red | Δ_lin | u_prev | sin | cos
            # (the moving-train version has Δ_chain before u_prev, hence 4*S_STA there)
            uprev_lo, uprev_hi = 1*S_STA, 2*S_STA   # ARM 2 layout: q | u_prev | sin | cos
            noise = torch.randn(xt.shape[0], L_in, S_STA, device=device) * CTX_NOISE_STD
            xt = xt.clone()
            xt[:, :L_in, uprev_lo:uprev_hi] = xt[:, :L_in, uprev_lo:uprev_hi] + noise

            s_flat = xs.squeeze(1)
            q_out  = xt[:, -L_out:, :S_STA]
            x      = film(xt, xs)
            feat, _ = rep(x)
            pred = head(feat) + lskip(q_out, s_flat)              # (B, L_out, 11)

            # core MSE (residual space, normalised)
            mse = crit(pred, yb)

            # peak-magnitude loss (per sensor)
            peak_p = pred.abs().amax(dim=1)
            peak_t = yb  .abs().amax(dim=1)
            peak_l = ((peak_p - peak_t) ** 2).mean()

            # tail-mean loss (residual offset / ratchet)
            tail   = L_out // 4
            tail_l = ((pred[:, -tail:].mean(1) - yb[:, -tail:].mean(1)) ** 2).mean()

            # spatial-smoothness regulariser along z
            smooth_l = ((grad_z(pred) - grad_z(yb)) ** 2).mean()

            # temporal-smoothness regulariser — penalises broadband HF jitter
            smooth_t = ((grad_t(pred) - grad_t(yb)) ** 2).mean()

            # per-bin-weighted log-|FFT| — directly attacks PSD mid-band gap
            spec_l = weighted_log_fft_loss(pred, yb)

            loss = (mse + W_PEAK*peak_l + W_TAIL*tail_l
                    + W_SMOOTH_Z*smooth_l + W_SMOOTH_T*smooth_t
                    + W_SPEC*spec_l)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(plist, 1.0)
            opt.step(); sched.step()
            total += mse.item() * xt.shape[0];  seen += xt.shape[0]
        tr_mse = total / seen

        film.eval(); rep.eval(); head.eval(); lskip.eval()
        total = 0.0; seen = 0
        with torch.no_grad():
            for xt, xs, yb in dl_va:
                xt = xt.to(device);  xs = xs.to(device);  yb = yb.to(device)
                s_flat = xs.squeeze(1)
                q_out  = xt[:, -L_out:, :S_STA]
                x = film(xt, xs);  feat, _ = rep(x)
                pred = head(feat) + lskip(q_out, s_flat)
                total += crit(pred, yb).item() * xt.shape[0];  seen += xt.shape[0]
        va_mse = total / seen
        history["tr"].append(tr_mse);  history["va"].append(va_mse)

        if va_mse < best_val:
            best_val = va_mse
            best_state = {
                "film":  {k: v.detach().cpu().clone() for k, v in film .state_dict().items()},
                "rep":   {k: v.detach().cpu().clone() for k, v in rep  .state_dict().items()},
                "head":  {k: v.detach().cpu().clone() for k, v in head .state_dict().items()},
                "lskip": {k: v.detach().cpu().clone() for k, v in lskip.state_dict().items()},
            }
        if epoch % 2 == 0 or epoch == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"    epoch {epoch:>2}  tr_mse={tr_mse:.5f}  va_mse={va_mse:.5f}  lr={lr:.2e}")

    wall = time.perf_counter() - t0
    print(f"  train time: {wall:.1f}s   best val MSE={best_val:.5f}")
    film .load_state_dict(best_state["film"])
    rep  .load_state_dict(best_state["rep"])
    head .load_state_dict(best_state["head"])
    lskip.load_state_dict(best_state["lskip"])
    return film, rep, head, lskip, history, wall


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Run
# ─────────────────────────────────────────────────────────────────────────────
film, rep, head, lskip, history, wall = train()

# Reported metrics come from the held-out TEST split; idx_val only drove checkpoint
# selection.  Rebinding beats renaming the ~15 downstream `*_val` sites.
idx_val_sel  = idx_val
best_val_out = min(history["va"])
idx_val = idx_te

print('saved best_wt_arm2.pt', flush=True)
pred_val = infer(film, rep, head, lskip, idx_val)               # (N_val, T, 11)
u_nl_val  = u_nl_raw [idx_val].astype(np.float32)
u_red_val = u_red_raw[idx_val].astype(np.float32)
u_lin_val = u_lin_raw[idx_val].astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 9b.  Physics low-pass filter  (cutoff set per-sample from the highest mode
#       of interest — same pattern as `good_multifidelity_3story_steelframe.py`).
#       Removes the surrogate's spurious broadband HF floor that the structure
#       cannot physically host.
# ─────────────────────────────────────────────────────────────────────────────
from scipy.signal import butter, sosfiltfilt

_FS = 1.0 / dt_data                       # sampling rate, kept for PSD/Welch
# Highest mode of interest = 6th (we already use the first 6 in static feats).
fn_mod_top = (omegas_raw[:, -1] / (2*np.pi)).astype(np.float64)  # highest stored mode
                                                                 # (5 modes stored, not 6)

# Three-band shelf for the physics low-pass.  Each tier is the fraction of
# original amplitude kept in that band — gives a smooth PSD slope from the
# pass-band through the mid-band into the high-band, instead of a brick-wall
# cliff at the cutoff.
LP_LOW_HZ        = 1.1     # upper edge of the unity-gain pass band
LP_MID_HZ        = 4.0     # upper edge of the gentler mid band
MID_BAND_GAIN    = 0.20    # gain in [LP_LOW_HZ, LP_MID_HZ]   (~−14 dB)
RESIDUAL_HF_GAIN = 0.02    # gain above LP_MID_HZ              (~−34 dB)


def physics_lowpass(pred, sample_indices):
    """Per-sample zero-phase three-band shelf along the time axis.

    Splits the prediction into three bands using two zero-phase Butterworth
    low-passes (at LP_LOW_HZ and LP_MID_HZ) and recombines them with
    band-specific gains:

        out = LP_low(pred)
            + MID_BAND_GAIN    · (LP_mid(pred) − LP_low(pred))
            + RESIDUAL_HF_GAIN · (pred − LP_mid(pred))

    This produces a smoothly-sloping PSD (pass-band → mid-band → HF tail)
    rather than the cliff-then-flat-shelf shape of a single-cutoff LP.
    Cutoffs are upper-bounded at 0.45·fs to stay below Nyquist.
    """
    out = pred.copy()
    a_mid = MID_BAND_GAIN
    a_hi  = RESIDUAL_HF_GAIN
    fc_low_max = min(LP_LOW_HZ, 0.45 * _FS)
    fc_mid_max = min(LP_MID_HZ, 0.45 * _FS)
    for i in range(len(sample_indices)):
        x = out[i]
        sos_low = butter(6, fc_low_max, fs=_FS, btype="low", output="sos")
        sos_mid = butter(6, fc_mid_max, fs=_FS, btype="low", output="sos")
        lp_low = sosfiltfilt(sos_low, x, axis=0)
        lp_mid = sosfiltfilt(sos_mid, x, axis=0)
        mid_band = lp_mid - lp_low
        hi_band  = x      - lp_mid
        out[i] = (lp_low + a_mid * mid_band + a_hi * hi_band).astype(np.float32)
    return out


pred_val = physics_lowpass(pred_val, idx_val)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Train-set inference  (used for the PSD plot below)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning inference on training set (for PSD comparison) …")
pred_tr  = infer(film, rep, head, lskip, idx_tr)               # (N_tr, T, 11)
pred_tr  = physics_lowpass(pred_tr, idx_tr)
u_nl_tr  = u_nl_raw[idx_tr].astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Metrics
# ─────────────────────────────────────────────────────────────────────────────
r2_overall = r2_score_t(torch.from_numpy(u_nl_val.ravel()),
                         torch.from_numpy(pred_val.ravel()))
r2_per_sta = [r2_score_t(torch.from_numpy(u_nl_val[:, :, s].ravel()),
                          torch.from_numpy(pred_val[:, :, s].ravel()))
              for s in range(S_STA)]
rmse = float(np.sqrt(np.mean((pred_val - u_nl_val) ** 2)))
peak_t = np.max(np.abs(u_nl_val), axis=1)
peak_p = np.max(np.abs(pred_val), axis=1)
peak_err = float(np.mean(np.abs(peak_t - peak_p) / (peak_t + 1e-12) * 100.0))

r2_red = r2_score_t(torch.from_numpy(u_nl_val.ravel()),
                     torch.from_numpy(u_red_val.ravel()))
r2_lin = r2_score_t(torch.from_numpy(u_nl_val.ravel()),
                     torch.from_numpy(u_lin_val.ravel()))
r2_red_pf = [r2_score_t(torch.from_numpy(u_nl_val[:, :, s].ravel()),
                         torch.from_numpy(u_red_val[:, :, s].ravel())) for s in range(S_STA)]
r2_lin_pf = [r2_score_t(torch.from_numpy(u_nl_val[:, :, s].ravel()),
                         torch.from_numpy(u_lin_val[:, :, s].ravel())) for s in range(S_STA)]

print(f"\n{'='*84}")
print("  u_red → u_nl  residual surrogate  —  tunnel beam")
print(f"{'='*84}")
print(f"  Reference  u_red  → u_nl  : overall = {r2_red:+.4f}")
print(f"  Reference  u_lin  → u_nl  : overall = {r2_lin:+.4f}")
print(f"  TRAINED                   : overall = {r2_overall:+.4f}")
print(f"  per-sensor R² (trained):  {[f'{v:+.3f}' for v in r2_per_sta]}")
print(f"  RMSE = {rmse*1e3:.3f} mm    PeakErr = {peak_err:.2f} %    train_time = {wall:.1f} s")

print(f"  unweighted mean per-sensor R² : {sum(r2_per_sta)/len(r2_per_sta):+.4f}   (pooled {r2_overall:+.4f})")
torch.save({"film": film.state_dict(), "rep": rep.state_dict(),
            "head": head.state_dict(), "lskip": lskip.state_dict(),
            "SPLIT": SPLIT, "SEED": SEED, "N": N, "DB_PATH": DB_PATH,
            "idx_tr": idx_tr, "idx_val": idx_val_sel, "idx_te": idx_te,
            "HIDDEN": HIDDEN, "L_in": L_in, "L_out": L_out,
            "SLIDING_STEP": SLIDING_STEP, "sigma_target": sigma_target,
            "r2_overall": r2_overall, "r2_per_sensor": r2_per_sta,
            "r2_unweighted": sum(r2_per_sta)/len(r2_per_sta),
            "best_val_mse": best_val_out, "train_time_s": wall},
           f"best_{TAG}.pt")
with open(f"{TAG}_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["label", "r2_overall"] + [f"r2_s{s+1}" for s in range(S_STA)] + ["rmse_m", "peak_err_pct", "train_time_s"]
    w.writerow(hdr)
    w.writerow(["u_red_ref", f"{r2_red:.6f}", *[f"{v:.6f}" for v in r2_red_pf], "-", "-", "0"])
    w.writerow(["u_lin_ref", f"{r2_lin:.6f}", *[f"{v:.6f}" for v in r2_lin_pf], "-", "-", "0"])
    w.writerow(["trained",   f"{r2_overall:.6f}", *[f"{v:.6f}" for v in r2_per_sta],
                f"{rmse:.6e}", f"{peak_err:.3f}", f"{wall:.1f}"])

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Plots
# ─────────────────────────────────────────────────────────────────────────────
# (a) per-sensor R² + training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
xs = np.arange(S_STA)
ax.plot(xs, r2_red_pf, "o--", color="lightgray", label="u_red ref")
ax.plot(xs, r2_lin_pf, "s--", color="gray",      label="u_lin ref")
ax.plot(xs, r2_per_sta,"D-",  color="seagreen",  label="trained")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f"{z:.0f}" for z in z_sensor], fontsize=8)
ax.set_xlabel("z [m]"); ax.set_ylabel("R²")
ax.set_title(f"Per-sensor R² on validation (overall trained = {r2_overall:+.3f})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ep = np.arange(1, len(history["tr"]) + 1)
ax.semilogy(ep, history["tr"], color="seagreen", lw=1.4, label="train")
ax.semilogy(ep, history["va"], color="seagreen", ls="--", lw=1.4, label="val")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (norm. residual)")
ax.set_title("Training loss")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TAG}_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# (b) qualitative time histories — 3 val samples × 3 stations (z = 50, 100, 150 m)
plot_sample = np.random.default_rng(7).choice(len(idx_val), size=3, replace=False)
plot_sta    = [S_STA // 4, S_STA // 2, (3*S_STA) // 4]
fig, axes = plt.subplots(3, 3, figsize=(17, 9), sharex=True)
fig.suptitle("u_red → u_nl  surrogate — 3 val samples × 3 sensors",
             fontsize=12, fontweight="bold")
for row, pi in enumerate(plot_sample):
    sample_i = idx_val[pi]
    for col, s in enumerate(plot_sta):
        ax = axes[row, col]
        ax.plot(t_axis, u_nl_raw [sample_i, :, s]*1e3, "k-",  lw=1.5, label="u_nl (true)")
        ax.plot(t_axis, u_red_raw[sample_i, :, s]*1e3, "0.5", lw=0.8, ls=":", label="u_red")
        ax.plot(t_axis, u_lin_raw[sample_i, :, s]*1e3, "0.7", lw=0.6, ls=":", alpha=0.7, label="u_lin")
        ax.plot(t_axis, pred_val [pi,       :, s]*1e3, "seagreen", lw=1.1, ls="--", label="pred")
        ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3)
        ax.set_title(f"sample {sample_i}  z={z_sensor[s]:.0f} m  f₁={fn1[sample_i]:.2f} Hz", fontsize=8)
        if col == 0: ax.set_ylabel("u [mm]", fontsize=9)
        if row == 0 and col == 2: ax.legend(fontsize=7, loc="upper right")
for ax in axes[-1]: ax.set_xlabel("t [s]", fontsize=9)
plt.tight_layout()
plt.savefig(f"{TAG}_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# (c) PSD per representative sensor
from scipy.signal import welch
NPERSEG = max(T // 4, 64)
def _psd(sig_NT):
    psds = []
    for sig in sig_NT:
        f_, p = welch(sig, fs=_FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        psds.append(p)
    return f_, np.asarray(psds)
fig, axes = plt.subplots(len(plot_sta), 2, figsize=(14, 10))
fig.suptitle("PSD comparison — u_red → u_nl surrogate", fontsize=12, fontweight="bold")
for row, s in enumerate(plot_sta):
    for col, (gt_arr, pr_arr, split, n_s) in enumerate([
        (u_nl_tr [:, :, s], pred_tr [:, :, s], "Train",      len(idx_tr)),
        (u_nl_val[:, :, s], pred_val[:, :, s], "Validation", len(idx_val)),
    ]):
        ax = axes[row, col]
        f_, gt_p = _psd(gt_arr); _, pr_p = _psd(pr_arr)
        for p in gt_p: ax.semilogy(f_, p, color="black",     alpha=0.05, lw=0.5)
        for p in pr_p: ax.semilogy(f_, p, color="firebrick", alpha=0.05, lw=0.5)
        ax.semilogy(f_, np.median(gt_p, axis=0), "k-", lw=2, label="GT median")
        ax.semilogy(f_, np.median(pr_p, axis=0), "r-", lw=2, label="Pred median")
        ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [m²/Hz]")
        ax.set_title(f"z = {z_sensor[s]:.0f} m  —  {split} (N={n_s})")
        ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TAG}_psd.png", dpi=150, bbox_inches="tight")
plt.show()

# (c2) 6 random validation samples: time history + PSD side-by-side
RNG_PICK = np.random.default_rng(SEED + 1)
n_pick   = min(6, len(idx_val))
pick     = RNG_PICK.choice(len(idx_val), size=n_pick, replace=False)
s_pick   = S_STA // 2                                 # mid-span sensor
fig, axes = plt.subplots(n_pick, 2, figsize=(14, 2.4 * n_pick))
fig.suptitle(f"Random validation samples — sensor z = {z_sensor[s_pick]:.0f} m",
             fontsize=12, fontweight="bold")
for row, pi in enumerate(pick):
    si    = idx_val[pi]
    gt_t  = u_nl_val[pi, :, s_pick]
    pr_t  = pred_val[pi, :, s_pick]
    # time
    ax_t = axes[row, 0]
    ax_t.plot(t_axis, gt_t * 1e3, "k-",  lw=1.2, label="GT u_nl")
    ax_t.plot(t_axis, pr_t * 1e3, "r-",  lw=1.0, alpha=0.85, label="Pred")
    ax_t.set_ylabel("u [mm]")
    ax_t.set_title(f"sample idx={si}", fontsize=9)
    if row == 0: ax_t.legend(fontsize=8)
    ax_t.grid(alpha=0.3)
    if row == n_pick - 1: ax_t.set_xlabel("time [s]")
    # PSD
    f_, gp = welch(gt_t, fs=_FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    _,  pp = welch(pr_t, fs=_FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    ax_p = axes[row, 1]
    ax_p.semilogy(f_, gp, "k-", lw=1.2, label="GT")
    ax_p.semilogy(f_, pp, "r-", lw=1.0, alpha=0.85, label="Pred")
    ax_p.set_ylabel("PSD [m²/Hz]")
    if row == 0: ax_p.legend(fontsize=8)
    ax_p.grid(True, which="both", alpha=0.3)
    if row == n_pick - 1: ax_p.set_xlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(f"{TAG}_random_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# (d) Beam-specific: peak |u| profile along z, true vs reduced vs predicted
fig, ax = plt.subplots(figsize=(9, 5))
peak_nl  = np.mean(np.max(np.abs(u_nl_val ), axis=1), axis=0) * 1e3   # mm
peak_red = np.mean(np.max(np.abs(u_red_val), axis=1), axis=0) * 1e3
peak_pr  = np.mean(np.max(np.abs(pred_val ), axis=1), axis=0) * 1e3
ax.plot(z_sensor, peak_nl,  color="k",        marker="D", ls="-",  lw=1.6, label="true  u_nl")
ax.plot(z_sensor, peak_red, color="0.5",      marker="o", ls="--", lw=1.0, label="u_red baseline")
ax.plot(z_sensor, peak_pr,  color="seagreen", marker="s", ls="--", lw=1.4, label="prediction")
ax.set_xlabel("z [m]"); ax.set_ylabel("mean peak |u|  [mm]")
ax.set_title("Spatial peak-displacement profile (val mean)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TAG}_spatial.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: wt_arm2_results.csv  wt_arm2_curves.png  "
      f"{TAG}_samples.png  wt_arm2_psd.png  "
      f"{TAG}_spatial.png")
