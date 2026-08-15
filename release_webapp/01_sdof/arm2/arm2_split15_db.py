"""
Arm 2 — purely data-driven baseline.

Same architecture, split, windowing, optimiser and metrics as the
good_multifidelity_*.py scripts. The ONLY differences are the three that
define the arm:

  1. u_lin is never loaded as an input channel,
  2. no EPP / physics-prior / ductility / modal features,
  3. the target is u_nl directly, not the residual du = u_nl - u_lin.

Static input is the raw `params` array of whichever database is used,
standardised -- no hand-built physics features, which is the point of the arm.

Usage:  python3 datadriven_arm2.py [system ...]   (default: all four)
Writes arm2_datadriven.csv
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# identical to the MF scripts
L_out, L_in, L_all, sliding_step = 100, 100, 200, 33
def _hp(n,c,d):
    v=os.environ.get(n)
    return c(v) if v not in (None,"") else d
HIDDEN       = _hp("HPO_HIDDEN", int, 112)
N_GRU_LAYERS = _hp("HPO_LAYERS", int, 1)
BATCH        = _hp("HPO_BATCH", int, 64)
LR           = _hp("HPO_LR", float, 1e-3)
DROPOUT      = _hp("HPO_DROPOUT", float, 0.3)
WD           = _hp("HPO_WD", float, 5e-3)
TUNE_TAG     = os.environ.get("TUNE_TAG", "")

# The MF arm's autoregressive context is anchored by u_lin, which is exact at every
# step. A data-driven model has no such anchor, so feeding back its own output makes
# it drift. FEEDBACK=0 drops that channel entirely (force + params -> response),
# removing the drift mechanism. Both variants are reported.
FEEDBACK = os.environ.get("ARM2_FEEDBACK", "1") == "1"
TAG = "fb" if FEEDBACK else "nofb"

SYSTEMS = {
    "SDOF":                (__import__("os").environ.get("DB","nonlinear_sdof_database_1000.npz"), ["u"], 15),
    "3-DOF":               (__import__("os").environ.get("DB3","nonlinear_3dof_database_1000.npz"), ["u1", "u2", "u3"], 15),
    "3-story steel frame": ("nonlinear_3story_steelframe_database_1000.npz",
                            ["u1", "u2", "u3"], 20),
    "3-story 2D MRF":      ("nonlinear_3story_2dframe_database_1000.npz",
                            ["u1", "u2", "u3"], 20),
}


# ── model: copied verbatim from good_multifidelity_3story_2dframe.py ────────
class FiLMStaticEncoder(nn.Module):
    def __init__(self, n_static, n_time, dropout=0.3):
        super().__init__()
        hidden = max(n_static * 2, 64)
        self.film_mlp = nn.Sequential(
            nn.Linear(n_static, hidden), nn.ReLU(), nn.Linear(hidden, n_time * 2))
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, L_all), nn.Tanh(), nn.Linear(L_all, L_all))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_time, x_static):
        s = x_static.squeeze(1)
        film = self.film_mlp(s).unsqueeze(1)
        g, b = film.chunk(2, dim=-1)
        x_out = (1.0 + g) * x_time + b
        sc = self.dropout(self.static_proj(s)).unsqueeze(2)
        return torch.cat([x_out, sc], dim=2)


class RepresentationLearning(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        if self.training:
            out = F.dropout(out, p=0.3, training=True)
        return out[:, -L_out:, :], h


class PredictionHead(nn.Module):
    def __init__(self, hidden_size, n_dof):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU())
        self.head = nn.Linear(64, n_dof, bias=False)
        nn.init.xavier_uniform_(self.shared[0].weight, gain=0.1)
        nn.init.zeros_(self.head.weight)

    def forward(self, feat):
        return self.head(self.shared(feat))


class WindowDataset(Dataset):
    def __init__(self, X, S, Y):
        self.X, self.S, self.Y = map(torch.from_numpy, (X, S, Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.S[i], self.Y[i]


def make_tukey_taper(n, alpha=0.3):  # identical to the MF scripts
    w = np.ones(n, dtype=np.float32)
    n_t = int(alpha * n / 2)
    if n_t > 0:
        t = 0.5 * (1 - np.cos(np.pi * np.arange(n_t) / n_t))
        w[:n_t] = t; w[-n_t:] = t[::-1]
    return w


def r2_score(y_true, y_pred):  # identical to the MF scripts
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def run(system):
    db, ukeys, n_epochs = SYSTEMS[system]
    n_dof = len(ukeys)
    d = np.load(db)
    good = np.where(d["converged"])[0]
    Fr = d["F"].astype(np.float32)[good]
    U = np.stack([d[k].astype(np.float32)[good] for k in ukeys], axis=-1)  # (N,T,dof)
    P = d["params"].astype(np.float32)[good]
    N, T = Fr.shape

    idx = np.random.default_rng(SEED).permutation(N)
    _ntr, _nva = int(0.70 * N), int(0.15 * N)
    idx_tr  = idx[:_ntr]
    idx_val = idx[_ntr:_ntr + _nva]      # model selection only
    idx_te  = idx[_ntr + _nva:]          # final metrics only
    print(f"[{system}] N={N} 70/15/15 -> train {len(idx_tr)} val {len(idx_val)} test {len(idx_te)}")

    # normalisation from the TRAIN split only -- no u_lin, no physics features
    f_sd = Fr[idx_tr].std() + 1e-12
    u_sd = U[idx_tr].std(axis=(0, 1)) + 1e-12                     # per-DOF scale
    p_mu, p_sd = P[idx_tr].mean(0), P[idx_tr].std(0) + 1e-12
    Fn, Un, Pn = Fr / f_sd, U / u_sd, (P - p_mu) / p_sd

    def make_windows(sample_idx):
        Xs, Ss, Ys = [], [], []
        for si in sample_idx:
            fp = np.concatenate([np.zeros(L_in, np.float32), Fn[si]])
            up = np.concatenate([np.zeros((L_in, n_dof), np.float32), Un[si]])
            for st in range(0, T, sliding_step):
                if st + L_all > len(fp):
                    break
                # autoregressive channel: true past in the context half, zeros ahead
                cols = [fp[st:st + L_all, None]]
                if FEEDBACK:
                    cols.append(np.concatenate([up[st:st + L_in],
                                                np.zeros((L_out, n_dof), np.float32)]))
                Xs.append(np.concatenate(cols, axis=1))
                Ss.append(Pn[si])
                Ys.append(up[st + L_in:st + L_all])
        return (np.asarray(Xs, np.float32), np.asarray(Ss, np.float32)[:, None, :],
                np.asarray(Ys, np.float32))

    X_tr, S_tr, Y_tr = make_windows(idx_tr)
    X_va, S_va, Y_va = make_windows(idx_val)
    print(f"[{system}] windows train/val = {len(X_tr)}/{len(X_va)}")

    n_static, n_time = S_tr.shape[2], X_tr.shape[2]
    film = FiLMStaticEncoder(n_static, n_time, dropout=DROPOUT).to(device)
    rep = RepresentationLearning(n_time + 1, HIDDEN, N_GRU_LAYERS, dropout=DROPOUT).to(device)
    head = PredictionHead(HIDDEN, n_dof).to(device)
    params = list(film.parameters()) + list(rep.parameters()) + list(head.parameters())

    dl_tr = DataLoader(WindowDataset(X_tr, S_tr, Y_tr), batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(WindowDataset(X_va, S_va, Y_va), batch_size=BATCH)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=n_epochs,
                                                steps_per_epoch=len(dl_tr))
    crit = nn.MSELoss()

    def set_mode(train):
        for m in (film, rep, head):
            m.train(train)

    ckpt_path = f"best_arm2_{TAG}{TUNE_TAG}_{system.replace(' ', '_').replace('-', '')}.pt"
    best, best_ep, t0 = np.inf, -1, time.perf_counter()
    for ep in range(1, n_epochs + 1):
        set_mode(True)
        for xb, sb, yb in dl_tr:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(rep(film(xb, sb))[0]), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        set_mode(False)
        tot, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xb, sb, yb in dl_va:
                xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
                p = head(rep(film(xb, sb))[0])
                tot += crit(p, yb).item() * len(xb)
                preds.append(p.cpu()); trues.append(yb.cpu())
        vl = tot / len(dl_va.dataset)
        vr2 = r2_score(torch.cat(trues), torch.cat(preds))
        if vl < best:
            best, best_ep = vl, ep
            torch.save({"epoch": ep, "val_loss": vl, "val_r2": vr2,
                        "film": film.state_dict(), "rep": rep.state_dict(),
                        "head": head.state_dict()}, ckpt_path)
        if ep % 5 == 0 or ep == 1:
            print(f"[{system}] ep {ep:3d}  val_mse {vl:.5f}  val_r2 {vr2:.4f}")
    print(f"[{system}] best ep {best_ep}  ({time.perf_counter()-t0:.0f}s)")

    # ── autoregressive overlap-add rollout ──────────────────────────────────
    # Mirrors infer_overlap_add() in the MF scripts exactly: 50% overlap,
    # Tukey(0.3) taper, and the fed-back context clipped to +-6 as a drift guard.
    # Anything weaker makes Arm 2 lose to Arm 3 on inference, not on information.
    idx_val = idx_te        # all metrics below are on the TEST set
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # best, as MF does
    film.load_state_dict(ck["film"]); rep.load_state_dict(ck["rep"]); head.load_state_dict(ck["head"])
    set_mode(False)
    infer_step = L_out // 2
    taper = make_tukey_taper(L_out, alpha=0.3)
    pred_full = np.zeros((len(idx_val), T, n_dof), np.float32)
    with torch.no_grad():
        for bi in range(0, len(idx_val), 32):
            chunk = idx_val[bi:bi + 32]
            nb = len(chunk)
            fp = np.stack([np.concatenate([np.zeros(L_in, np.float32), Fn[si]])
                           for si in chunk])
            acc = np.zeros((nb, T + L_in, n_dof), np.float32)
            wgt = np.zeros((nb, T + L_in, 1), np.float32)
            est = np.zeros((nb, T + L_in, n_dof), np.float32)
            sb = torch.from_numpy(Pn[chunk][:, None, :]).to(device)
            for st in range(0, T, infer_step):
                if st + L_all > T + L_in:
                    break
                cols = [fp[:, st:st + L_all, None]]
                if FEEDBACK:
                    cols.append(np.concatenate(
                        [est[:, st:st + L_in],
                         np.zeros((nb, L_out, n_dof), np.float32)], axis=1))
                xb = torch.from_numpy(np.concatenate(cols, axis=2)).to(device)
                p = head(rep(film(xb, sb))[0]).cpu().numpy()
                os_, oe = st + L_in, st + L_all
                acc[:, os_:oe] += p * taper[None, :, None]
                wgt[:, os_:oe] += taper[None, :, None]
                sw = np.where(wgt[:, os_:oe] > 0, wgt[:, os_:oe], 1.0)
                est[:, os_:oe] = np.clip(acc[:, os_:oe] / sw, -6.0, 6.0)
            sw = np.where(wgt > 0, wgt, 1.0)
            pred_full[bi:bi + nb] = (acc / sw)[:, L_in:L_in + T]

    pred = pred_full * u_sd
    true = U[idx_val]
    rows = []
    for k in range(n_dof):
        pk_t = np.max(np.abs(true[:, :, k]), axis=1)
        pk_p = np.max(np.abs(pred[:, :, k]), axis=1)
        rel = np.abs(pk_t - pk_p) / (pk_t + 1e-12) * 100.0
        rows.append({
            "system": system, "variant": TAG, "dof": ukeys[k], "best_epoch": best_ep,
            "arm2_R2": round(r2_score(true[:, :, k].ravel(), pred[:, :, k].ravel()), 4),
            "arm2_MAPE_pct": round(float(rel.mean()), 2),
            "arm2_RMSE_mm": round(float(np.sqrt(
                np.mean((pred[:, :, k] - true[:, :, k]) ** 2, axis=1)).mean() * 1e3), 4),
        })
        print(f"  {system} {ukeys[k]}: R2={rows[-1]['arm2_R2']:.4f} "
              f"peak%={rows[-1]['arm2_MAPE_pct']:.2f} RMSE={rows[-1]['arm2_RMSE_mm']:.3f}mm")
    return rows


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SYSTEMS)
    out = []
    for s in wanted:
        out += run(s)
    hdr = list(out[0].keys())
    new = not os.path.exists(f"arm2_datadriven{TUNE_TAG}.csv")
    with open(f"arm2_datadriven{TUNE_TAG}.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if new:
            w.writeheader()
        w.writerows(out)
    print(f"\nwrote arm2_datadriven.csv (+{len(out)} rows)")
