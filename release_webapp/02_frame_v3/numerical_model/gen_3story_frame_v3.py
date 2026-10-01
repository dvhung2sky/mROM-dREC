"""
Non-linear 3-Story Steel 2D MRF Database Generator — FEM (no shear chain)
==========================================================================
Full 2D moment-resisting frame modelled with Euler-Bernoulli beam-column
elements. Concentrated plastic hinges (bilinear kinematic hardening) at
beam ends and column bases — strong-column-weak-beam (SCWB) design.
Per-story geometry and sections sampled independently (irregular).

Frame topology (fixed for every sample)
───────────────────────────────────────
  2 bays × 3 stories  →  3 columns × 4 elevations  =  12 nodes
  9 column elements + 6 beam elements  =  15 elements
  Base nodes fully fixed  →  27 free DOFs of 36 total.
  DOFs per node: [u_x, u_y, θ_z].

Plastic hinges (15 total)
─────────────────────────
  • both ends of every beam          (12)
  • base of every column (ground)    (3)
  Bilinear kinematic hardening:
      Mp     = Z · Fy
      K_ref  = 4 · E · I / L          (rotational stiffness ref)
      post-yield slope = α · K_ref
  Plastic rotation ψ is an element-internal variable updated by
  radial-return at each Newton iteration.

Material (A992 Grade 50)
────────────────────────
  E  = 200 GPa
  Fy = 345 MPa

Per-story sampled parameters (irregular by construction)
────────────────────────────────────────────────────────
  h_i         story height             [m]
  I_c,i, A_c,i, Z_c,i      column section (I, A, Z_plastic)
  I_b,i, A_b,i, Z_b,i      beam   section (I, A, Z_plastic)
  m_i         seismic floor mass       [kg]
  α_i         post-yield ratio         [–]

Per-structure:
  L_bay       bay width                [m]
  ξ           damping ratio            [–]

Dynamics
────────
  Newmark-β (β=1/4, γ=1/2), modified-Newton iteration with elastic
  tangent K_T = 4/Δt² · M + 2/Δt · C + K_elastic (factored once per
  sample). Rayleigh damping anchored at the first and third natural
  frequencies from the free-DOF generalised eigenvalue problem.

Force
─────
  F(t) applied horizontally at floor-1 (distributed 1/3 per column
  node). Per-sample scaled so worst-hinge demand max(|M|/Mp) across
  the linear run falls within [0.6, 4.0]  →  mix of near-elastic and
  moderately yielded responses.

Output (.npz)
─────────────
  params      (N, 32)  columns in PARAM_COLS
  F           (N, T)   lateral force (total)                  [N]
  u1, u2, u3  (N, T)   mean lateral displacement per floor    [m]
  u1_lin, u2_lin, u3_lin   (N, T)  linear counterpart         [m]
  converged   (N,)  bool
  t           (T,)  [s]
  fs          scalar [Hz]
  param_cols  (P,) str
"""

import numpy as np
import sys
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh, lu_factor, lu_solve

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES = int(__import__("os").environ.get("N_SAMPLES", 1000))
FS        = 100.0
DT        = 1.0 / FS
N_T       = 2000        # v3: 20 s records (was 10 s)
t_arr     = np.arange(N_T) * DT
RNG_SEED  = 2026
rng       = np.random.default_rng(RNG_SEED)

# Material: A992 Grade 50 steel
E_STEEL  = 200.0e9     # Pa
FY_STEEL = 345.0e6     # Pa

# Frame topology (fixed)
N_BAY   = 2
N_COL   = N_BAY + 1        # 3 columns per story
N_STORY = 3
N_LEV   = N_STORY + 1      # 4 elevations incl. ground
N_NODES = N_COL * N_LEV    # 12
DOF_PER = 3                # u_x, u_y, θ_z
N_DOF_T = N_NODES * DOF_PER   # 36

def node_id(i, j):   return j * N_COL + i
def node_dofs(n):    return [3 * n, 3 * n + 1, 3 * n + 2]
def ux_dof(i, j):    return 3 * node_id(i, j) + 0

# Base fully fixed
_FIXED = []
for _i in range(N_COL):
    _FIXED.extend(node_dofs(node_id(_i, 0)))
FIXED_DOFS = np.array(_FIXED, dtype=int)
FREE_DOFS  = np.array(sorted(set(range(N_DOF_T)) - set(_FIXED)), dtype=int)
N_FREE     = len(FREE_DOFS)

# Floor horizontal DOFs (for output averaging + force distribution)
FLOOR_UX = [[ux_dof(i, j) for i in range(N_COL)] for j in range(N_LEV)]
# Floor rotation DOFs (θz at each column node on each level) — for elastic-frame
# rotation output channels (expose bending state the GRU needs to anticipate hinges).
FLOOR_RZ = [[3 * node_id(i, j) + 2 for i in range(N_COL)] for j in range(N_LEV)]

PARAM_COLS = [
    "m1","m2","m3", "h1","h2","h3", "L_bay", "xi",
    "Ic1","Ic2","Ic3", "Ac1","Ac2","Ac3", "Zc1","Zc2","Zc3",
    "Ib1","Ib2","Ib3", "Ab1","Ab2","Ab3", "Zb1","Zb2","Zb3",
    "alpha1","alpha2","alpha3",
    "fn1","fn2","fn3", "target_mu",
]


# ─────────────────────────────────────────────────────────────────────────────
# Force generators (unit peak) — identical flavours to prior scripts
# ─────────────────────────────────────────────────────────────────────────────

def _peak_norm(signal, F_peak):
    mx = np.max(np.abs(signal))
    return signal / mx * F_peak if mx > 1e-12 else signal


def force_multisine(t, rng, F_peak):
    N_comp = 40
    freqs  = rng.uniform(0.3, 10.0, N_comp)
    amps   = rng.exponential(scale=1.0, size=N_comp)
    phases = rng.uniform(0.0, 2 * np.pi, N_comp)
    F = np.sum([a * np.sin(2 * np.pi * f * t + phi)
                for a, f, phi in zip(amps, freqs, phases)], axis=0)
    return _peak_norm(F, F_peak)


def force_chirp_noise(t, rng, F_peak):
    from scipy.signal import chirp as sp_chirp
    f0, f1    = rng.uniform(0.3, 2.0), rng.uniform(5.0, 10.0)
    chirp_sig = sp_chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear')
    white  = rng.standard_normal(len(t))
    freqs  = np.fft.rfftfreq(len(t), d=DT)
    spec   = np.fft.rfft(white)
    f_lo   = rng.uniform(0.3, 1.5)
    f_hi   = rng.uniform(5.0, 10.0)
    spec  *= (freqs >= f_lo) & (freqs <= f_hi)
    noise  = np.fft.irfft(spec, n=len(t))
    w_c    = rng.uniform(0.4, 0.8)
    F = w_c * chirp_sig / (np.std(chirp_sig) + 1e-12) + \
        (1 - w_c) * noise / (np.std(noise) + 1e-12)
    return _peak_norm(F, F_peak)


def force_burst_train(t, rng, F_peak):
    F = np.zeros_like(t)
    for _ in range(int(rng.integers(3, 7))):
        t_c   = rng.uniform(0.2 * t[-1], 0.9 * t[-1])
        sigma = rng.uniform(0.4, 1.5)
        freq  = rng.uniform(0.6, 8.0)
        amp   = rng.uniform(0.3, 1.0)
        env   = np.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        F    += amp * env * np.sin(2 * np.pi * freq * t)
    return _peak_norm(F, F_peak)


_FORCE_GENS = [force_multisine, force_chirp_noise, force_burst_train]


def generate_unit_force(t, rng):
    F = rng.choice(_FORCE_GENS)(t, rng, 1.0).astype(np.float64)
    F -= F.mean()                                  # kill DC (anti-ratcheting)
    ramp = np.clip(t / 0.1, 0.0, 1.0) * np.clip((t[-1] - t) / 0.2, 0.0, 1.0)
    F *= ramp
    return _peak_norm(F, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Euler-Bernoulli 2D beam-column — local stiffness & rotation
# ─────────────────────────────────────────────────────────────────────────────

def K_local_beamcol(E, A, I, L):
    """Local 6×6 stiffness. DOF order: [u1, v1, θ1, u2, v2, θ2]."""
    EA = E * A / L
    EI = E * I
    L2, L3 = L * L, L * L * L
    K = np.zeros((6, 6))
    K[0, 0] =  EA;         K[0, 3] = -EA
    K[3, 0] = -EA;         K[3, 3] =  EA
    K[1, 1] =  12*EI/L3;   K[1, 2] =  6*EI/L2;   K[1, 4] = -12*EI/L3;  K[1, 5] =  6*EI/L2
    K[2, 1] =   6*EI/L2;   K[2, 2] =  4*EI/L;    K[2, 4] =  -6*EI/L2;  K[2, 5] =  2*EI/L
    K[4, 1] = -12*EI/L3;   K[4, 2] = -6*EI/L2;   K[4, 4] =  12*EI/L3;  K[4, 5] = -6*EI/L2
    K[5, 1] =   6*EI/L2;   K[5, 2] =  2*EI/L;    K[5, 4] =  -6*EI/L2;  K[5, 5] =  4*EI/L
    return K


def T_rot(angle):
    """Local-to-global rotation: d_local = T · d_global."""
    c, s = np.cos(angle), np.sin(angle)
    T = np.zeros((6, 6))
    T[0, 0] =  c; T[0, 1] =  s
    T[1, 0] = -s; T[1, 1] =  c
    T[2, 2] = 1
    T[3, 3] =  c; T[3, 4] =  s
    T[4, 3] = -s; T[4, 4] =  c
    T[5, 5] = 1
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Frame builder
# ─────────────────────────────────────────────────────────────────────────────

def build_frame(h, L_bay, Ic, Zc, Ac, Ib, Zb, Ab, alpha):
    """Assemble stacked per-element arrays for the 3-story MRF.
    Columns first (9), then beams (6). Hinges at column bases (j=0) and
    at both ends of every beam — SCWB design.
    """
    Ke_list, T_list, dofs_list = [], [], []
    MpA_list, MpB_list = [], []
    hasA_list, hasB_list = [], []
    Kref_list, alpha_list = [], []

    # ── Columns ──
    for j in range(N_STORY):                 # story index (spans j → j+1)
        L = h[j]
        I, A = Ic[j], Ac[j]
        Ke = K_local_beamcol(E_STEEL, A, I, L)
        Tm = T_rot(np.pi / 2)                # column vertical
        Kref = 4.0 * E_STEEL * I / L
        Mp_c = Zc[j] * FY_STEEL
        for i in range(N_COL):
            n1, n2 = node_id(i, j), node_id(i, j + 1)
            dofs = node_dofs(n1) + node_dofs(n2)
            Ke_list.append(Ke); T_list.append(Tm); dofs_list.append(dofs)
            hasA = (j == 0)                  # plastic hinge only at column base
            hasB = False
            MpA_list.append(Mp_c if hasA else 0.0)
            MpB_list.append(0.0)
            hasA_list.append(hasA); hasB_list.append(hasB)
            Kref_list.append(Kref); alpha_list.append(alpha[j])

    # ── Beams ──
    for j in range(1, N_LEV):                # floor elevation (1..3)
        L = L_bay
        I, A = Ib[j - 1], Ab[j - 1]
        Ke = K_local_beamcol(E_STEEL, A, I, L)
        Tm = T_rot(0.0)                      # beam horizontal
        Kref = 4.0 * E_STEEL * I / L
        Mp_b = Zb[j - 1] * FY_STEEL
        for i in range(N_BAY):
            n1, n2 = node_id(i, j), node_id(i + 1, j)
            dofs = node_dofs(n1) + node_dofs(n2)
            Ke_list.append(Ke); T_list.append(Tm); dofs_list.append(dofs)
            MpA_list.append(Mp_b); MpB_list.append(Mp_b)
            hasA_list.append(True); hasB_list.append(True)
            Kref_list.append(Kref); alpha_list.append(alpha[j - 1])

    return {
        'K_loc': np.stack(Ke_list),
        'T'    : np.stack(T_list),
        'dofs' : np.array(dofs_list, dtype=int),
        'Mp_A' : np.array(MpA_list),
        'Mp_B' : np.array(MpB_list),
        'has_A': np.array(hasA_list),
        'has_B': np.array(hasB_list),
        'K_ref': np.array(Kref_list),
        'alpha': np.array(alpha_list),
    }


def assemble_K(frame):
    """Global elastic stiffness (N_DOF_T × N_DOF_T)."""
    K = np.zeros((N_DOF_T, N_DOF_T))
    KG = np.einsum('eji,ejk,ekl->eil', frame['T'], frame['K_loc'], frame['T'])
    for e in range(frame['K_loc'].shape[0]):
        d = frame['dofs'][e]
        K[np.ix_(d, d)] += KG[e]
    return K


def assemble_M_diag(m_floor, eps_factor=1e-5):
    """Diagonal lumped-mass vector (N_DOF_T,).
    Floor mass is divided equally across the 3 column nodes of the floor
    and placed on both horizontal (u_x) and vertical (u_y) DOFs. All other
    DOFs (rotational, base — though base is constrained out anyway) get a
    small epsilon mass to keep M positive definite.
    """
    M = np.full(N_DOF_T, eps_factor * m_floor.max())
    for j in range(1, N_LEV):
        m_node = m_floor[j - 1] / N_COL
        for i in range(N_COL):
            M[ux_dof(i, j)]              = m_node
            M[3 * node_id(i, j) + 1]     = m_node
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Internal force & radial-return (vectorised over elements)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fint(frame, u_global, psi_A, psi_B):
    """Return assembled internal force f_int (N_DOF_T,) and end moments
    M_A, M_B (each N_E,) given current displacements and plastic rotations.
    """
    dofs  = frame['dofs']
    d_eg  = u_global[dofs]                                      # (N_E, 6)
    d_el  = np.einsum('eij,ej->ei', frame['T'], d_eg)           # local
    d_eff = d_el.copy()
    d_eff[:, 2] -= psi_A
    d_eff[:, 5] -= psi_B
    f_loc = np.einsum('eij,ej->ei', frame['K_loc'], d_eff)      # local force
    M_A   = f_loc[:, 2].copy()
    M_B   = f_loc[:, 5].copy()
    f_eg  = np.einsum('eji,ej->ei', frame['T'], f_loc)          # T.T · f_loc
    f_int = np.zeros(N_DOF_T)
    np.add.at(f_int, dofs.ravel(), f_eg.ravel())
    return f_int, M_A, M_B


def return_map(frame, M_A, M_B, psi_A, psi_B):
    """Bilinear kinematic-hardening radial return for all hinges.
    back moment = α · K_ref · ψ ; yield surface |M − back| ≤ Mp.
    """
    K_ref = frame['K_ref']
    alpha = frame['alpha']
    # End A
    back_A  = alpha * K_ref * psi_A
    trial_A = M_A - back_A
    f_yA    = np.abs(trial_A) - frame['Mp_A']
    act_A   = frame['has_A'] & (f_yA > 0.0)
    dpsi_A  = np.where(act_A, f_yA / ((1.0 + alpha) * K_ref), 0.0)
    psi_A_new = psi_A + np.sign(trial_A) * dpsi_A
    # End B
    back_B  = alpha * K_ref * psi_B
    trial_B = M_B - back_B
    f_yB    = np.abs(trial_B) - frame['Mp_B']
    act_B   = frame['has_B'] & (f_yB > 0.0)
    dpsi_B  = np.where(act_B, f_yB / ((1.0 + alpha) * K_ref), 0.0)
    psi_B_new = psi_B + np.sign(trial_B) * dpsi_B
    return psi_A_new, psi_B_new


# ─────────────────────────────────────────────────────────────────────────────
# Newmark-β average acceleration with modified-Newton iteration
# ─────────────────────────────────────────────────────────────────────────────

def newmark_solve(frame, K_el, M_diag, C_mat, F_global, dt,
                  bilinear=True, max_iter=20, tol=1e-5):
    """Implicit Newmark-β (β=1/4, γ=1/2). Modified-Newton: tangent is
    K_T = K_el + 4/Δt² · M + 2/Δt · C, factored once.

    F_global: (N_T, N_DOF_T) external force per time step (on fixed DOFs
               too; fixed DOF forces are discarded).
    Returns: (u_out (N_T, N_DOF_T) float32, M_A_peak (N_E,), M_B_peak (N_E,))
             or (None, None, None) on divergence.
    """
    N_t  = F_global.shape[0]
    N_E  = frame['K_loc'].shape[0]
    a_c  = 4.0 / (dt * dt)
    b_c  = 2.0 / dt

    K_T_full = K_el + a_c * np.diag(M_diag) + b_c * C_mat
    free = FREE_DOFS
    K_T  = K_T_full[np.ix_(free, free)]
    lu   = lu_factor(K_T)

    u = np.zeros(N_DOF_T)
    v = np.zeros(N_DOF_T)
    a = np.zeros(N_DOF_T)
    psi_A = np.zeros(N_E)
    psi_B = np.zeros(N_E)

    u_out   = np.zeros((N_t, N_DOF_T), dtype=np.float32)
    M_A_peak = np.zeros(N_E)
    M_B_peak = np.zeros(N_E)

    for n in range(1, N_t):
        F_n1 = F_global[n]

        # Newmark history contribution: F_eff = F + M·a_hist + C·v_hist
        a_hist = a_c * u + (4.0 / dt) * v + a
        v_hist = b_c * u + v
        F_eff_full = F_n1 + M_diag * a_hist + C_mat @ v_hist

        # Initial guess for u_{n+1}: extrapolation
        u_new = u + dt * v + 0.5 * dt * dt * a
        psi_A_new = psi_A.copy()
        psi_B_new = psi_B.copy()

        F_eff_free_norm = np.linalg.norm(F_eff_full[free]) + 1e-12

        for k in range(max_iter):
            # Internal force at current u_new, psi (state-determination)
            f_int, M_A, M_B = compute_fint(frame, u_new, psi_A_new, psi_B_new)
            if bilinear:
                psi_A_new, psi_B_new = return_map(frame, M_A, M_B,
                                                   psi_A_new, psi_B_new)
                f_int, M_A, M_B = compute_fint(frame, u_new,
                                                psi_A_new, psi_B_new)

            Kdyn_u = a_c * M_diag * u_new + C_mat @ (b_c * u_new)
            R_full = F_eff_full - Kdyn_u - f_int
            R = R_full[free]
            if np.linalg.norm(R) < tol * F_eff_free_norm:
                break
            du = lu_solve(lu, R)
            u_new[free] += du
        # ── end Newton iteration ──

        # Newmark state update
        a_new = a_c * (u_new - u) - (4.0 / dt) * v - a
        v_new = b_c * (u_new - u) - v
        u, v, a = u_new, v_new, a_new
        psi_A, psi_B = psi_A_new, psi_B_new

        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 5.0:
            return None, None, None

        u_out[n] = u.astype(np.float32)
        np.maximum(M_A_peak, np.abs(M_A), out=M_A_peak)
        np.maximum(M_B_peak, np.abs(M_B), out=M_B_peak)

    return u_out, M_A_peak, M_B_peak


# ─────────────────────────────────────────────────────────────────────────────
# Modal analysis (for Rayleigh damping anchoring)
# ─────────────────────────────────────────────────────────────────────────────

def first_n_frequencies(K_el, M_diag, n=3):
    free = FREE_DOFS
    K_ff = K_el[np.ix_(free, free)]
    M_ff = np.diag(M_diag[free])
    # Generalised eigen-problem K φ = ω² M φ
    w2, _ = eigh(K_ff, M_ff)
    w2 = np.clip(w2, a_min=0.0, a_max=None)
    freqs = np.sqrt(w2) / (2.0 * np.pi)
    return np.sort(freqs)[:n]


def floor_modes(K_el, M_diag, m_floor, n=3):
    """First n modes condensed to floor-mean lateral DOFs, mass-normalised.

    Returns Phi (3 x n) with Phi^T M_floor Phi = I, and omega (n,).
    These are exactly the quantities a modal identification would deliver."""
    free = FREE_DOFS
    w2, V = eigh(K_el[np.ix_(free, free)], np.diag(M_diag[free]))
    w2 = np.clip(w2, 0.0, None)
    order = np.argsort(w2)[:n]
    w = np.sqrt(w2[order])
    full = np.zeros((N_DOF_T, n))
    full[free, :] = V[:, order]
    Phi = np.stack([full[FLOOR_UX[j], :].mean(axis=0) for j in (1, 2, 3)])   # (3, n)
    Mf = np.diag(m_floor)
    gen = np.diag(Phi.T @ Mf @ Phi)
    Phi = Phi / np.sqrt(np.maximum(gen, 1e-30))[None, :]
    return Phi, w


def modal_K(Phi, w, m_floor):
    """Low-fidelity stiffness reconstructed from identified modes:
           K = M Phi Omega^2 Phi^T M
    Exact for the linear floor system when the modal basis is complete;
    it carries no plastic-hinge information, which is the deficiency the
    multi-fidelity corrector has to learn."""
    Mf = np.diag(m_floor)
    return Mf @ Phi @ np.diag(w ** 2) @ Phi.T @ Mf


def newmark_3dof(K, M_f, C, F_floor, dt):
    """Linear 3-DOF Newmark (average acceleration) for the reduced floor model."""
    beta, gam = 0.25, 0.5
    nt = F_floor.shape[0]
    u = np.zeros(3); ud = np.zeros(3); udd = np.zeros(3)
    out = np.zeros((nt, 3), dtype=np.float32)
    Keff = M_f / (beta * dt * dt) + C * gam / (beta * dt) + K
    lu = lu_factor(Keff)
    for k in range(1, nt):
        up = u + dt * ud + dt * dt * (0.5 - beta) * udd
        vp = ud + dt * (1 - gam) * udd
        rhs = (F_floor[k]
               + M_f @ (up / (beta * dt * dt))
               + C @ ((gam / (beta * dt)) * up - vp))
        u_new = lu_solve(lu, rhs)
        udd = (u_new - up) / (beta * dt * dt)
        ud = vp + gam * dt * udd
        u = u_new
        out[k] = u
    return out


def rayleigh_CK(K_el, M_diag, xi, f_anchor_lo, f_anchor_hi):
    """C = a0·M + a1·K with ξ at two anchor frequencies."""
    w_lo = 2.0 * np.pi * f_anchor_lo
    w_hi = 2.0 * np.pi * f_anchor_hi
    if w_lo >= w_hi:
        w_hi = w_lo * 2.0
    a0 = 2.0 * xi * w_lo * w_hi / (w_lo + w_hi)
    a1 = 2.0 * xi / (w_lo + w_hi)
    C = a0 * np.diag(M_diag) + a1 * K_el
    return C, a0, a1


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 74)
print("  Non-linear 3-Story Steel 2D MRF (FEM)  —  Database Generator")
print(f"  fs = {FS} Hz, N_t = {N_T}, T = {(N_T-1)*DT:.2f} s")
print(f"  Topology: {N_BAY} bays × {N_STORY} stories, {N_NODES} nodes, "
      f"{N_FREE} free DOFs, 15 elements, 15 plastic hinges (SCWB)")
print("=" * 74)

params_list, F_list = [], []
u1_nl_list, u2_nl_list, u3_nl_list = [], [], []
u1_mod_list, u2_mod_list, u3_mod_list = [], [], []
Phi_list = []
u1_lin_list, u2_lin_list, u3_lin_list = [], [], []
th1_lin_list, th2_lin_list, th3_lin_list = [], [], []
converged_list = []

n_failed = 0
t_start  = time.time()
attempt  = 0

while len(params_list) < N_SAMPLES:
    attempt += 1

    # ── Sample irregular frame geometry / sections ──
    h     = np.array([rng.uniform(3.8, 4.8),
                      rng.uniform(3.2, 4.0),
                      rng.uniform(3.0, 3.8)])
    L_bay = rng.uniform(5.0, 8.0)

    # Columns: taper (bigger at bottom) with ±15% jitter
    Ic_g  = rng.uniform(3.0e-4, 8.0e-4)
    tI    = rng.uniform(0.55, 0.85)
    Ic    = np.array([Ic_g              * rng.uniform(0.92, 1.12),
                      Ic_g * tI         * rng.uniform(0.85, 1.15),
                      Ic_g * tI * tI    * rng.uniform(0.80, 1.15)])

    Zc_g  = rng.uniform(2.5e-3, 5.5e-3)
    tZ    = rng.uniform(0.55, 0.85)
    Zc    = np.array([Zc_g              * rng.uniform(0.92, 1.12),
                      Zc_g * tZ         * rng.uniform(0.85, 1.15),
                      Zc_g * tZ * tZ    * rng.uniform(0.80, 1.15)])

    Ac    = np.array([rng.uniform(0.015, 0.030),
                      rng.uniform(0.012, 0.025),
                      rng.uniform(0.010, 0.020)])

    # Beams (per floor 1..3)
    Ib    = np.array([rng.uniform(1.5e-4, 5.0e-4),
                      rng.uniform(1.2e-4, 4.0e-4),
                      rng.uniform(0.8e-4, 3.0e-4)])
    Zb    = np.array([rng.uniform(1.5e-3, 4.0e-3),
                      rng.uniform(1.2e-3, 3.2e-3),
                      rng.uniform(0.8e-3, 2.5e-3)])
    Ab    = np.array([rng.uniform(0.010, 0.025),
                      rng.uniform(0.008, 0.020),
                      rng.uniform(0.006, 0.018)])

    # Floor seismic mass (roof typically lighter)
    m_floor = np.array([rng.uniform(40000., 70000.),
                        rng.uniform(40000., 70000.),
                        rng.uniform(25000., 50000.)])

    xi    = rng.uniform(0.02, 0.05)
    alpha = np.array([rng.uniform(0.01, 0.05),
                      rng.uniform(0.01, 0.05),
                      rng.uniform(0.01, 0.05)])

    # ── Build frame & system matrices ──
    frame   = build_frame(h, L_bay, Ic, Zc, Ac, Ib, Zb, Ab, alpha)
    K_el    = assemble_K(frame)
    M_diag  = assemble_M_diag(m_floor)

    # Modal analysis & Rayleigh C
    try:
        freqs = first_n_frequencies(K_el, M_diag, n=3)
        fn1, fn2, fn3 = float(freqs[0]), float(freqs[1]), float(freqs[2])
        if not (0.3 < fn1 < 8.0):           # sanity: sway fundamental in range
            n_failed += 1
            continue
    except Exception:
        n_failed += 1
        continue
    C_mat, a0, a1 = rayleigh_CK(K_el, M_diag, xi, fn1, fn3)

    # ── Force distribution vector (lateral load on floor 1, spread across cols) ──
    # v3: load at EVERY floor, but coherent -- a common history shaped by a random
    # floor pattern, plus a 30% independent component per floor.
    F_common = generate_unit_force(t_arr, rng)
    patt = np.sort(rng.uniform(0.4, 1.0, N_STORY))          # increases with height
    patt = patt / np.abs(patt).max()
    F_unit_floor = np.stack([
        patt[j] * (F_common + 0.30 * generate_unit_force(t_arr, rng))
        for j in range(N_STORY)])
    F_unit_full = np.zeros((N_T, N_DOF_T))
    for j in range(1, N_LEV):
        for i in range(N_COL):
            F_unit_full[:, ux_dof(i, j)] += F_unit_floor[j - 1] / N_COL

    # ── Linear run (unit peak) ──
    u_lin_u, MA_peak_lin, MB_peak_lin = newmark_solve(
        frame, K_el, M_diag, C_mat, F_unit_full, DT, bilinear=False)
    if u_lin_u is None:
        n_failed += 1
        continue

    # Worst-hinge demand under unit force
    Mp_A = frame['Mp_A']; Mp_B = frame['Mp_B']
    has_A = frame['has_A']; has_B = frame['has_B']
    demands = []
    for e in range(len(has_A)):
        if has_A[e]:
            demands.append(MA_peak_lin[e] / Mp_A[e])
        if has_B[e]:
            demands.append(MB_peak_lin[e] / Mp_B[e])
    demand_unit = max(max(demands), 1e-12)

    # Target worst-hinge ductility/demand → force scale
    target_mu = rng.uniform(float(__import__("os").environ.get("MU_LO", 0.6)),
                            float(__import__("os").environ.get("MU_HI", 2.0)))   # v3: 0.6-2.0; MU_HI is the Table 6 knob
    scale     = target_mu / demand_unit

    F_global  = F_unit_full * scale
    F_applied = (F_unit_floor * scale).astype(np.float32)   # (3, T) per-floor force
    u_lin_scaled = u_lin_u * np.float32(scale)

    # ── LOW-FIDELITY ARM: 3-DOF floor model with K = M Phi Omega^2 Phi^T M ──
    Phi_f, w_f = floor_modes(K_el, M_diag, m_floor, n=3)
    K_lf  = modal_K(Phi_f, w_f, m_floor)
    M_f   = np.diag(m_floor)
    w_lo, w_hi = 2*np.pi*fn1, 2*np.pi*fn3
    a0 = 2.0*xi*w_lo*w_hi/(w_lo+w_hi); a1 = 2.0*xi/(w_lo+w_hi)
    C_lf  = a0*M_f + a1*K_lf
    u_mod = newmark_3dof(K_lf, M_f, C_lf, F_applied.T, DT)      # (T, 3)

    # ── Nonlinear run (with plastic hinges) ──
    u_nl, MA_peak_nl, MB_peak_nl = newmark_solve(
        frame, K_el, M_diag, C_mat, F_global, DT, bilinear=True)
    converged = u_nl is not None
    if not converged:
        u_nl = np.full((N_T, N_DOF_T), np.nan, dtype=np.float32)
        n_failed += 1

    # ── Extract floor-mean lateral displacements ──
    def floor_mean(u_arr, j, dof_group=FLOOR_UX):
        cols = dof_group[j]
        return u_arr[:, cols].mean(axis=1)

    u1_nl_i = floor_mean(u_nl,          1)
    u2_nl_i = floor_mean(u_nl,          2)
    u3_nl_i = floor_mean(u_nl,          3)
    u1_mod_i, u2_mod_i, u3_mod_i = u_mod[:, 0], u_mod[:, 1], u_mod[:, 2]
    u1_lin_i = floor_mean(u_lin_scaled, 1)
    u2_lin_i = floor_mean(u_lin_scaled, 2)
    u3_lin_i = floor_mean(u_lin_scaled, 3)
    # Elastic-frame floor-mean rotations — exposes bending state to the GRU.
    th1_lin_i = floor_mean(u_lin_scaled, 1, FLOOR_RZ)
    th2_lin_i = floor_mean(u_lin_scaled, 2, FLOOR_RZ)
    th3_lin_i = floor_mean(u_lin_scaled, 3, FLOOR_RZ)

    params_list.append([
        m_floor[0], m_floor[1], m_floor[2],
        h[0], h[1], h[2], L_bay, xi,
        Ic[0], Ic[1], Ic[2], Ac[0], Ac[1], Ac[2], Zc[0], Zc[1], Zc[2],
        Ib[0], Ib[1], Ib[2], Ab[0], Ab[1], Ab[2], Zb[0], Zb[1], Zb[2],
        alpha[0], alpha[1], alpha[2],
        fn1, fn2, fn3, target_mu,
    ])
    F_list.append(F_applied.astype(np.float32))
    u1_nl_list.append(u1_nl_i.astype(np.float32))
    u2_nl_list.append(u2_nl_i.astype(np.float32))
    u3_nl_list.append(u3_nl_i.astype(np.float32))
    Phi_list.append(Phi_f.astype(np.float32))
    u1_mod_list.append(u1_mod_i.astype(np.float32))
    u2_mod_list.append(u2_mod_i.astype(np.float32))
    u3_mod_list.append(u3_mod_i.astype(np.float32))
    u1_lin_list.append(u1_lin_i.astype(np.float32))
    u2_lin_list.append(u2_lin_i.astype(np.float32))
    u3_lin_list.append(u3_lin_i.astype(np.float32))
    th1_lin_list.append(th1_lin_i.astype(np.float32))
    th2_lin_list.append(th2_lin_i.astype(np.float32))
    th3_lin_list.append(th3_lin_i.astype(np.float32))
    converged_list.append(converged)

    n_done = len(params_list)
    if n_done % 20 == 0 or n_done == N_SAMPLES:
        elapsed   = time.time() - t_start
        rate      = n_done / elapsed if elapsed > 0 else 1
        remaining = (N_SAMPLES - n_done) / rate
        filled    = int(30 * n_done / N_SAMPLES)
        bar       = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {n_done}/{N_SAMPLES}  "
            f"{elapsed:.1f}s  ~{remaining:.0f}s left  "
            f"(failed: {n_failed})"
        )
        sys.stdout.flush()

elapsed_total = time.time() - t_start
n_converged   = sum(converged_list)
print(f"\n\n  Done — {n_converged}/{N_SAMPLES} converged in {elapsed_total:.1f}s "
      f"({n_failed} failures / {attempt} attempts)")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
params_arr    = np.array(params_list,    dtype=np.float32)
F_arr_out     = np.array(F_list,         dtype=np.float32)
Phi_arr       = np.array(Phi_list,       dtype=np.float32)
u1_mod_arr    = np.array(u1_mod_list,    dtype=np.float32)
u2_mod_arr    = np.array(u2_mod_list,    dtype=np.float32)
u3_mod_arr    = np.array(u3_mod_list,    dtype=np.float32)
u1_nl_arr     = np.array(u1_nl_list,     dtype=np.float32)
u2_nl_arr     = np.array(u2_nl_list,     dtype=np.float32)
u3_nl_arr     = np.array(u3_nl_list,     dtype=np.float32)
u1_lin_arr    = np.array(u1_lin_list,    dtype=np.float32)
u2_lin_arr    = np.array(u2_lin_list,    dtype=np.float32)
u3_lin_arr    = np.array(u3_lin_list,    dtype=np.float32)
th1_lin_arr   = np.array(th1_lin_list,   dtype=np.float32)
th2_lin_arr   = np.array(th2_lin_list,   dtype=np.float32)
th3_lin_arr   = np.array(th3_lin_list,   dtype=np.float32)
converged_arr = np.array(converged_list, dtype=bool)

OUT_NPZ = __import__("os").environ.get("OUT_DB", "frame3_v3_database.npz")
np.savez_compressed(
    OUT_NPZ,
    params     = params_arr,
    F          = F_arr_out,
    u1         = u1_nl_arr,
    u2         = u2_nl_arr,
    u3         = u3_nl_arr,
    Phi_floor  = Phi_arr,
    u1_mod     = u1_mod_arr,
    u2_mod     = u2_mod_arr,
    u3_mod     = u3_mod_arr,
    u1_lin     = u1_lin_arr,
    u2_lin     = u2_lin_arr,
    u3_lin     = u3_lin_arr,
    theta1_lin = th1_lin_arr,
    theta2_lin = th2_lin_arr,
    theta3_lin = th3_lin_arr,
    t          = t_arr.astype(np.float32),
    fs         = np.float32(FS),
    converged  = converged_arr,
    param_cols = np.array(PARAM_COLS),
)
print(f"  Saved → {OUT_NPZ}")

with open("nonlinear_3story_2dframe_database_params.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(PARAM_COLS + ["converged"])
    for row, c in zip(params_arr, converged_arr):
        w.writerow(list(row) + [int(c)])
print("  Saved → nonlinear_3story_2dframe_database_params.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
mask = converged_arr
if mask.sum() > 0:
    u1p_nl = np.nanmax(np.abs(u1_nl_arr[mask]), axis=1)
    u2p_nl = np.nanmax(np.abs(u2_nl_arr[mask]), axis=1)
    u3p_nl = np.nanmax(np.abs(u3_nl_arr[mask]), axis=1)
    u1p_l  = np.nanmax(np.abs(u1_lin_arr[mask]), axis=1)
    u2p_l  = np.nanmax(np.abs(u2_lin_arr[mask]), axis=1)
    u3p_l  = np.nanmax(np.abs(u3_lin_arr[mask]), axis=1)
    print("\n" + "=" * 74)
    print("  Database summary  (2D MRF FEM)")
    print("=" * 74)
    print(f"  Samples / converged : {N_SAMPLES} / {n_converged}")
    print(f"  h1 / h2 / h3 range  : "
          f"[{params_arr[:,3].min():.2f}-{params_arr[:,3].max():.2f}] / "
          f"[{params_arr[:,4].min():.2f}-{params_arr[:,4].max():.2f}] / "
          f"[{params_arr[:,5].min():.2f}-{params_arr[:,5].max():.2f}] m")
    print(f"  L_bay               : {params_arr[:,6].min():.2f} – {params_arr[:,6].max():.2f} m")
    print(f"  m1/m2/m3 tonnes     : "
          f"[{params_arr[:,0].min()/1e3:.1f}-{params_arr[:,0].max()/1e3:.1f}] / "
          f"[{params_arr[:,1].min()/1e3:.1f}-{params_arr[:,1].max()/1e3:.1f}] / "
          f"[{params_arr[:,2].min()/1e3:.1f}-{params_arr[:,2].max()/1e3:.1f}]")
    print(f"  fn1/fn2/fn3 (Hz)    : "
          f"[{params_arr[:,29].min():.2f}-{params_arr[:,29].max():.2f}] / "
          f"[{params_arr[:,30].min():.2f}-{params_arr[:,30].max():.2f}] / "
          f"[{params_arr[:,31].min():.2f}-{params_arr[:,31].max():.2f}]")
    print(f"  ξ range             : {params_arr[:,7].min():.3f} – {params_arr[:,7].max():.3f}")
    print(f"  target µ range      : {params_arr[:,32].min():.2f} – {params_arr[:,32].max():.2f}")
    print(f"\n  Converged peak displacements:")
    print(f"  |u1_nl|  [m]        : {u1p_nl.min():.5f} – {u1p_nl.max():.5f}")
    print(f"  |u2_nl|  [m]        : {u2p_nl.min():.5f} – {u2p_nl.max():.5f}")
    print(f"  |u3_nl|  [m]        : {u3p_nl.min():.5f} – {u3p_nl.max():.5f}")
    r1 = u1p_nl / (u1p_l + 1e-12)
    r2 = u2p_nl / (u2p_l + 1e-12)
    r3 = u3p_nl / (u3p_l + 1e-12)
    print(f"  nl/lin DOF-1        : {r1.min():.3f} – {r1.max():.3f}  (med {np.median(r1):.3f})")
    print(f"  nl/lin DOF-2        : {r2.min():.3f} – {r2.max():.3f}  (med {np.median(r2):.3f})")
    print(f"  nl/lin DOF-3        : {r3.min():.3f} – {r3.max():.3f}  (med {np.median(r3):.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# Preview plot
# ─────────────────────────────────────────────────────────────────────────────
converged_idx = np.where(converged_arr)[0]
if len(converged_idx) > 0:
    plot_idx = rng.choice(converged_idx, size=min(6, len(converged_idx)), replace=False)
    plot_idx.sort()

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Non-linear 3-Story Steel 2D MRF (FEM) — 6 Random Samples  [N=1000]",
                 fontsize=13, fontweight='bold')
    gs = GridSpec(6, 5, figure=fig, hspace=0.08, wspace=0.35)

    for row, i in enumerate(plot_idx):
        F_i   = F_arr_out[i].sum(axis=0)   # v2: (3, T) per-floor -> total base shear
        p     = params_arr[i]
        target_mu = p[32]

        ax_f = fig.add_subplot(gs[row, 0])
        ax_f.plot(t_arr, F_i / 1e3, color='steelblue', linewidth=0.7)
        ax_f.axhline(0, color='k', linewidth=0.4)
        ax_f.set_ylabel('F [kN]', fontsize=7)
        ax_f.tick_params(labelsize=6)
        if row == 0: ax_f.set_title('Force F(t)', fontsize=9)
        if row < 5: ax_f.set_xticklabels([])
        ax_f.text(0.98, 0.05, f"µ*={target_mu:.2f}",
                  transform=ax_f.transAxes, fontsize=6, va='bottom', ha='right',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='whitesmoke', alpha=0.9))

        for col, (u_lin_arr, u_nl_arr, colour, dof_label) in enumerate([
            (u1_lin_arr, u1_nl_arr, 'firebrick', 'u1'),
            (u2_lin_arr, u2_nl_arr, 'darkgreen', 'u2'),
            (u3_lin_arr, u3_nl_arr, 'navy',      'u3'),
        ]):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.plot(t_arr, u_lin_arr[i] * 1e3, color='gray', linewidth=0.6, alpha=0.7, label='linear')
            ax.plot(t_arr, u_nl_arr[i]  * 1e3, color=colour, linewidth=0.7, label='non-linear')
            ax.set_ylabel(f'{dof_label} [mm]', fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0:
                fn_val = p[29 + col]
                ax.set_title(f'Floor {col+1} lateral disp.', fontsize=9)
                if col == 0: ax.legend(fontsize=6, loc='upper left')
            if row < 5: ax.set_xticklabels([])
            ax.text(0.98, 0.96, f"fn={p[29+col]:.2f}Hz",
                    transform=ax.transAxes, fontsize=6, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='lightyellow', alpha=0.85))

        ax_rel = fig.add_subplot(gs[row, 4])
        d21_lin = (u2_lin_arr[i] - u1_lin_arr[i]) * 1e3
        d21_nl  = (u2_nl_arr[i]  - u1_nl_arr[i])  * 1e3
        d32_lin = (u3_lin_arr[i] - u2_lin_arr[i]) * 1e3
        d32_nl  = (u3_nl_arr[i]  - u2_nl_arr[i])  * 1e3
        ax_rel.plot(t_arr, d21_lin, color='lightgray', linewidth=0.5, alpha=0.8)
        ax_rel.plot(t_arr, d21_nl,  color='purple',    linewidth=0.7, label='u2-u1')
        ax_rel.plot(t_arr, d32_lin, color='silver',    linewidth=0.5, alpha=0.8)
        ax_rel.plot(t_arr, d32_nl,  color='teal',      linewidth=0.7, label='u3-u2')
        ax_rel.set_ylabel('Δu [mm]', fontsize=7)
        ax_rel.tick_params(labelsize=6)
        if row == 0:
            ax_rel.set_title('Inter-storey drifts', fontsize=9)
            ax_rel.legend(fontsize=6, loc='upper left')
        if row < 5: ax_rel.set_xticklabels([])

    fig.text(0.5, 0.02, 'Time [s]', ha='center', fontsize=10)
    plt.savefig('nonlinear_3story_2dframe_database_preview.png',
                dpi=140, bbox_inches='tight')
    print("\n  Preview figure → nonlinear_3story_2dframe_database_preview.png")
