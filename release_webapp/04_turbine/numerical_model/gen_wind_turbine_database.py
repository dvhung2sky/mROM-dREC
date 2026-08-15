"""
Wind-turbine benchmark generator — planar fore-aft, per wind_turbine_formulation.docx.

Tapered tubular tower + rigid RNA + embedded monopile on elastoplastic p-y soil,
excited by turbulent wind. Euler-Bernoulli beam FE, lumped base plastic hinge,
Newmark + Newton with velocity-dependent aerodynamic load.

Stores three fidelity levels at n_s = 5 sensor elevations (formulation section 16):
    u_nl   full nonlinear      (soil yielding + base hinge + nonlinear aero)
    u_lin  full FE, elastic    (linearised aero, no yielding)   -> isolates material NL
    u_red  5-mode reduction of the u_lin system (section 13)    -> + reduction error

Usage:  python3 gen_wind_turbine_database.py [N] [--shard k/n] [--out FILE]
"""
import sys
import time

import numpy as np
from scipy.linalg import eigh

G = 9.81
RHO_AIR = 1.225
C_D_TOWER = 0.7
R_ROT = 63.0
A_DISC = np.pi * R_ROT ** 2
V_RATED = 11.4
CT_MAX = 0.8                 # formulation 9.3: ~0.8 below rated
N_TOW, N_PILE = 20, 16       # formulation 11.3
H_T, Z_HUB = 87.6, 90.0
H_G = Z_HUB - H_T            # RNA CoM offset above tower top
E_OVERHANG = 5.0
D_T_TOP, T_T_TOP = 3.87, 0.019
ETA_HINGE = 10.0             # formulation 7
ALPHA_HINGE = 0.03           # hardening ratio of the base hinge (not fixed by the doc)
B_SOIL = 0.20                # formulation 6.3
U_MAX = 10.0                 # divergence bound [m]; clean DB max is 2.40 m
J_RNA_NOM = 2.0e7            # kg m^2 about y through RNA CoM (doc gives only +-15%)
N_MODES_RED = 5              # = n_sensors, formulation 13
DT = 0.02                    # formulation 11.3
T_REC = 60.0                 # 60 s is a hard constraint of this study
T_SPIN = 20.0                # simulated then DISCARDED.  Records used to start from
                             # rest with gravity + full wind applied at t=0; that step
                             # transient decays with tau = 1/(2 pi zeta f1) ~ 60 s, i.e.
                             # as long as the whole record.  Measured effect on the old
                             # 600 s set: permanent shift 2.7x too large at the top
                             # sensor over the first 60 s.  Combined with the
                             # static-equilibrium start below, 20 s is enough.
DECIMATE = 5                 # store at 0.1 s -> Nyquist 5 Hz (covers modes 1-2 at
                             # 0.23-2.07 Hz).  Was 10; at a 60 s record that left only
                             # 3 windows per sample for L_in=L_out=100.

# CALIBRATION. Section 2.1's linear wall taper between the stated endpoints gives a
# 268 t tower, but section 15 states 347 t -- the document is internally inconsistent
# by a factor 1.297. Scaling the wall thickness by that factor reproduces the stated
# tower mass exactly and puts the FIXED-BASE first fore-aft frequency at 0.304 Hz,
# within 5% of section 15's 0.32 Hz. The soil-founded system then sits at 0.267 Hz,
# which is the physically correct reduction for a 32 m monopile.
T_SCALE = 1.297


# ── section properties ──────────────────────────────────────────────────────
def sec_props(D, t):
    """Exact hollow circular: area, second moment, plastic modulus (eqs 4-6)."""
    Di = D - 2 * t
    A = np.pi / 4 * (D ** 2 - Di ** 2)
    I = np.pi / 64 * (D ** 4 - Di ** 4)
    Zp = (D ** 3 - Di ** 3) / 6
    return A, I, Zp


def api_sand_coeffs(phi_deg):
    """API RP2A C1, C2, C3 from friction angle."""
    phi = np.radians(phi_deg)
    a = phi / 2.0
    b = np.radians(45.0) + phi / 2.0
    K0, Ka = 0.4, np.tan(np.radians(45.0) - phi / 2.0) ** 2
    tb, tbf = np.tan(b), np.tan(b - phi)
    C1 = (K0 * np.tan(phi) * np.sin(b) / (tbf * np.cos(a))
          + tb ** 2 * np.tan(a) / tbf + K0 * tb * (np.tan(phi) * np.sin(b) - np.tan(a)))
    C2 = tb / tbf - Ka
    C3 = Ka * (tb ** 8 - 1.0) + K0 * np.tan(phi) * tb ** 4
    return C1, C2, C3


def ct_curve(V):
    """Thrust coefficient: ~constant below rated, feathering above (eq 35 context)."""
    V = np.maximum(np.abs(V), 1e-6)
    return np.where(V <= V_RATED, CT_MAX, CT_MAX * (V_RATED / V) ** 2)


def dct_dV(V):
    V = np.maximum(np.abs(V), 1e-6)
    return np.where(V <= V_RATED, 0.0, -2.0 * CT_MAX * V_RATED ** 2 / V ** 3)


# ── mesh + constant operators ───────────────────────────────────────────────
class Model:
    """Planar FE model. DOFs: [u_0..u_{nu-1}, th_0..th_{nn-1}].
    The mudline node is duplicated (pile side / tower side): the two share one
    translational DOF and have separate rotations joined by the plastic hinge."""

    def __init__(self, p):
        self.p = p
        Lp = p["L_p"]
        zp = np.linspace(-Lp, 0.0, N_PILE + 1)
        zt = np.linspace(0.0, H_T, N_TOW + 1)
        self.z = np.concatenate([zp, zt])            # node 0..N_PILE = pile, then tower
        self.nn = len(self.z)
        self.i_pile_top, self.i_tow_base = N_PILE, N_PILE + 1
        self.i_top = self.nn - 1                     # tower top node

        # translational DOF map: pile-top and tower-base share one u
        umap = np.arange(self.nn)
        umap[self.i_tow_base:] -= 1
        self.umap = umap
        self.nu = self.nn - 1
        self.thmap = self.nu + np.arange(self.nn)
        self.ndof = self.nu + self.nn

        # element connectivity (skip the zero-length pair)
        self.elems = [(i, i + 1) for i in range(self.nn - 1) if i != self.i_pile_top]

        self._geometry()
        self._matrices()
        self._soil()
        self._hinge()

    def _geometry(self):
        p = self.p
        z = self.z
        D = np.where(z >= 0, p["D_b"] + (D_T_TOP - p["D_b"]) * np.clip(z, 0, None) / H_T, p["D_b"])
        tw = np.where(z >= 0, p["t_b"] + (T_T_TOP - p["t_b"]) * np.clip(z, 0, None) / H_T, p["t_b"])
        self.D_node, self.t_node = D, tw * T_SCALE

    def _matrices(self):
        p = self.p
        n = self.ndof
        M = np.zeros((n, n)); K = np.zeros((n, n)); Kg = np.zeros((n, n))
        # axial force from self weight above each element (compression negative)
        self.Le = np.array([self.z[b] - self.z[a] for a, b in self.elems])
        Dm = np.array([0.5 * (self.D_node[a] + self.D_node[b]) for a, b in self.elems])
        tm = np.array([0.5 * (self.t_node[a] + self.t_node[b]) for a, b in self.elems])
        Am, Im, _ = sec_props(Dm, tm)
        self.Am, self.Im = Am, Im
        zmid = np.array([0.5 * (self.z[a] + self.z[b]) for a, b in self.elems])
        wt = p["rho_s"] * Am * self.Le * G
        Nax = np.empty(len(self.elems))
        for e in range(len(self.elems)):
            above = (zmid > zmid[e]) & (zmid > 0)
            Nax[e] = -(p["M_rna"] * G + wt[above].sum()) if zmid[e] >= 0 else \
                     -(p["M_rna"] * G + wt[zmid > 0].sum())
        for e, (a, b) in enumerate(self.elems):
            L, A, I, N = self.Le[e], Am[e], Im[e], Nax[e]
            d = [self.umap[a], self.thmap[a], self.umap[b], self.thmap[b]]
            me = p["rho_s"] * A * L / 420.0 * np.array([
                [156, 22 * L, 54, -13 * L], [22 * L, 4 * L ** 2, 13 * L, -3 * L ** 2],
                [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L ** 2, -22 * L, 4 * L ** 2]])
            ke = p["E"] * I / L ** 3 * np.array([
                [12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
                [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
            kg = N / (30.0 * L) * np.array([
                [36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2],
                [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
            for i in range(4):
                for j in range(4):
                    M[d[i], d[j]] += me[i, j]; K[d[i], d[j]] += ke[i, j]; Kg[d[i], d[j]] += kg[i, j]
        # RNA rigid link at tower top (eq 18)
        iu, ith = self.umap[self.i_top], self.thmap[self.i_top]
        Mr, Jr, hg = p["M_rna"], p["J_rna"], H_G
        M[iu, iu] += Mr; M[iu, ith] += Mr * hg
        M[ith, iu] += Mr * hg; M[ith, ith] += Mr * hg ** 2 + Jr
        self.M, self.K_e, self.K_g = M, K, Kg

    def _soil(self):
        p = self.p
        zp = self.z[:N_PILE + 1]
        dz = np.gradient(zp); dz[0] *= 0.5; dz[-1] *= 0.5
        depth = -zp                                    # positive downwards
        C1, C2, C3 = api_sand_coeffs(p["phi"])
        gam = p["gamma_s"]
        pu = np.minimum((C1 * depth + C2 * p["D_b"]) * gam * depth, C3 * p["D_b"] * gam * depth)
        self.K_soil = np.maximum(p["n_h"] * depth * dz, 1e-6)     # eq 21
        # FY_SOIL_SCALE multiplies the API-sand ultimate resistance: HIGHER = stronger
        # soil = MILDER nonlinearity. Ported verbatim from sens_gen/ so the sweep knob is
        # identical to the one behind the published Table 9.
        self.Fy_soil = np.maximum(pu * dz, 1e-3) * float(
            __import__("os").environ.get("FY_SOIL_SCALE", 1.0))
        self.soil_dofs = self.umap[:N_PILE + 1]

    def _hinge(self):
        p = self.p
        A0, I0, Zp0 = sec_props(p["D_b"], p["t_b"])
        self.M_p = Zp0 * p["fy"]                                   # eq 25
        self.K_h = ETA_HINGE * p["E"] * I0 / self.Le[N_PILE - 1 if N_PILE else 0]
        self.h_dofs = (self.thmap[self.i_pile_top], self.thmap[self.i_tow_base])

    # ---- nonlinear restoring pieces -------------------------------------
    def soil_force(self, u, ypl):
        y = u[self.soil_dofs]
        yy = y - ypl
        yy_y = self.Fy_soil / self.K_soil
        el = np.abs(yy) <= yy_y
        f = np.where(el, self.K_soil * yy,
                     np.sign(yy) * (self.Fy_soil + B_SOIL * self.K_soil * (np.abs(yy) - yy_y)))
        kt = np.where(el, self.K_soil, B_SOIL * self.K_soil)
        return f, kt, el

    def hinge_force(self, u, thpl):
        a, b = self.h_dofs
        dth = u[b] - u[a] - thpl
        dth_y = self.M_p / self.K_h
        if abs(dth) <= dth_y:
            return self.K_h * dth, self.K_h, True
        m = np.sign(dth) * (self.M_p + ALPHA_HINGE * self.K_h * (abs(dth) - dth_y))
        return m, ALPHA_HINGE * self.K_h, False


# ── wind ────────────────────────────────────────────────────────────────────
def synth_wind(p, nt, dt, rng):
    """Kaimal spectrum, random phases, IFFT (eqs 31-33)."""
    Vh = p["V_hub"]
    sig = p["I_ref"] * (0.75 * Vh + 5.6)
    Lu = 8.1 * 42.0
    f = np.fft.rfftfreq(nt, dt); f[0] = 1e-9
    S = 4.0 * sig ** 2 * Lu / Vh / (1.0 + 6.0 * f * Lu / Vh) ** (5.0 / 3.0)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    X = np.sqrt(S) * np.exp(1j * ph); X[0] = 0.0
    v = np.fft.irfft(X, nt)
    v *= sig / (v.std() + 1e-12)                     # enforce IEC sigma exactly
    return Vh + v


def build_loads(mdl, p, nt, dt, rng):
    t = np.arange(nt) * dt
    Vhub_t = synth_wind(p, nt, dt, rng)
    # synth_wind enforces mean=V_hub and sigma=IEC over the FULL simulation, but only
    # the last T_REC is kept -- so re-enforce both on that window, or V_hub and I_ref
    # would mislabel the sample they are attached to (measured: 10% scatter, 32% max,
    # when a window is taken from a longer record without this).
    ns = int(round(T_SPIN / dt))
    v = Vhub_t - p['V_hub']
    v *= (p['I_ref'] * (0.75 * p['V_hub'] + 5.6)) / (v[ns:].std() + 1e-12)
    Vhub_t = p['V_hub'] + v - v[ns:].mean()
    harm = 1.0 + p["a3"] * np.sin(2 * np.pi * p["Omega"] / 60.0 * 3 * t + rng.uniform(0, 2 * np.pi)) \
               + (p["a3"] / 3.0) * np.sin(2 * np.pi * p["Omega"] / 60.0 * t + rng.uniform(0, 2 * np.pi))
    z = mdl.z
    shear = np.where(z > 0, np.clip(z, 1e-3, None) / Z_HUB, 0.0) ** p["alpha_sh"]
    return t, Vhub_t, harm, shear


# ── external force and its velocity Jacobian ────────────────────────────────
def ext_force(mdl, p, ud, Vh_now, harm_now, shear, linear_aero):
    """Returns (f_ext, C_aero). linear_aero=True freezes C_T at the mean point."""
    n = mdl.ndof
    f = np.zeros(n); Ca = np.zeros((n, n))
    iu, ith = mdl.umap[mdl.i_top], mdl.thmap[mdl.i_top]

    # rotor thrust through relative velocity (eqs 34-35)
    Vrel = Vh_now - ud[iu]
    if linear_aero:
        ct, dct = CT_MAX if p["V_hub"] <= V_RATED else CT_MAX * (V_RATED / p["V_hub"]) ** 2, 0.0
    else:
        ct, dct = float(ct_curve(Vrel)), float(dct_dV(Vrel))
    T = 0.5 * RHO_AIR * A_DISC * ct * Vrel * abs(Vrel) * harm_now
    f[iu] += T
    f[ith] += H_G * T
    dTdud = -(RHO_AIR * A_DISC * ct * abs(Vrel)
              + 0.5 * RHO_AIR * A_DISC * dct * Vrel * abs(Vrel)) * harm_now
    Ca[iu, iu] += -dTdud
    Ca[ith, iu] += -H_G * dTdud

    # distributed tower drag (eq 37), consistent nodal loads
    for e, (a, b) in enumerate(mdl.elems):
        if mdl.z[a] < 0:
            continue
        L = mdl.Le[e]
        Dm = 0.5 * (mdl.D_node[a] + mdl.D_node[b])
        sm = 0.5 * (shear[a] + shear[b])
        ua, ub = mdl.umap[a], mdl.umap[b]
        vloc = Vh_now * sm - 0.5 * (ud[ua] + ud[ub])
        w = 0.5 * RHO_AIR * C_D_TOWER * Dm * vloc * abs(vloc)
        dw = -0.5 * RHO_AIR * C_D_TOWER * Dm * 2.0 * abs(vloc) * 0.5
        for (di, c) in ((ua, 0.5 * L), (mdl.thmap[a], L * L / 12.0),
                        (ub, 0.5 * L), (mdl.thmap[b], -L * L / 12.0)):
            f[di] += w * c
            Ca[di, ua] += -dw * c
            Ca[di, ub] += -dw * c
    return f, Ca


def gravity_load(mdl, p):
    g = np.zeros(mdl.ndof)
    g[mdl.thmap[mdl.i_top]] += p["M_rna"] * G * E_OVERHANG      # RNA overturning
    return g


def newmark(mdl, p, t, Vhub_t, harm, shear, nonlinear, dt):
    """Constant-average-acceleration Newmark + Newton (eqs 41-46)."""
    beta, gam = 0.25, 0.5
    n, nt = mdl.ndof, len(t)
    K0 = mdl.K_e + mdl.K_g
    # elastic soil + hinge added to the linear part when running elastic
    Kel = K0.copy()
    for i, d in enumerate(mdl.soil_dofs):
        Kel[d, d] += mdl.K_soil[i]
    a, b = mdl.h_dofs
    Kel[a, a] += mdl.K_h; Kel[b, b] += mdl.K_h
    Kel[a, b] -= mdl.K_h; Kel[b, a] -= mdl.K_h

    # Rayleigh damping from the elastic system (eqs 26-27)
    w2, _ = eigh(Kel, mdl.M, subset_by_index=[0, 1])
    w1, wq = np.sqrt(max(w2[0], 1e-9)), np.sqrt(max(w2[1], 1e-9))
    a0 = 2 * p["xi_s"] * w1 * wq / (w1 + wq); a1 = 2 * p["xi_s"] / (w1 + wq)
    C = a0 * mdl.M + a1 * Kel

    ud = np.zeros(n); udd = np.zeros(n)
    ypl = np.zeros(len(mdl.soil_dofs)); thpl = 0.0
    fg = gravity_load(mdl, p)
    # Start at the static equilibrium under gravity + the initial thrust rather than
    # from rest, so there is no step transient to decay through the record.
    fe0, _ = ext_force(mdl, p, ud, Vhub_t[0], harm[0], shear,
                       linear_aero=not nonlinear)
    u = np.linalg.solve(Kel, fe0 + fg)
    out = np.zeros((nt, n), dtype=np.float32)
    yielded_soil = 0; yielded_hinge = 0
    nonconv = 0; diverged = -1
    out[0] = u                      # was left at zero by the k=1.. loop

    for k in range(1, nt):
        up = u + dt * ud + dt * dt * (0.5 - beta) * udd
        vp = ud + dt * (1 - gam) * udd
        u_k, ud_k = up.copy(), vp.copy()
        udd_k = np.zeros(n)
        ypl_tr, thpl_tr = ypl.copy(), thpl
        conv = False
        for _ in range(12):
            fe, Ca = ext_force(mdl, p, ud_k, Vhub_t[k], harm[k], shear,
                               linear_aero=not nonlinear)
            if nonlinear:
                fs, kts, el_s = mdl.soil_force(u_k, ypl_tr)
                mh, kth, el_h = mdl.hinge_force(u_k, thpl_tr)
                fint = K0 @ u_k
                np.add.at(fint, mdl.soil_dofs, fs)
                fint[a] -= mh; fint[b] += mh
                KT = K0.copy()
                for i, d in enumerate(mdl.soil_dofs):
                    KT[d, d] += kts[i]
                KT[a, a] += kth; KT[b, b] += kth; KT[a, b] -= kth; KT[b, a] -= kth
            else:
                fint = Kel @ u_k
                KT = Kel
            r = fe + fg - mdl.M @ udd_k - C @ ud_k - fint
            Keff = mdl.M / (beta * dt * dt) + (C + Ca) * gam / (beta * dt) + KT
            du = np.linalg.solve(Keff, r)
            u_k += du
            ud_k = vp + gam / (beta * dt) * (u_k - up)
            udd_k = (u_k - up) / (beta * dt * dt)
            if np.linalg.norm(du) <= 1e-8 * max(np.linalg.norm(u_k), 1e-12):
                conv = True
                break
        if not conv:
            nonconv += 1
        if nonlinear:
            fs, kts, el_s = mdl.soil_force(u_k, ypl_tr)
            yy = u_k[mdl.soil_dofs] - ypl_tr
            yy_y = mdl.Fy_soil / mdl.K_soil
            over = np.abs(yy) > yy_y
            ypl[over] = ypl_tr[over] + (1 - B_SOIL) * (np.abs(yy[over]) - yy_y[over]) * np.sign(yy[over])
            yielded_soil += int(over.any())
            dth = u_k[b] - u_k[a] - thpl_tr
            dth_y = mdl.M_p / mdl.K_h
            if abs(dth) > dth_y:
                thpl = thpl_tr + (1 - ALPHA_HINGE) * (abs(dth) - dth_y) * np.sign(dth)
                yielded_hinge += 1
        u, ud, udd = u_k, ud_k, udd_k
        out[k] = u
        if not np.isfinite(u).all() or np.abs(u).max() > U_MAX:
            diverged = k
            break
    return (out, np.abs(ypl).max(), abs(thpl), yielded_soil, yielded_hinge, Kel, C,
            nonconv, diverged)


def modal_reduced(mdl, p, Kel, C, t, Vhub_t, harm, shear, dt, m=N_MODES_RED):
    """Section 13: first m modes of the linearised elastic system, decoupled."""
    w2, Phi = eigh(Kel, mdl.M, subset_by_index=[0, m - 1])
    w = np.sqrt(np.maximum(w2, 1e-9))
    xi = np.diag(Phi.T @ C @ Phi) / (2 * w)
    iu = mdl.umap[mdl.i_top]
    # constant-C_T aerodynamic dashpot at the hub (deliberately mis-set off-design)
    ct0 = CT_MAX if p["V_hub"] <= V_RATED else CT_MAX * (V_RATED / p["V_hub"]) ** 2
    ca = RHO_AIR * A_DISC * ct0 * p["V_hub"]
    xi = xi + (Phi[iu, :] ** 2) * ca / (2 * w)
    fg = gravity_load(mdl, p)
    nt = len(t); q = np.zeros((nt, m)); qd = np.zeros(m); qdd = np.zeros(m)
    # Same static-equilibrium start as the full model.  If u_red began from rest while
    # u_nl began at equilibrium, u_red would carry a spurious transient into the record
    # and the low-fidelity arm would be penalised for an initial-condition mismatch.
    fe0, _ = ext_force(mdl, p, np.zeros(mdl.ndof), Vhub_t[0], harm[0], shear,
                       linear_aero=True)
    q[0] = (Phi.T @ (fe0 + fg)) / w ** 2
    beta, gam = 0.25, 0.5
    for k in range(1, nt):
        fe, _ = ext_force(mdl, p, np.zeros(mdl.ndof), Vhub_t[k], harm[k], shear, linear_aero=True)
        F = Phi.T @ (fe + fg)
        qp = q[k - 1] + dt * qd + dt * dt * (0.5 - beta) * qdd
        vp = qd + dt * (1 - gam) * qdd
        # solving for ACCELERATION, so the denominator is the acceleration form
        # (1 + c*gam*dt + k*beta*dt^2), not the effective-stiffness form.
        keff = 1.0 + 2 * xi * w * gam * dt + w ** 2 * beta * dt * dt
        rhs = F - 2 * xi * w * vp - w ** 2 * qp
        qdd = rhs / keff
        q[k] = qp + beta * dt * dt * qdd
        qd = vp + gam * dt * qdd
    return (Phi @ q.T).T.astype(np.float32)


# ── parameter sampling (formulation section 14) ─────────────────────────────
PARAM_NAMES = ["E", "rho_s", "fy", "D_b", "t_b", "M_rna", "J_rna", "n_h", "phi",
               "gamma_s", "L_p", "xi_s", "V_hub", "I_ref", "alpha_sh", "Omega", "a3"]


def sample_params(rng):
    p = dict(
        E=rng.uniform(200, 215) * 1e9,
        rho_s=rng.uniform(8300, 8700),
        fy=rng.uniform(325, 390) * 1e6,
        D_b=rng.uniform(5.6, 6.4),
        t_b=rng.uniform(0.024, 0.030),
        M_rna=rng.uniform(315, 385) * 1e3,
        J_rna=J_RNA_NOM * rng.uniform(0.85, 1.15),
        n_h=rng.uniform(16, 40) * 1e6,
        phi=rng.uniform(30, 38),
        gamma_s=rng.uniform(9, 11) * 1e3,
        L_p=rng.uniform(28, 36),
        xi_s=rng.uniform(0.005, 0.010),
        V_hub=rng.uniform(6, 24),
        I_ref=rng.uniform(0.12, 0.18),
        alpha_sh=rng.uniform(0.10, 0.20),
        # HARD: down to idling/start-up speeds so 3P sweeps through f1 (=0.231-0.304 Hz)
        Omega=rng.uniform(4.5, 12.1),
        # HARD: stronger 3P thrust harmonic, so the resonance is actually driven
        a3=rng.uniform(0.08, 0.20),
    )
    return p


def nominal_params():
    return dict(E=210e9, rho_s=8500.0, fy=355e6, D_b=6.00, t_b=0.027,
                M_rna=350e3, J_rna=J_RNA_NOM, n_h=25e6, phi=34.0, gamma_s=10e3,
                L_p=32.0, xi_s=0.0075, V_hub=11.4, I_ref=0.14, alpha_sh=0.14,
                Omega=12.1, a3=0.07)


def sensor_nodes(mdl):
    """n_s = 5 sensors spread over the tower (figure 1)."""
    zt = mdl.z[mdl.i_tow_base:]
    targets = np.array([0.15, 0.35, 0.55, 0.75, 1.00]) * H_T
    idx = [mdl.i_tow_base + int(np.argmin(np.abs(zt - zz))) for zz in targets]
    return np.array(idx)


def run_one(seed, dt=DT, T=T_REC):
    rng = np.random.default_rng(seed)
    p = sample_params(rng)
    nt = int(round((T_SPIN + T) / dt))          # spin-up is simulated, then dropped
    mdl = Model(p)
    t, Vh, harm, shear = build_loads(mdl, p, nt, dt, rng)
    (u_nl, ypl_max, thpl, ns_y, nh_y, Kel, C,
     nonconv, diverged) = newmark(mdl, p, t, Vh, harm, shear, True, dt)
    u_li, *_ = newmark(mdl, p, t, Vh, harm, shear, False, dt)
    u_rd = modal_reduced(mdl, p, Kel, C, t, Vh, harm, shear, dt)
    sn = sensor_nodes(mdl)
    sd = mdl.umap[sn]
    d = slice(int(round(T_SPIN / dt)), None, DECIMATE)   # drop the spin-up
    return dict(
        u_nl=u_nl[d][:, sd], u_lin=u_li[d][:, sd], u_red=u_rd[d][:, sd],
        u_hub=u_nl[d][:, mdl.umap[mdl.i_top]],
        V_hub_t=Vh[d].astype(np.float32),
        params=np.array([p[k] for k in PARAM_NAMES], np.float64),
        z_sensor=mdl.z[sn], ypl_max=ypl_max, th_pl=thpl,
        n_soil_yield=ns_y, n_hinge_yield=nh_y,
        nonconv=nonconv, diverged=diverged,
        omegas=np.sqrt(np.maximum(eigh(Kel, mdl.M, subset_by_index=[0, 4])[0], 0)),
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        p = nominal_params(); mdl = Model(p)
        Kel = mdl.K_e + mdl.K_g
        for i, dd in enumerate(mdl.soil_dofs):
            Kel[dd, dd] += mdl.K_soil[i]
        a, b = mdl.h_dofs
        Kel[a, a] += mdl.K_h; Kel[b, b] += mdl.K_h; Kel[a, b] -= mdl.K_h; Kel[b, a] -= mdl.K_h
        w2, _ = eigh(Kel, mdl.M, subset_by_index=[0, 4])
        f = np.sqrt(np.maximum(w2, 0)) / (2 * np.pi)
        Tmax = 0.5 * RHO_AIR * A_DISC * CT_MAX * V_RATED ** 2
        print(f"ndof={mdl.ndof}  nodes={mdl.nn}")
        print(f"f1 = {f[0]:.4f} Hz     (formulation section 15: ~0.32 Hz)")
        print(f"f[1:5] = {np.round(f[1:], 3)}")
        print(f"peak thrust at rated = {Tmax/1e3:.0f} kN   (section 15: ~730 kN)")
        print(f"M_p = {mdl.M_p/1e6:.1f} MN.m   K_h = {mdl.K_h/1e9:.2f} GN.m/rad")
        print(f"soil Fy range = [{mdl.Fy_soil.min()/1e3:.1f}, {mdl.Fy_soil.max()/1e3:.1f}] kN")
        sys.exit(0)

    N = int(args[0]) if args else 200
    shard, nshard = 0, 1
    out = "wind_turbine_database.npz"
    for i, a in enumerate(args):
        if a == "--shard":
            shard, nshard = map(int, args[i + 1].split("/"))
        if a == "--out":
            out = args[i + 1]
    ids = [s for s in range(N) if s % nshard == shard]
    recs, t0 = [], time.perf_counter()
    for j, s in enumerate(ids):
        recs.append(run_one(1000 + s))
        r = recs[-1]
        if r["diverged"] >= 0 or r["nonconv"]:
            print("  seed {}: diverged_at_step={}  nonconv_steps={}".format(
                1000 + s, r["diverged"], r["nonconv"]), flush=True)
        if j % 5 == 0 or j == len(ids) - 1:
            el = time.perf_counter() - t0
            print(f"[shard {shard}] {j+1}/{len(ids)}  {el:.0f}s  "
                  f"({el/(j+1):.1f} s/sample)", flush=True)
    keys_stack = ["u_nl", "u_lin", "u_red", "u_hub", "V_hub_t", "params", "omegas"]
    data = {k: np.stack([r[k] for r in recs]) for k in keys_stack}
    for k in ["ypl_max", "th_pl", "n_soil_yield", "n_hinge_yield",
              "nonconv", "diverged"]:
        data[k] = np.array([r[k] for r in recs])
    data["z_sensor"] = recs[0]["z_sensor"]
    data["param_names"] = np.array(PARAM_NAMES)
    data["seeds"] = np.array([1000 + s for s in ids])
    np.savez_compressed(out, **data)
    print(f"saved {out}  N={len(recs)}  u_nl{data['u_nl'].shape}")
