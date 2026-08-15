"""
Tunnel-beam MOVING-TRAIN dataset.

Per sample, the same train of moving point loads (axles) traverses the beam at
a randomly drawn speed and structural realization.  Three forward models are
evaluated and stored alongside the load history:

    u_lin  : full FE linear elastic (Elastic Winkler springs) — OpenSees
    u_nl   : full FE plastic (Steel01 bilinear, kinematic hardening, b=0.20)
    u_red  : 11-DOF reduced linear MDOF (sensor-tributary load, modal Newmark)
    u_chain: lumped chain-mass MDOF, full mesh, 2 DOFs/node (translation + θ),
             linear soil; sparse Newmark-β.

Per-sample parameters (12, sampled independently per sample):
    E, zeta, t_wall, rho, k_sand, k_clay, z_int, xi, q_cap_sand, q_cap_clay,
    v_kmh, train_type ∈ {0=metro, 1=passenger, 2=freight}

Output npz keys mirror tunnel_full_vs_reduced.npz exactly, plus u_chain.
Run:  python generate_tunnel_moving_train_database.py
"""

import time
import numpy as np
import openseespy.opensees as ops
from scipy.sparse import lil_matrix, csc_matrix, diags
from scipy.sparse.linalg import splu


# ── Geometry / mesh / time / sensors (match tunnel_full_vs_reduced.npz) ─────
L            = 200.0
D            = 6.2

N_EL         = 200
N_NODE       = N_EL + 1
DZ           = L / N_EL
Z_FULL       = np.linspace(0.0, L, N_NODE)
DZ_TRIB      = np.full(N_NODE, DZ);  DZ_TRIB[0] = DZ_TRIB[-1] = DZ / 2.0

T_TOTAL      = 20.0
DT           = 0.02
M_STEP       = int(round(T_TOTAL / DT)) + 1                  # 1001
T_FULL       = np.arange(M_STEP) * DT

S_SENSOR     = 11
Z_SENSOR     = np.linspace(0.0, L, S_SENSOR)
SENSOR_IDX   = np.round(Z_SENSOR / DZ).astype(int)

N_MODES_KEEP   = S_SENSOR
RAYLEIGH_MODES = (1, 6)
N_SAMPLES      = int(__import__("os").environ.get("N_SAMPLES", 1000))

# ── Train catalogue (3 fixed configurations) ───────────────────────────────
TRAIN_TYPES = ['metro', 'passenger', 'freight']
TRAIN_SPECS = {
    'metro':     dict(N_CARS=3, CAR_LENGTH=18.0, BOGIE_OFFSET_A=3.0,
                      BOGIE_OFFSET_B=15.0, AXLE_HALF_GAP=1.10, P_AXLE= 70.0e3),
    'passenger': dict(N_CARS=5, CAR_LENGTH=22.0, BOGIE_OFFSET_A=4.0,
                      BOGIE_OFFSET_B=18.0, AXLE_HALF_GAP=1.25, P_AXLE=130.0e3),
    'freight':   dict(N_CARS=6, CAR_LENGTH=14.0, BOGIE_OFFSET_A=2.5,
                      BOGIE_OFFSET_B=11.5, AXLE_HALF_GAP=0.9,  P_AXLE=200.0e3),
}
RAMP_DIST    = 15.0                                     # m, spatial on/off ramp

# ── Per-sample parameter ranges (mirror reference where overlapping) ───────
E_MEAN, E_COV         = 35.0e9, 0.10
ZETA_LO, ZETA_HI      = 1/10, 1/5
TWALL_MEAN, TWALL_REL = 0.35, 0.10
RHO_MEAN,  RHO_REL    = 2500.0, 0.05
KSAND_MEAN, KSAND_COV = 33.0e6, 0.30
KCLAY_MEAN, KCLAY_COV =  5.0e6, 0.30
ZINT_LO,   ZINT_HI    = 90.0, 110.0
XI_LO,     XI_HI      = 0.01, 0.03
# Capacity ranges chosen so all three train classes can drive springs into the
# post-yield branch on at least the soft-soil samples.  With D·ΔZ = 6.2 m × 1.0 m
# the per-spring yield force is F_y = q_cap · D · ΔZ, giving:
#   sand : F_y ≈ 19 – 93 kN  ⇒ even metro (70 kN/axle) can yield in soft-sand
#   clay : F_y ≈  6 – 31 kN  ⇒ clear yield for all train classes in the clay zone
QCAPS_LO,  QCAPS_HI   =   3.0e3,  15.0e3      # sand bearing capacity [Pa]
QCAPC_LO,  QCAPC_HI   =   1.0e3,   5.0e3      # clay bearing capacity [Pa]
VKMH_LO,   VKMH_HI    = 60.0, 140.0

PARAM_NAMES = ['E', 'zeta', 't_wall', 'rho', 'k_sand', 'k_clay',
               'z_int', 'xi', 'q_cap_sand', 'q_cap_clay',
               'v_kmh', 'train_type']

B_RATIO          = 0.20                               # Steel01 post-yield ratio
MAX_PEAK_U_M     = 0.150                              # safeguard, 150 mm

# A sample counts as "nonlinear" when the linear and plastic FE responses
# diverge by more than this fraction of the peak nonlinear response.
NONLIN_THRESHOLD = 0.01                               # 1 %


# ── Sampling helpers ───────────────────────────────────────────────────────
def lognormal(mean, cov, rng):
    log_var = np.log(1.0 + cov*cov)
    log_mu  = np.log(mean) - 0.5 * log_var
    return float(np.exp(log_mu + np.sqrt(log_var) * rng.standard_normal()))


def sample_params(rng):
    return {
        'E':          lognormal(E_MEAN,     E_COV,     rng),
        'zeta':       float(rng.uniform(ZETA_LO,  ZETA_HI)),
        't_wall':     float(rng.uniform(TWALL_MEAN*(1-TWALL_REL), TWALL_MEAN*(1+TWALL_REL))),
        'rho':        float(rng.uniform(RHO_MEAN  *(1-RHO_REL),   RHO_MEAN  *(1+RHO_REL))),
        'k_sand':     lognormal(KSAND_MEAN, KSAND_COV, rng),
        'k_clay':     lognormal(KCLAY_MEAN, KCLAY_COV, rng),
        'z_int':      float(rng.uniform(ZINT_LO,  ZINT_HI)),
        'xi':         float(rng.uniform(XI_LO,    XI_HI)),
        'q_cap_sand': float(rng.uniform(QCAPS_LO, QCAPS_HI)),
        'q_cap_clay': float(rng.uniform(QCAPC_LO, QCAPC_HI)),
        'v_kmh':      float(rng.uniform(VKMH_LO,  VKMH_HI)),
        'train_type': int(rng.integers(0, 3)),
    }


def section_props(t_wall):
    A = np.pi / 4.0 * (D**2 - (D - 2*t_wall)**2)
    I = np.pi / 64.0 * (D**4 - (D - 2*t_wall)**4)
    return A, I


def _zone_average(p, key_sand, key_clay):
    out = np.zeros(N_NODE)
    z_int = p['z_int']
    for i, z in enumerate(Z_FULL):
        z_lo = z - DZ_TRIB[i] / 2.0
        z_hi = z + DZ_TRIB[i] / 2.0
        if z_hi <= z_int:
            v = p[key_sand]
        elif z_lo >= z_int:
            v = p[key_clay]
        else:
            len_s = z_int - z_lo
            len_c = z_hi - z_int
            v = (p[key_sand]*len_s + p[key_clay]*len_c) / DZ_TRIB[i]
        out[i] = v
    return out


def build_nodal_spring_stiffness(p):
    k_line = _zone_average(p, 'k_sand', 'k_clay')
    return k_line * D * DZ_TRIB


def build_nodal_capacity(p):
    q_cap = _zone_average(p, 'q_cap_sand', 'q_cap_clay')
    return q_cap * D * DZ_TRIB


def sensor_tributaries():
    DZ_S = np.zeros(S_SENSOR)
    sp   = np.diff(Z_SENSOR)
    DZ_S[0]    = sp[0] / 2.0
    DZ_S[-1]   = sp[-1] / 2.0
    DZ_S[1:-1] = (sp[:-1] + sp[1:]) / 2.0
    return DZ_S


# ── Train load ─────────────────────────────────────────────────────────────
def build_axle_offsets(spec):
    raw = []
    for c in range(spec['N_CARS']):
        z_car = c * spec['CAR_LENGTH']
        for bog in (spec['BOGIE_OFFSET_A'], spec['BOGIE_OFFSET_B']):
            zb = z_car + bog
            raw.append(zb - spec['AXLE_HALF_GAP'])
            raw.append(zb + spec['AXLE_HALF_GAP'])
    raw = np.asarray(raw)
    return raw - raw.max()                                  # front axle = 0


def position_envelope(p):
    p   = np.asarray(p, dtype=float)
    env = np.zeros_like(p)
    in_beam = (p >= 0.0) & (p <= L)
    env[in_beam] = 1.0
    rising = in_beam & (p < RAMP_DIST)
    env[rising] = 0.5 * (1.0 - np.cos(np.pi * p[rising] / RAMP_DIST))
    falling = in_beam & (p > L - RAMP_DIST)
    env[falling] = 0.5 * (1.0 - np.cos(np.pi * (L - p[falling]) / RAMP_DIST))
    return env


def build_load_history(spec, v_mps):
    """Nodal vertical force F[i, k] (downward = negative) from the moving
    train.  Each axle is split between its two adjacent nodes by linear
    interpolation, scaled by the spatial on/off ramp envelope."""
    offsets = build_axle_offsets(spec)
    P       = spec['P_AXLE']
    F       = np.zeros((N_NODE, M_STEP))
    for k, t in enumerate(T_FULL):
        positions = v_mps * t + offsets
        envs      = position_envelope(positions)
        for pos, env in zip(positions, envs):
            if env == 0.0:
                continue
            j  = int(np.floor(pos / DZ))
            j  = min(max(j, 0), N_NODE - 2)
            xi = (pos - j * DZ) / DZ
            F[j,     k] += -P * env * (1.0 - xi)
            F[j + 1, k] += -P * env * xi
    return F


# ── OpenSees model build ───────────────────────────────────────────────────
def _build_opensees_model(p, k_node, F_cap=None, kind='elastic'):
    A_p, I_p = section_props(p['t_wall'])
    rhoA_p   = p['rho'] * A_p
    EI_p     = p['zeta'] * p['E'] * I_p

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    for i, z in enumerate(Z_FULL):
        ops.node(i + 1, float(z), 0.0)
    for i in range(N_NODE):
        ops.node(N_NODE + 1 + i, float(Z_FULL[i]), 0.0)
        ops.fix(N_NODE + 1 + i, 1, 1, 1)
    for i in range(N_NODE):
        ops.fix(i + 1, 1, 0, 0)
    for i in range(N_NODE):
        ops.mass(i + 1, 0.0, float(rhoA_p * DZ_TRIB[i]), 0.0)
    ops.geomTransf('Linear', 1)
    for e in range(N_EL):
        ops.element('elasticBeamColumn', e + 1, e + 1, e + 2,
                    float(A_p), float(p['E']), float(p['zeta'] * I_p), 1)
    for i in range(N_NODE):
        mat_tag = 10_000 + i
        K       = float(k_node[i])
        if kind == 'elastic':
            ops.uniaxialMaterial('Elastic', mat_tag, K)
        elif kind == 'plastic':
            F_y = float(F_cap[i])
            ops.uniaxialMaterial('Steel01', mat_tag, F_y, K, B_RATIO)
        else:
            raise ValueError(f'unknown kind={kind!r}')
        ops.element('zeroLength', 20_000 + i,
                    N_NODE + 1 + i, i + 1,
                    '-mat', mat_tag, '-dir', 2)
    return rhoA_p, EI_p


def extract_eigen(p, k_node):
    _build_opensees_model(p, k_node, kind='elastic')
    eigvals = ops.eigen(N_MODES_KEEP)
    omegas  = np.sqrt(np.maximum(np.array(eigvals), 0.0))
    Phi     = np.zeros((N_NODE, N_MODES_KEEP))
    for m in range(N_MODES_KEEP):
        for i in range(N_NODE):
            Phi[i, m] = ops.nodeEigenvector(i + 1, m + 1, 2)
    return omegas, Phi


def run_fe_transient(p, k_node, F_cap, F_load, omegas, kind='elastic'):
    _build_opensees_model(p, k_node, F_cap=F_cap, kind=kind)
    w_a, w_b = omegas[RAYLEIGH_MODES[0]-1], omegas[RAYLEIGH_MODES[1]-1]
    alpha = 2.0 * p['xi'] * w_a * w_b / (w_a + w_b)
    beta_ = 2.0 * p['xi']             / (w_a + w_b)
    ops.rayleigh(alpha, 0.0, beta_, 0.0)

    for i in range(N_NODE):
        ts_tag = 100 + i
        ops.timeSeries('Path', ts_tag, '-dt', DT, '-values', *F_load[i, :].tolist())
        ops.pattern('Plain', ts_tag, ts_tag)
        ops.load(i + 1, 0.0, 1.0, 0.0)

    ops.constraints('Plain'); ops.numberer('RCM'); ops.system('UmfPack')
    ops.test('NormDispIncr', 1.0e-7, 25)
    if kind == 'plastic':
        ops.algorithm('KrylovNewton')
    else:
        ops.algorithm('Linear')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    u_sensor = np.zeros((S_SENSOR, M_STEP))
    for k in range(1, M_STEP):
        ok = ops.analyze(1, DT)
        if ok != 0 and kind == 'plastic':
            ops.algorithm('NewtonLineSearch', '-type', 'Bisection')
            ok = ops.analyze(1, DT)
            ops.algorithm('KrylovNewton')
        if ok != 0:
            for _ in range(4):
                if ops.analyze(1, DT / 4.0) != 0:
                    raise RuntimeError(f'{kind} Newmark failed at step {k}')
        for j, idx in enumerate(SENSOR_IDX):
            u_sensor[j, k] = ops.nodeDisp(int(idx) + 1, 2)
    return u_sensor


# ── Lumped chain-mass MDOF (linear soil) ───────────────────────────────────
def assemble_chain_matrices(k_node, EI_p, rhoA_p):
    nd = 2 * N_NODE
    K  = lil_matrix((nd, nd))
    Le = DZ
    Ke = (EI_p / Le**3) * np.array([
        [ 12.0,    6.0*Le,  -12.0,    6.0*Le   ],
        [ 6.0*Le,  4.0*Le**2, -6.0*Le, 2.0*Le**2],
        [-12.0,   -6.0*Le,   12.0,   -6.0*Le   ],
        [ 6.0*Le,  2.0*Le**2, -6.0*Le, 4.0*Le**2],
    ])
    for e in range(N_EL):
        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += Ke[a, b]
    for i in range(N_NODE):
        K[2*i, 2*i] += k_node[i]
    K = csc_matrix(K)
    M_diag = np.zeros(nd)
    for i in range(N_NODE):
        m_t = rhoA_p * DZ_TRIB[i]
        M_diag[2*i]     = m_t
        M_diag[2*i + 1] = m_t * DZ**2 / 12.0
    return K, M_diag


def simulate_mdof_chain(k_node, F_load, omegas_ref, EI_p, rhoA_p, xi):
    K, M_diag = assemble_chain_matrices(k_node, EI_p, rhoA_p)
    nd     = 2 * N_NODE
    M_mat  = diags(M_diag, format='csc')
    w1, wN = omegas_ref[0], omegas_ref[-1]
    alpha  = 2.0 * xi * w1 * wN / (w1 + wN)
    beta_  = 2.0 * xi             / (w1 + wN)
    C      = alpha * M_mat + beta_ * K
    gamma, beta = 0.5, 0.25
    K_eff  = (M_mat + gamma * DT * C + beta * DT**2 * K).tocsc()
    solve  = splu(K_eff).solve

    d  = np.zeros(nd); v_ = np.zeros(nd); a = np.zeros(nd)
    F_dof = np.zeros(nd)
    u_sensor = np.zeros((S_SENSOR, M_STEP))
    for k in range(1, M_STEP):
        F_dof.fill(0.0)
        F_dof[0::2] = F_load[:, k]
        d_pred = d + DT * v_ + DT**2 * (0.5 - beta) * a
        v_pred = v_ + DT * (1.0 - gamma) * a
        rhs    = F_dof - C @ v_pred - K @ d_pred
        a_new  = solve(rhs)
        d      = d_pred + beta * DT**2 * a_new
        v_     = v_pred + gamma * DT   * a_new
        a      = a_new
        for j, idx in enumerate(SENSOR_IDX):
            u_sensor[j, k] = d[2 * int(idx)]
    return u_sensor


# ── 11-DOF reduced linear MDOF (modal Newmark) ─────────────────────────────
def build_reduced(omegas, Phi_full, rhoA_p, xi, DZ_S):
    M_eq    = np.diag(rhoA_p * DZ_S)
    Phi_red = Phi_full[SENSOR_IDX, :]
    try:
        Phi_inv = np.linalg.inv(Phi_red)
    except np.linalg.LinAlgError:
        Phi_inv = np.linalg.pinv(Phi_red, rcond=1e-10)
    K_eq = M_eq @ Phi_red @ np.diag(omegas**2) @ Phi_inv
    if not np.all(np.isfinite(K_eq)):
        Phi_inv = np.linalg.pinv(Phi_red, rcond=1e-8)
        K_eq    = M_eq @ Phi_red @ np.diag(omegas**2) @ Phi_inv
    K_eq = 0.5 * (K_eq + K_eq.T)
    w_a, w_b = omegas[RAYLEIGH_MODES[0]-1], omegas[RAYLEIGH_MODES[1]-1]
    alpha = 2.0 * xi * w_a * w_b / (w_a + w_b)
    beta_ = 2.0 * xi             / (w_a + w_b)
    C_eq  = alpha * M_eq + beta_ * K_eq
    xi_modes = alpha / (2.0 * omegas) + beta_ * omegas / 2.0
    return M_eq, C_eq, K_eq, Phi_red, xi_modes


class NewmarkModal:
    def __init__(self, omegas, xi_modes, Phi_red, dt, gamma=0.5, beta=0.25):
        self.omegas, self.xi_modes, self.Phi_red = omegas, xi_modes, Phi_red
        self.dt, self.gamma, self.beta = dt, gamma, beta
        self.a1 = 1.0 / (beta * dt * dt)
        self.a2 = 1.0 / (beta * dt)
        self.a3 = 1.0 / (2.0 * beta) - 1.0
        self.a4 = gamma / (beta * dt)
        self.a5 = gamma / beta - 1.0
        self.a6 = (gamma / (2.0 * beta) - 1.0) * dt
        self.k_n = omegas**2
        self.c_n = 2.0 * xi_modes * omegas
        self.K_eff_inv = 1.0 / (self.k_n + self.a1 + self.a4 * self.c_n)

    def run(self, F_phys, u0_phys, v0_phys):
        n_step = F_phys.shape[1]
        eta0,  *_ = np.linalg.lstsq(self.Phi_red, u0_phys, rcond=None)
        veta0, *_ = np.linalg.lstsq(self.Phi_red, v0_phys, rcond=None)
        F_modal   = self.Phi_red.T @ F_phys
        n_modes   = F_modal.shape[0]
        eta  = np.zeros((n_modes, n_step))
        veta = np.zeros_like(eta)
        aeta = np.zeros_like(eta)
        eta[:, 0]  = eta0
        veta[:, 0] = veta0
        aeta[:, 0] = F_modal[:, 0] - self.c_n * veta0 - self.k_n * eta0
        for k in range(1, n_step):
            rhs = (F_modal[:, k]
                   + (self.a1*eta[:, k-1] + self.a2*veta[:, k-1] + self.a3*aeta[:, k-1])
                   + self.c_n * (self.a4*eta[:, k-1] + self.a5*veta[:, k-1] + self.a6*aeta[:, k-1]))
            eta[:, k]  = self.K_eff_inv * rhs
            aeta[:, k] = self.a1 * (eta[:, k] - eta[:, k-1]) - self.a2 * veta[:, k-1] - self.a3 * aeta[:, k-1]
            veta[:, k] = veta[:, k-1] + self.dt * ((1.0 - self.gamma)*aeta[:, k-1] + self.gamma*aeta[:, k])
        return self.Phi_red @ eta


def force_sensor_strip(F_load):
    """Aggregate full-mesh nodal forces into the 11 sensor strips by summing
    the nodes whose tributary lies closer to that sensor than to its neighbours."""
    bnd = np.concatenate(([0.0], 0.5 * (Z_SENSOR[:-1] + Z_SENSOR[1:]), [L]))
    F_s = np.zeros((S_SENSOR, M_STEP))
    for j in range(S_SENSOR):
        if j < S_SENSOR - 1:
            mask = (Z_FULL >= bnd[j]) & (Z_FULL <  bnd[j + 1])
        else:
            mask = (Z_FULL >= bnd[j]) & (Z_FULL <= bnd[j + 1])
        F_s[j, :] = F_load[mask, :].sum(axis=0)
    return F_s


# ── Driver ─────────────────────────────────────────────────────────────────
def main():
    print('═══ Tunnel beam, MOVING-TRAIN dataset ═══')
    print(f'Mesh: {N_EL} elements; time: {M_STEP} steps × {DT}s = {T_TOTAL}s')
    print(f'Samples: {N_SAMPLES}    sensors: {S_SENSOR} at z = {Z_SENSOR.tolist()}')
    print(f'Random parameters: {PARAM_NAMES}')

    DZ_S       = sensor_tributaries()
    rng_master = np.random.default_rng(20260426)
    MAX_ATTEMPTS = 4 * N_SAMPLES
    seed_pool    = rng_master.integers(0, 2**31 - 1, size=MAX_ATTEMPTS)
    seeds        = np.zeros((N_SAMPLES, 2), dtype=seed_pool.dtype)

    u_lin_arr   = np.zeros((N_SAMPLES, S_SENSOR, M_STEP), dtype=np.float32)
    u_nl_arr    = np.zeros_like(u_lin_arr)
    u_red_arr   = np.zeros_like(u_lin_arr)
    u_chain_arr = np.zeros_like(u_lin_arr)
    q_arr       = np.zeros_like(u_lin_arr)              # nodal force on sensor strip
    timings     = np.zeros((N_SAMPLES, 5))   # [load, eigen, FE-lin, FE-nl, red+chain]
    err_red_lin = np.zeros(N_SAMPLES)
    err_red_nl  = np.zeros(N_SAMPLES)
    err_lin_nl  = np.zeros(N_SAMPLES)
    params_arr  = np.zeros((N_SAMPLES, len(PARAM_NAMES)))
    omegas_arr  = np.zeros((N_SAMPLES, N_MODES_KEEP))
    M_eq_arr    = np.zeros((N_SAMPLES, S_SENSOR, S_SENSOR))
    K_eq_arr    = np.zeros((N_SAMPLES, S_SENSOR, S_SENSOR))
    C_eq_arr    = np.zeros((N_SAMPLES, S_SENSOR, S_SENSOR))

    z0 = np.zeros(S_SENSOR)
    print('\n── Sample loop ──')
    t_loop   = time.time()
    s        = 0
    attempt  = 0
    n_failed = 0
    while s < N_SAMPLES and attempt < MAX_ATTEMPTS:
        seed_p = int(seed_pool[attempt])
        attempt += 1
        rng_p   = np.random.default_rng(seed_p)
        p       = sample_params(rng_p)
        spec    = TRAIN_SPECS[TRAIN_TYPES[p['train_type']]]
        v_mps   = p['v_kmh'] / 3.6

        try:
            k_node = build_nodal_spring_stiffness(p)
            F_cap  = build_nodal_capacity(p)
            A_p, I_p = section_props(p['t_wall'])
            rhoA_p   = p['rho'] * A_p
            EI_p     = p['zeta'] * p['E'] * I_p

            t = time.time()
            F_load = build_load_history(spec, v_mps)
            t_load = time.time() - t

            t = time.time()
            omegas, Phi_full = extract_eigen(p, k_node)
            M_eq, C_eq, K_eq, Phi_red, xi_modes = build_reduced(
                omegas, Phi_full, rhoA_p, p['xi'], DZ_S)
            integrator = NewmarkModal(omegas, xi_modes, Phi_red, DT)
            t_eig = time.time() - t

            t = time.time()
            u_lin = run_fe_transient(p, k_node, F_cap, F_load, omegas, kind='elastic')
            t_lin = time.time() - t

            t = time.time()
            u_nl  = run_fe_transient(p, k_node, F_cap, F_load, omegas, kind='plastic')
            t_nl  = time.time() - t

            t = time.time()
            F_s    = force_sensor_strip(F_load)
            u_red  = integrator.run(F_s, z0, z0)
            u_chain = simulate_mdof_chain(k_node, F_load, omegas, EI_p, rhoA_p, p['xi'])
            t_red  = time.time() - t
        except RuntimeError as exc:
            n_failed += 1
            print(f'  [skip attempt {attempt:4d}] FE non-convergence: {exc}  '
                  f'(failures: {n_failed})')
            continue

        if not (np.all(np.isfinite(u_lin)) and np.all(np.isfinite(u_nl))
                and np.all(np.isfinite(u_red)) and np.all(np.isfinite(u_chain))):
            n_failed += 1
            print(f'  [skip attempt {attempt:4d}] non-finite displacements  '
                  f'(failures: {n_failed})')
            continue

        peak_u = max(np.max(np.abs(u_lin)), np.max(np.abs(u_nl)))
        if peak_u > MAX_PEAK_U_M:
            n_failed += 1
            print(f'  [skip attempt {attempt:4d}] peak |u| = {peak_u*1e3:.1f} mm '
                  f'> {MAX_PEAK_U_M*1e3:.0f} mm  (failures: {n_failed})')
            continue

        seeds[s]      = (seed_p, p['train_type'])
        params_arr[s] = [p[k] for k in PARAM_NAMES]
        omegas_arr[s] = omegas
        M_eq_arr [s]  = M_eq
        K_eq_arr [s]  = K_eq
        C_eq_arr [s]  = C_eq
        timings[s]    = (t_load, t_eig, t_lin, t_nl, t_red)

        lin_max = np.max(np.abs(u_lin)) + 1e-30
        nl_max  = np.max(np.abs(u_nl))  + 1e-30
        err_red_lin[s] = np.max(np.abs(u_red - u_lin)) / lin_max
        err_red_nl [s] = np.max(np.abs(u_red - u_nl )) / nl_max
        err_lin_nl [s] = np.max(np.abs(u_lin - u_nl )) / nl_max

        u_lin_arr  [s] = u_lin  .astype(np.float32)
        u_nl_arr   [s] = u_nl   .astype(np.float32)
        u_red_arr  [s] = u_red  .astype(np.float32)
        u_chain_arr[s] = u_chain.astype(np.float32)
        q_arr      [s] = force_sensor_strip(F_load).astype(np.float32)

        if (s + 1) % 5 == 0 or s == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / (s + 1) * (N_SAMPLES - s - 1)
            print(f'  {s+1:4d}/{N_SAMPLES}  '
                  f'train={TRAIN_TYPES[p["train_type"]]:9s}  '
                  f'v={p["v_kmh"]:5.1f}km/h  '
                  f'f₁={omegas[0]/(2*np.pi):4.2f}Hz  '
                  f'lin={t_lin:5.2f}s  nl={t_nl:5.2f}s  '
                  f'red+chain={t_red*1e3:5.0f}ms   '
                  f'ε(red|lin)={err_red_lin[s]*100:4.2f}%  '
                  f'ε(lin|nl)={err_lin_nl[s]*100:4.2f}%   '
                  f'ETA={eta/60:5.1f}min')
        s += 1

    if s < N_SAMPLES:
        raise RuntimeError(
            f'Only {s}/{N_SAMPLES} samples after {attempt} attempts ({n_failed} failed)')
    print(f'  collected {s} in {attempt} attempts ({n_failed} skipped)')

    total = time.time() - t_loop
    print(f'\n── Done in {total/60:.1f} min ──')

    # ── Nonlinear-sample summary ───────────────────────────────────────────
    nl_mask = err_lin_nl > NONLIN_THRESHOLD
    nl_pct  = 100.0 * nl_mask.mean()
    print(f'\nNonlinear-case rate '
          f'(max|u_lin − u_nl| / max|u_nl| > {NONLIN_THRESHOLD*100:.1f}%):')
    print(f'  overall : {nl_mask.sum():4d}/{N_SAMPLES} samples = {nl_pct:5.1f}%')
    train_type_col = params_arr[:, PARAM_NAMES.index("train_type")].astype(int)
    for tt_id, tt_name in enumerate(TRAIN_TYPES):
        m  = (train_type_col == tt_id)
        if m.sum() == 0:
            continue
        sub = nl_mask & m
        print(f'  {tt_name:<10}: {sub.sum():4d}/{m.sum():<3d} = '
              f'{100.0*sub.sum()/m.sum():5.1f}%   '
              f'mean ε(lin|nl) = {err_lin_nl[m].mean()*100:5.2f}%   '
              f'max ε(lin|nl) = {err_lin_nl[m].max()*100:5.2f}%')
    if nl_mask.any():
        print(f'  among nonlinear samples : '
              f'mean ε(lin|nl) = {err_lin_nl[nl_mask].mean()*100:5.2f}%   '
              f'max = {err_lin_nl[nl_mask].max()*100:5.2f}%')

    # Save dataset (mirrors tunnel_full_vs_reduced.npz, plus u_chain) ───────
    out = __import__('os').environ.get('OUT_DB', 'tunnel_moving_train_database.npz')
    np.savez_compressed(
        out,
        u_lin=u_lin_arr, u_nl=u_nl_arr, u_red=u_red_arr, u_chain=u_chain_arr,
        q_sensor=q_arr,
        z_sensor=Z_SENSOR, t=T_FULL, seeds=seeds,
        timings=timings,
        err_red_lin=err_red_lin, err_red_nl=err_red_nl, err_lin_nl=err_lin_nl,
        params=params_arr, param_names=np.array(PARAM_NAMES),
        omegas=omegas_arr, M_eq=M_eq_arr, K_eq=K_eq_arr, C_eq=C_eq_arr,
        L=L, D=D,
        dt=DT, T_total=T_TOTAL,
        train_types=np.array(TRAIN_TYPES),
    )
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
