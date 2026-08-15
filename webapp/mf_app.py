#!/usr/bin/env python3
"""
Multi-fidelity surrogate demonstrator — companion app for the four release examples.

Serves release_webapp/ : pick a case study, a seed and a sample, and the
response histories are produced on demand from the archived database and the
archived checkpoints.

    Reference       full non-linear FE                      (u_nl, the target)
    Arm 1           low-fidelity model, no learning         (u_red / u_lin)
    Arm 2           data-driven, load -> response           (trained)
    Arm 3           multi-fidelity, Arm 1 + learned residual(trained)

Capability differs per row, and the UI states it rather than hiding it:
SDOF, tunnel and turbine ship .pt for all five seeds and run live. The frame
archives per-seed JSON only, so it is served from the seed-42 predictions
exported by the published run itself.

Run:   python3 webapp/mf_app.py        then open http://127.0.0.1:8000
"""
from __future__ import annotations

import csv
import json
import re
import os
import statistics as st
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    import torch
except ImportError:          # a stored-only distribution ships no weights to load
    torch = None

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(ROOT, "release_webapp")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SEEDS = [42]   # the published results are the seed-42 run

# ── case registry ────────────────────────────────────────────────────────────
CASES = {
    "sdof": dict(
        row="SDOF (eta 0.60-0.96, N=1500)", metric_key="pooled",
        label="SDOF oscillator",
        sub="bilinear kinematic hardening, η ∈ [0.60, 0.96], N = 1500",
        dir="01_sdof", db="sdof_eta60_n1500.npz",
        kind="sdof", sensor_name="DOF", unit="mm", scale=1e3,
        stored="database/pub_sdof_seed42_arm{}_pred.npz",
        arm3_script="arm3/sdof_arm3_noepp_db.py", arm3_ckpt="best_nlsdof_multifidelity.pt",
        arm2_script="arm2/arm2_split15_db.py",   arm2_ckpt="best_arm2_nofb_SDOF.pt",
        env={"DB": "release_webapp/01_sdof/database/sdof_eta60_n1500.npz",
             "ARM2_FEEDBACK": "0"},
    ),
    "frame3": dict(
        row="3-story frame v3", metric_key="pooled",
        label="3-story 2-bay steel moment frame",
        sub="full 2-D FEM, 15 concentrated SCWB hinges, N = 970",
        dir="02_frame_v3", db="frame3_v3_arm_database.npz",
        kind="stored", sensor_name="Floor", unit="mm", scale=1e3, seeds=[42],
        # arm2/ and arm3/ archive per-seed JSON metrics only; these two .npz are the
        # all-sample predictions exported from the published seed-42 run itself.
        stored="database/pub_frame3_seed42_arm{}_pred.npz",
    ),
    "turbine": dict(
        # V_hub-capped envelope: the 1248 of 1597 samples with V_hub <= 20.2 m/s, seed 42.
        # Above rated wind the pitch controller sheds thrust and aerodynamic damping rises,
        # so the fore-aft fluctuation shrinks against the mean — that regime dominates the
        # per-sample failures. Capping it lifts the Arm 3 per-sample median 0.6829 -> 0.8223
        # and drops samples below zero 16% -> 13%. Cost: Arm 1 and Arm 2 gain more than Arm 3
        # does, so A3-A2 on the per-sample median narrows +0.705 -> +0.441. Artefacts ship
        # under webapp/ rather than in the release bundle, which holds the published runs.
        row="Turbine V_hub<=20.2", metric_key="unweighted",
        label="Wind turbine on a monopile",
        sub="revised 60 s example, hub wind ≤ 20.2 m/s operating envelope, N = 1248",
        dir="04_turbine", db="wt_v20_full.npz",
        kind="wt", sensor_name="Sensor", unit="mm", scale=1e3,
        stored="database/pub_turbine_seed42_arm{}_pred.npz",
        arm3_script="arm3/wt_arm3_v20.py", arm3_ckpt="best_wt_arm3_v20_full_s{seed}.pt",
        arm2_script="arm2/wt_arm2_v20.py", arm2_ckpt="best_wt_arm2_v20_full_s{seed}.pt",
        env={"DB_PATH": "release_webapp/04_turbine/database/wt_v20_full.npz",
             "N_MAX": "0", "FLIP": "1", "STEP": "50", "SAMPLE_W": "0"},
        # not in RESULTS.csv; each arm's own results CSV carries the test-split metrics
        results_csv=("arm3/wt_arm3_v20_full{suf}_results.csv",
                     "arm2/wt_arm2_v20_full{suf}_results.csv"),
        metric="unweighted mean over sensors",
    ),
    "tunnel": dict(
        row="Tunnel moving-train (N=500)", metric_key="unweighted",
        label="Buried tunnel under a moving train",
        sub="200 m Euler–Bernoulli tube on an elastoplastic Winkler bed, N = 500",
        dir="03_tunnel", db="tunnel_moving_train_database.npz",
        kind="wt", sensor_name="Sensor", unit="mm", scale=1e3,
        stored="database/pub_tunnel_seed42_arm{}_pred.npz",
        arm3_script="arm3/red_to_nl_tunnel_half.py",
        arm3_ckpt="best_tunnel_half_arm3_s{seed}.pt",
        arm2_script="arm2/red_to_nl_tunnel_arm2_half.py",
        arm2_ckpt="best_tunnel_half_arm2_s{seed}.pt",
        # N_MAX caps the pool at the first 500 samples *before* the split, so it
        # must be set or the seeds reproduce a different (1000-sample) run.
        env={"N_MAX": "500"},
        # summary row is the unweighted mean of the 11 per-sensor R2, not the
        # pooled value the other rows use — the response spans 0.05-11 mm along
        # the tube, so pooling hides the near-field sensors entirely.
        metric="unweighted mean over sensors",
    ),
}

CUT_WT = "film, rep, head, lskip, history, wall = train()"
_cache: dict = {}
_lock = threading.Lock()


def rel(case, *p):
    # most rows live in the release bundle; a row may set root=HERE to ship its own
    # artefacts under webapp/ instead (see the turbine)
    return os.path.join(CASES[case].get("root", REL), CASES[case]["dir"], *p)


def ckpt_path(case, arm, seed):
    """Three archived layouts: seed in a subdirectory (SDOF), seed in the filename
    via TAG (tunnel, turbine seeds 0-3), or no seed at all because the run used the
    default TAG (turbine seed 42)."""
    # the checkpoint always sits beside the script that wrote it (or in a seedN/ under it)
    base = os.path.dirname(rel(case, CASES[case][f"{arm}_script"]))
    tpl = CASES[case][f"{arm}_ckpt"]
    names = [tpl.format(seed=seed), re.sub(r"_s\{seed\}", "", tpl)]
    for name in names:
        for cand in (os.path.join(base, f"seed{seed}", name), os.path.join(base, name)):
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f"no {names[0]} for {case}/{arm}/seed{seed}")


# ── loaders ──────────────────────────────────────────────────────────────────
# Statistics the training scripts reduce over idx_tr. When the shipped database carries
# only the test split (make_demo_db.py) those rows are zeros, so the value is substituted
# from demo_* keys stored alongside the data. The anchor is the first use of each.
# q_sd needs no entry: q_sensor is shipped whole, so it computes correctly.
DEMO_ANCHORS = {"sigma_target": "target_norm = (residual_raw / sigma_target)"}


def demo_consts(case, arm):
    """demo_* constants stored in the database, empty for a full database.

    Deliberately takes no lock: the only caller is load_wt(), which already runs inside
    get()'s `with _lock`, and _lock is a plain threading.Lock. Re-acquiring it here
    deadlocks the request thread. The cache write is idempotent, so racing it is harmless.
    """
    key = (case, "demo", arm)
    if key not in _cache:
        db = CASES[case].get("db")
        p = rel(case, "database", db) if db else None
        out = {}
        if p and os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            # stored per arm, since Arm 3 normalises the residual u_nl - u_red and
            # Arm 2 the response itself: demo_sigma_target_arm3 -> sigma_target
            sfx = "_" + arm
            out = {k[5:-len(sfx)]: d[k] for k in d.files
                   if k.startswith("demo_") and k.endswith(sfx)}
        _cache[key] = out
    return _cache[key]


def _exec_prefix(path, cut, env=None, demo=None):
    # the release scripts read DB / SEED / ARM2_FEEDBACK from the environment
    # (see each row's RUN.sh); without them they silently fall back to the
    # wrong database and to seed 42.
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    src = open(path).read()
    # Some scripts pin SEED as a module-level literal instead of reading it from the
    # environment (both SDOF arms do), which silently makes the seed selector inert:
    # the seed-N checkpoint would be evaluated against the seed-42 split and its
    # seed-42 normalisation. Rewrite the literal for those.
    if (env or {}).get("SEED") is not None:
        src = re.sub(r"^SEED\s*=\s*\d+", f"SEED = {env['SEED']}", src, count=1, flags=re.M)
    # Both tunnel scripts hard-code a bare filename, which resolves only if a copy of the
    # 172 MB database happens to sit in the working directory. Point it at the bundle.
    m = re.search(r'^DB_PATH = "([A-Za-z0-9_.\-]+\.npz)"', src, flags=re.M)
    if m:
        cand = os.path.join(os.path.dirname(path), os.pardir, "database", m.group(1))
        if os.path.exists(cand):
            src = src[:m.start()] + f"DB_PATH = {os.path.abspath(cand)!r}" + src[m.end():]
    for name, val in (demo or {}).items():
        anchor = DEMO_ANCHORS.get(name)
        if anchor and anchor in src:
            src = src.replace(anchor, f'{name} = _DEMO["{name}"]\n{anchor}', 1)
    if cut not in src:
        raise RuntimeError(f"{os.path.basename(path)}: cut marker not found")
    ns = {"__name__": "_mf_" + os.path.basename(path).replace(".", "_"),
          "_DEMO": demo or {}}
    exec(compile(src[: src.index(cut)], path, "exec"), ns)
    return ns


def _env(case, seed):
    e = dict(CASES[case].get("env") or {})
    e["SEED"] = str(seed)
    return e


def load_wt(case, arm, seed):
    """f3v3/wt lineage: infer(film, rep, head, lskip, idx) -> (n, T, S)."""
    ns = _exec_prefix(rel(case, CASES[case][f"{arm}_script"]), CUT_WT, _env(case, seed),
                      demo_consts(case, arm))
    film = ns["FiLMStaticEncoder"](n_static=ns["N_STATIC"], n_time=ns["N_TIME_CH"],
                                   dropout=ns["FILM_DROPOUT"])
    rep = ns["RepresentationLearning"](input_size=ns["N_TIME_CH"] + 1, dropout=ns["GRU_DROPOUT"])
    head = ns["JointHead"](hidden_size=rep.out_size)
    lskip = ns["LoadSkip"](n_static=ns["N_STATIC"])
    # the 70/15/15 tunnel checkpoint also carries SPLIT/SEED metadata, which the
    # weights_only=True default refuses to unpickle. These files are ours.
    sd = torch.load(ckpt_path(case, arm, seed), map_location="cpu", weights_only=False)
    for m, k in ((film, "film"), (rep, "rep"), (head, "head"), (lskip, "lskip")):
        m.load_state_dict(sd[k]); m.eval()
    fn = lambda idx: ns["infer"](film, rep, head, lskip, idx)
    # Every wt-lineage script filters the raw network output through
    # physics_lowpass() before scoring, and that definition sits *after* the
    # cut marker, so exec'ing the prefix alone silently serves unfiltered
    # predictions and undershoots the published R2.
    src = open(rel(case, CASES[case][f"{arm}_script"])).read()
    a, b = "from scipy.signal import butter", "\npred_val = physics_lowpass("
    if a in src and b in src:
        exec(compile(src[src.index(a):src.index(b)], "<lowpass>", "exec"), ns)
        raw, lp = fn, ns["physics_lowpass"]
        fn = lambda idx, raw=raw, lp=lp: lp(raw(idx), idx)
    return dict(ns=ns, fn=fn)


def load_sdof_arm3(case, seed):
    """good_multifidelity lineage: the training loop sits at module level between the
    model construction and infer_overlap_add(), so splice around it rather than
    exec a prefix that would retrain the network."""
    for k, v in _env(case, seed).items():
        os.environ[k] = str(v)
    path = rel(case, CASES[case]["arm3_script"])
    src = open(path).read()
    # the script pins SEED as a literal and never reads it from the environment
    src = re.sub(r"^SEED\s*=\s*\d+", f"SEED = {seed}", src, count=1, flags=re.M)
    tr = src.index("\nfor epoch in range(1, N_EPOCHS")          # training starts here
    inf = src.index("\ndef infer_overlap_add(")                  # and ends before this
    end = src.index("pred_u, true_u = infer_overlap_add(")
    # keep any helpers (make_tukey_taper, ...) that live inside the skipped region
    import ast
    tl, il = src[:tr].count("\n") + 1, src[:inf].count("\n") + 1
    helpers = "\n".join(
        seg for n in ast.parse(src).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and tl <= n.lineno <= il
        and (seg := ast.get_source_segment(src, n)))
    ns = {"__name__": "_mf_sdof_arm3"}
    exec(compile(src[:tr] + "\n" + helpers + "\n" + src[inf:end], path, "exec"), ns)
    # infer_overlap_add() hardcodes torch.load("best_nlsdof_multifidelity.pt"),
    # which resolves to the project root rather than the seed directory. Shim the
    # module-level `torch` it resolves against so that one filename is redirected.
    want = ckpt_path(case, "arm3", seed)

    class _TorchShim:
        def __init__(self, mod, target):
            self._m, self._t = mod, target

        def __getattr__(self, n):
            return getattr(self._m, n)

        def load(self, f, *a, **k):
            if isinstance(f, str) and f.endswith(os.path.basename(self._t)):
                f = self._t
            return self._m.load(f, *a, **k)

    ns["torch"] = _TorchShim(torch, want)
    return dict(ns=ns, fn=lambda idx: ns["infer_overlap_add"](idx)[0][..., None])


def load_sdof_arm2(case, seed):
    """datadriven lineage: prep + overlap-add rollout replicated from run()."""
    p = rel(case, CASES[case]["arm2_script"])
    ns = _exec_prefix(p, "def run(system)", _env(case, seed))
    dev = ns["device"]
    L_in, L_out, L_all = ns["L_in"], ns["L_out"], ns["L_all"]
    db, ukeys, _ = ns["SYSTEMS"]["SDOF"]
    d = np.load(os.path.join(ROOT, db))
    good = np.where(d["converged"])[0]
    Fr = d["F"].astype(np.float32)[good]
    U = np.stack([d[k].astype(np.float32)[good] for k in ukeys], -1)
    P = d["params"].astype(np.float32)[good]
    N, T = Fr.shape
    idx = np.random.default_rng(ns["SEED"]).permutation(N)
    ntr, nva = int(0.70 * N), int(0.15 * N)
    f_sd = Fr[idx[:ntr]].std() + 1e-12
    u_sd = U[idx[:ntr]].std(axis=(0, 1)) + 1e-12
    p_mu, p_sd = P[idx[:ntr]].mean(0), P[idx[:ntr]].std(0) + 1e-12
    Fn, Pn = Fr / f_sd, (P - p_mu) / p_sd
    n_dof = len(ukeys)

    n_time = 1                                     # FEEDBACK=0 checkpoint: force only
    film = ns["FiLMStaticEncoder"](P.shape[1], n_time).to(dev)
    rep = ns["RepresentationLearning"](n_time + 1, ns["HIDDEN"], ns["N_GRU_LAYERS"],
                                       dropout=ns["DROPOUT"]).to(dev)
    head = ns["PredictionHead"](ns["HIDDEN"], n_dof).to(dev)
    ck = torch.load(ckpt_path(case, "arm2", seed), map_location=dev, weights_only=False)
    film.load_state_dict(ck["film"]); rep.load_state_dict(ck["rep"]); head.load_state_dict(ck["head"])
    for m in (film, rep, head):
        m.eval()
    taper = ns["make_tukey_taper"](L_out, alpha=0.3)

    def fn(sample_idx):
        chunk = np.asarray(sample_idx)
        nb = len(chunk)
        fp = np.stack([np.concatenate([np.zeros(L_in, np.float32), Fn[si]]) for si in chunk])
        acc = np.zeros((nb, T + L_in, n_dof), np.float32)
        wgt = np.zeros((nb, T + L_in, 1), np.float32)
        sb = torch.from_numpy(Pn[chunk][:, None, :]).to(dev)
        with torch.no_grad():
            for stp in range(0, T, L_out // 2):
                if stp + L_all > T + L_in:
                    break
                xb = torch.from_numpy(fp[:, stp:stp + L_all, None]).to(dev)
                pr = head(rep(film(xb, sb))[0]).cpu().numpy()
                acc[:, stp + L_in:stp + L_all] += pr * taper[None, :, None]
                wgt[:, stp + L_in:stp + L_all] += taper[None, :, None]
        sw = np.where(wgt > 0, wgt, 1.0)
        return ((acc / sw)[:, L_in:] * u_sd).astype(np.float32)

    # The Arm 3 script draws rng.permutation(N) twice (a duplicated line), so its
    # split is NOT this one — 159 of Arm 3's 225 test samples are Arm 2 training
    # samples. Expose Arm 2's own split so the two can be reconciled.
    return dict(ns=ns, fn=fn, good=good,
                idx=dict(idx_tr=idx[:ntr], idx_val=idx[ntr:ntr + nva], idx_te=idx[ntr + nva:]))


def get(case, arm, seed):
    key = (case, arm, seed)
    with _lock:
        if key not in _cache:
            t0 = time.time()
            print(f"  loading {case}/{arm}/seed{seed} …", flush=True)
            if CASES[case]["kind"] == "wt":
                _cache[key] = load_wt(case, arm, seed)
            elif arm == "arm3":
                _cache[key] = load_sdof_arm3(case, seed)
            else:
                _cache[key] = load_sdof_arm2(case, seed)
            print(f"  ready in {time.time() - t0:.1f}s", flush=True)
        return _cache[key]


# ── baseline-only rows: read the database directly ───────────────────────────
def kind_of(case):
    """"stored" when the exported prediction bundles are present, else the declared kind.

    The distribution ships bundles and no checkpoints, so it serves stored curves and
    never imports torch. The working copy has no bundles and runs live inference. Same
    file, no flag, and the answer is a fact about the directory rather than a setting.
    """
    c = CASES[case]
    tpl = c.get("stored")
    if tpl and all(os.path.exists(rel(case, tpl.format(a))) for a in (3, 2)):
        return "stored"
    return c["kind"]


def load_stored(case):
    """Predictions exported from the published run itself (see RUN.sh / the
    seed-sweep runner), so no retraining and no second model lineage."""
    key = (case, "stored", 0)
    with _lock:
        if key not in _cache:
            f = CASES[case]["stored"]
            d3 = np.load(rel(case, f.format(3)))
            d2 = np.load(rel(case, f.format(2)))
            idx2 = {"idx_tr": d2[k] for k in ("idx_tr2",) if k in d2.files}
            if "idx_tr2" in d2.files:                 # arms with different splits
                idx2 = {"idx_tr": d2["idx_tr2"], "idx_val": d2["idx_va2"],
                        "idx_te": d2["idx_te2"]}
            _cache[key] = dict(ref=d3["u_nl"], lin=d3["u_red"],
                               mf=d3["pred"], dd=d2["pred"],
                               idx={"idx_tr": d3["idx_tr"], "idx_val": d3["idx_va"],
                                    "idx_te": d3["idx_te"]}, idx2=idx2)
        return _cache[key]


def load_meta(case):
    """t and z_sensor for a stored case, from a metadata file instead of a database.

    A stored distribution ships no database at all, so the few vectors the plot needs
    live in <case>_meta.npz. Falls back to the database when one is present.
    """
    key = (case, "meta", 0)
    if key not in _cache:
        p = rel(case, "database", f"{case}_meta.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            _cache[key] = dict(t=np.asarray(d["t"], float),
                               z=(np.asarray(d["z_sensor"], float).tolist()
                                  if "z_sensor" in d.files else None))
        else:
            b = load_baseline(case)
            _cache[key] = dict(t=b["t"], z=b["z"])
    return _cache[key]


def load_baseline(case):
    key = (case, "db", 0)
    with _lock:
        if key not in _cache:
            d = np.load(rel(case, "database", CASES[case]["db"]), allow_pickle=True)
            tr = lambda a: np.transpose(a, (0, 2, 1))          # (N,S,T) -> (N,T,S)
            has_bulk = "u_nl" in d.files and d["u_nl"].ndim == 3
            _cache[key] = dict(ref=tr(d["u_nl"]) if has_bulk else None,
                               lin=(tr(d["u_red"] if "u_red" in d.files else d["u_lin"])
                                    if has_bulk else None),
                               t=np.asarray(d["t"], float),
                               z=(np.asarray(d["z_sensor"], float).tolist() if "z_sensor" in d.files else None))
        return _cache[key]


# ── metrics ──────────────────────────────────────────────────────────────────
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def rms_frac(y, p):
    """RMS of (y - p) as a percentage of the RMS of y. Unlike a peak comparison
    this sees a constant plastic drift, which is what the SDOF ratcheting
    samples are dominated by."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(((y - p) ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-30) * 100)


def pk(y, p):
    a = np.abs(y).max()
    return float(abs(a - np.abs(p).max()) / (a + 1e-30) * 100)


def split_of(ns, sid):
    """train / validation / test membership, read from the script's own indices."""
    for key, name in (("idx_te", "test"), ("idx_val", "validation"), ("idx_tr", "train")):
        v = ns.get(key)
        if v is not None and sid in set(np.asarray(v).tolist()):
            return name
    return None


def predict(case, sid, seed):
    c = CASES[case]
    sc = c["scale"]
    _split = None
    if kind_of(case) == "stored":
        st = load_stored(case)
        b = load_meta(case)
        ref, lin = np.asarray(st["ref"][sid], float), np.asarray(st["lin"][sid], float)
        t, z, n = b["t"], b["z"], st["ref"].shape[0]
        if not 0 <= sid < n:
            raise IndexError(f"id must be in [0, {n-1}]")
        _split = split_of(st["idx"], sid)
        o = split_of(st.get("idx2") or {}, sid)
        if o and o != _split:                        # keep the honest Arm 2 caveat
            _split = f"{_split} (Arm 2: {o})"
        arms = {"linear": lin,
                "multifidelity": np.asarray(st["mf"][sid], float),
                "datadriven": np.asarray(st["dd"][sid], float)}
    elif not has_weights(case):
        b = load_baseline(case)
        ref, lin, t, z = b["ref"][sid], b["lin"][sid], b["t"], b["z"]
        arms = {"linear": lin}
        n = b["ref"].shape[0]
    elif c["kind"] == "wt":
        a3 = get(case, "arm3", seed); ns = a3["ns"]
        n = int(ns["N"])
        if not 0 <= sid < n:
            raise IndexError(f"id must be in [0, {n-1}]")
        ref = np.asarray(ns["u_nl_raw"][sid], float)
        lin = np.asarray(ns["u_red_raw"][sid], float)
        t = np.asarray(ns["t_axis"], float)
        z = np.asarray(ns["z_sensor"], float).tolist() if ns.get("z_sensor") is not None else None
        _split = split_of(ns, sid)
        arms = {"linear": lin,
                "multifidelity": np.asarray(a3["fn"](np.array([sid]))[0], float)}
        if c.get("arm2_script"):
            arms["datadriven"] = np.asarray(
                get(case, "arm2", seed)["fn"](np.array([sid]))[0], float)
    else:                                                        # sdof
        a3 = get(case, "arm3", seed); ns = a3["ns"]
        ref = np.asarray(ns["u_nl_raw"], float)
        n = ref.shape[0]
        if not 0 <= sid < n:
            raise IndexError(f"id must be in [0, {n-1}]")
        ref = ref[sid][:, None] if ref.ndim == 2 else ref[sid]
        lin = np.asarray(ns["u_lin_raw"], float)[sid]
        lin = lin[:, None] if lin.ndim == 1 else lin
        t = next((np.asarray(ns[k], float) for k in ("t_arr", "t_axis", "t_vec") if k in ns),
                 np.arange(ref.shape[0]) / float(ns.get("FS", ns.get("fs", 100.0))))
        z = None
        _split = split_of(ns, sid)
        arms = {"linear": lin,
                "multifidelity": np.asarray(a3["fn"](np.array([sid]))[0], float)}
        if c.get("arm2_script"):
            a2 = get(case, "arm2", seed)
            arms["datadriven"] = np.asarray(a2["fn"](np.array([sid]))[0], float)
            # The two SDOF arms do not share a split (see load_sdof_arm2), so a
            # sample held out for Arm 1 / Arm 3 is usually Arm 2 training data.
            # Say so on the badge rather than let Arm 2 look good for free.
            o = split_of(a2["idx"], sid)
            if o and o != _split:
                _split = f"{_split} (Arm 2: {o})"

    S = ref.shape[1]
    series = {"reference": [np.round(ref[:, s] * sc, 4).tolist() for s in range(S)]}
    per = []
    for k, v in arms.items():
        series[k] = [np.round(np.asarray(v)[:, s] * sc, 4).tolist() for s in range(S)]
    for s in range(S):
        per.append({"sensor": s + 1,
                    "r2": {k: r2(ref[:, s], np.asarray(v)[:, s]) for k, v in arms.items()},
                    "peak_err": {k: pk(ref[:, s], np.asarray(v)[:, s]) for k, v in arms.items()},
                    "peak_ref": float(np.abs(ref[:, s]).max() * sc)})
    return {"case": case, "id": sid, "seed": seed, "n": n,
            "t": np.round(t, 5).tolist(), "z": z,
            "sensor_name": c["sensor_name"], "unit": c["unit"],
            "arms": list(arms), "series": series, "per_sensor": per,
            "overall": {"r2": {k: r2(ref, np.asarray(v)) for k, v in arms.items()},
                        "nonlinearity_pct": rms_frac(ref, arms["linear"])},
            "split": _split}


# ── published results ────────────────────────────────────────────────────────
def csv_metrics(case):
    """Runs that are not in RESULTS.csv, read from the results CSVs they wrote
    themselves — one row per seed. Both arms record the same u_red row, so Arm 1
    comes from either. Seed 42 used the default TAG, so its files carry no suffix."""
    c = CASES[case]
    rows = []
    for seed in (c.get("seeds") or SEEDS):
        suf = "" if seed == 42 else f"_s{seed}"
        out = {}
        for f, arm in zip(c["results_csv"], ("arm3", "arm2")):
            path = rel(case, f.format(suf=suf))
            if not os.path.exists(path):
                out = {}; break
            for r in csv.DictReader(open(path)):
                v = [float(r[f"r2_s{i}"]) for i in range(1, 6)]
                val = st.mean(v) if c["metric_key"] == "unweighted" else float(r["r2_overall"])
                out["arm1" if r["label"] == "u_red_ref" else
                    (arm if r["label"] == "trained" else None)] = val
        if all(a in out for a in ("arm1", "arm2", "arm3")):
            rows.append({a: out[a] for a in ("arm1", "arm2", "arm3")})
    return rows


def results_table():
    # RESULTS.csv carries both metrics for the tunnel and the turbine, so key on
    # (example, metric) and let each case pick the one its published row quotes.
    rows = {}
    with open(os.path.join(REL, "RESULTS.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault((r["example"], r["metric"]), []).append(
                {k: float(r[k]) for k in ("arm1", "arm2", "arm3")})
    out = []
    for cid, c in CASES.items():
        v = csv_metrics(cid) if c.get("results_csv") else rows.get((c["row"], c["metric_key"]))
        if not v:
            continue
        e = {"id": cid, "label": c["label"], "sub": c["sub"], "seeds": len(v),
             "metric": c.get("metric")}
        for a in ("arm1", "arm2", "arm3"):
            x = [d[a] for d in v]
            e[a] = {"mean": st.mean(x), "sd": (st.stdev(x) if len(x) > 1 else 0.0)}
        e["resid_removed"] = st.mean([1 - (1 - d["arm3"]) / (1 - d["arm1"]) for d in v]) * 100
        e["wins"] = sum(d["arm3"] > d["arm1"] for d in v)
        out.append(e)
    return out


def has_weights(case):
    c = CASES[case]
    if kind_of(case) in ("baseline", "stored"):
        return kind_of(case) == "stored"
    try:
        return all(os.path.exists(ckpt_path(case, a, (c.get("seeds") or SEEDS)[0]))
                   for a in ("arm2", "arm3") if c.get(f"{a}_script"))
    except FileNotFoundError:
        return False


def split_index(case, seed):
    """{split name: [sample ids]} for the selected seed, or {} if unknown."""
    c = CASES[case]
    if kind_of(case) == "stored":
        idx = load_stored(case)["idx"]
    elif c["kind"] == "baseline" or not has_weights(case):
        return {}
    else:
        idx = get(case, "arm3", seed)["ns"]
    out = {}
    for key, name in (("idx_tr", "training"), ("idx_val", "validation"),
                      ("idx_te", "test")):
        v = idx.get(key)
        if v is not None:
            out[name] = np.asarray(v).astype(int).tolist()
    # The SDOF Arm 3 script rebinds idx_val = idx_te before its final metrics, so
    # the two lists come back identical; offering both in the picker would imply a
    # validation split that is no longer reachable.
    if out.get("validation") == out.get("test"):
        out.pop("validation", None)
    return out


MISSING = set()      # cases whose database is absent; see fetch_data.py


def case_list():
    out = []
    for cid, c in CASES.items():
        live = has_weights(cid)
        out.append({"id": cid, "label": c["label"], "sub": c["sub"],
                    "live": live, "stored": kind_of(cid) == "stored",
                    "seeds": (c.get("seeds") or SEEDS) if live else [],
                    "note": c.get("note", ""),
                    "n": _n_samples(cid, c, live),
                    "caveat": bool(c.get("note")) and live})
    return out


def _n_samples(cid, c, live):
    """None when the database is absent, so one missing file cannot stop the app.

    The distributed repository ships without the large databases (fetch_data.py pulls
    them), and a clone should still start and serve whichever cases are present.
    """
    try:
        if kind_of(cid) == "stored":
            return load_stored(cid)["ref"].shape[0]
        return load_baseline(cid)["ref"].shape[0] if not live else None
    except FileNotFoundError:
        MISSING.add(cid)
        return None


# ── http ─────────────────────────────────────────────────────────────────────
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(),
                                  "text/html; charset=utf-8")
            if u.path == "/plotly.js":
                import plotly.offline as po
                return self._send(200, po.get_plotlyjs(), "application/javascript")
            if u.path == "/assets/huce_logo.png":
                f = os.path.join(HERE, "assets", "huce_logo.png")
                if os.path.exists(f):
                    return self._send(200, open(f, "rb").read(), "image/png")
                return self._send(404, b"", "image/png")
            if u.path == "/api/cases":
                return self._send(200, json.dumps(case_list()))
            if u.path == "/api/split":
                return self._send(200, json.dumps(split_index(
                    q.get("case", [""])[0], int(q.get("seed", ["42"])[0]))))
            if u.path == "/api/results":
                return self._send(200, json.dumps(results_table()))
            if u.path == "/api/predict":
                case = q.get("case", [""])[0]
                if case not in CASES:
                    return self._send(400, json.dumps({"error": f"unknown case {case!r}"}))
                return self._send(200, json.dumps(predict(
                    case, int(q.get("id", ["0"])[0]), int(q.get("seed", ["42"])[0]))))
            self._send(404, json.dumps({"error": "not found"}))
        except Exception as exc:                                    # noqa: BLE001
            import traceback; traceback.print_exc()
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


def main():
    port = int(os.environ.get("PORT", "8000"))
    print("Multi-fidelity surrogate demonstrator  —  release_webapp")
    for c in case_list():
        print(f"  {c['id']:9s} {'live inference' if c['live'] else 'reference + Arm 1 only'}"
              f"   {c['label']}")
    if MISSING:
        print(f"\n  database missing for: {', '.join(sorted(MISSING))}"
              f"\n  run  python3 fetch_data.py  to download it")
    print(f"\n  http://127.0.0.1:{port}    (Ctrl-C to stop)\n")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
