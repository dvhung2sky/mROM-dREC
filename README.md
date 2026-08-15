# mROM-dREC

Source code and processed data for *Multi-fidelity framework for nonlinear structural
dynamic response: mechanics-informed reduced models with data-driven residual correctors*.

The repository contains an interactive viewer for the three-arm benchmark on four
structural systems: a low-fidelity mechanics-informed reduced model with no learning
(Arm 1), a purely data-driven surrogate (Arm 2), and the multi-fidelity corrector that
adds a learned residual to Arm 1 (Arm 3).

## Run it

```bash
git clone https://github.com/dvhung2sky/mROM-dREC
cd mROM-dREC
pip install -r requirements.txt
python3 webapp/mf_app.py
```

Then open <http://127.0.0.1:8000>. Use `PORT=8802 python3 webapp/mf_app.py` for a
different port. Run it from the repository root.

NumPy is the only requirement. There is no web framework, no download step, and the
application makes no network requests.

## What is included

| path | contents |
|---|---|
| `webapp/mf_app.py` | the viewer, one file on the Python standard-library HTTP server |
| `webapp/index.html` | the interface |
| `release_webapp/0*/arm2`, `arm3` | training scripts for both learned arms, and the per-seed result tables and logs of the published runs |
| `release_webapp/0*/numerical_model` | the finite element models that generate the high-fidelity data |
| `release_webapp/0*/database` | processed results: the response histories for the held-out test samples |

## Numerical models

Each system ships the finite element model that produces its high-fidelity data, so the
databases behind the published results can be regenerated from source.

| system | model | solver |
|---|---|---|
| `01_sdof/numerical_model` | bilinear oscillator, Newmark integration | NumPy |
| `02_frame_v3/numerical_model` | two-bay three-storey frame, concentrated plastic hinges at the member ends | NumPy |
| `03_tunnel/numerical_model` | 200 m Euler–Bernoulli tube on an elastoplastic Winkler bed under a moving train | **OpenSees** (`openseespy`) |
| `04_turbine/numerical_model` | tapered tower and embedded monopile, rigid rotor–nacelle assembly, elastoplastic API-sand p–y springs, mudline plastic hinge, thrust from the wind speed relative to the moving hub | NumPy / SciPy |

`openseespy` is required for the tunnel generator only; nothing else in the repository
needs it, and the viewer needs neither.

The turbine pipeline runs in three steps, wrapped by `04_turbine/numerical_model/RUN.sh`:
`gen_wind_turbine_database.py` evaluates the model in parallel shards,
`consolidate_wt.py` merges them into the layout the learning scripts expect, and
`cap_vhub.py` keeps the V_hub ≤ 20.2 m/s operating envelope, which is the 1248 of 1597
samples the published turbine results use.

## Scope of the distributed results

The viewer serves **archived predictions** for the samples of the seed-42 held-out test
split, exported from the published runs:

| system | test samples | of |
|---|---|---|
| SDOF oscillator | 225 | 1500 |
| Three-storey steel frame | 146 | 970 |
| Buried tunnel under a moving train | 75 | 500 |
| Wind turbine on a monopile | 187 | 1248 |

For each of those samples the repository carries four response histories, the non-linear
finite element reference and one prediction per arm, together with the split indices so
that any displayed sample can be confirmed to have been held out from training.

Three things are deliberately **not** distributed: the trained network weights, the
training and validation samples, and the full simulation databases. The viewer therefore
needs no PyTorch and performs no inference; it displays results that were computed once
by the scripts in this repository. Those scripts and the finite element generators are
included, so the databases and the models can be regenerated from source.

Metrics shown next to each plot, the coefficient of determination, the root-mean-square
error as a percentage of the response amplitude and the peak error, are computed by the
viewer from the archived curves rather than read from a table.

## Notes on reading the results

The oscillator's two learned arms do not share a data split, a consequence of the
published training scripts, so a sample held out from Arm 1 and Arm 3 is frequently
training or validation data for Arm 2. The viewer states this on the split badge rather
than letting Arm 2 appear stronger than it is.

For the tunnel and the wind turbine the summary metric is the unweighted mean over
sensors rather than the pooled value, because the response amplitude varies by two orders
of magnitude along the tunnel and pooling would hide the low-amplitude sensors entirely.

## Regenerating everything

`release_webapp/0*/numerical_model` builds the high-fidelity databases; `openseespy` is
required for the tunnel generator only. `release_webapp/0*/arm2` and `arm3` then train the
two learned arms and write the result tables reproduced here. Training uses a GPU; the
viewer does not.
