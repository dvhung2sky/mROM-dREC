# mROM-dREC

Source code for *Multi-fidelity framework for nonlinear structural dynamic response:
mechanics-based reduced-order models with data-driven residual correctors*.

The framework is compared in three arms on four structural systems:
- **Arm 1:** a mechanics-based reduced model with no learning.
- **Arm 2:** a purely data-driven surrogate.
- **Arm 3:** the multi-fidelity corrector, which adds a learned residual to Arm 1.

The repository contains:
- the finite element models that generate the high-fidelity data;
- the training scripts and configurations of both learned arms;
- the seeds, the per-seed results, and the notebooks and scripts behind every table and figure;
- a viewer for the archived test predictions.

## Contents

| path | contents |
|---|---|
| `regenerate_databases.sh` | rebuilds every high-fidelity database from the finite element models (see *Data*) |
| `release_webapp/0*/numerical_model` | finite element models and data-generation scripts of the four systems |
| `release_webapp/0*/arm2`, `arm3` | training scripts of Arms 2 and 3 with their configurations; per-seed results and logs of the published runs |
| `release_webapp/0*/RUN.sh` | the exact commands and seeds of the published runs |
| `release_webapp/0*/database` | archived predictions for the seed-42 test samples, with their split indices |
| `release_webapp/RESULTS.csv`, `make_table2.py` | per-seed R² of all runs; `python3 make_table2.py` regenerates `TABLE.csv` (Table 2) |
| `colab/` | Colab notebooks that reproduce Table 2, the MF-IA baseline and the supplementary studies |
| `figures/` | scripts that draw Figs. 5 and 6 |
| `webapp/` | the viewer: `python3 webapp/mf_app.py`, then open <http://127.0.0.1:8000> (NumPy only) |

## Data

The high-fidelity databases are **not distributed**. Two of them exceed GitHub's 100 MB
file limit, and all of them can be rebuilt from the finite element models in this
repository:

```bash
pip install numpy scipy openseespy       # openseespy is needed for the tunnel only
bash regenerate_databases.sh             # data/: the four databases of Table 2 (about 330 MB)
bash regenerate_databases.sh levels      # also data/levels/: the 20 databases of Table 6
```

Each generator draws its samples from a fixed random seed, so the script rebuilds the
same parameter sets as the published databases. NumPy, SciPy or OpenSees versions that
differ from ours may change the stored responses by round-off, but they do not change
the samples. Generation runs on a CPU. Measured on one core of an Apple M1 Max, it takes about
5 s for the oscillator, 6 min for the frame, 17 min for the tunnel and 77 min for the
turbine. The script splits the turbine across eight parallel processes.

| database | system | samples | settings |
|---|---|---|---|
| `sdof_eta60_n1500.npz` | oscillator | 1500 | η = u_y / max\|u_lin\| in 0.60–0.96 |
| `frame3_v3_database.npz`, `frame3_v3_arm_database.npz` | frame | 1000 generated, 970 pass the validity screen | target ductility μ in 0.6–2.0 |
| `tunnel_moving_train_database.npz` | tunnel | 1000 generated, the first 500 used (`N_MAX=500`) | bearing-capacity scale Q = 1.0 |
| `wt_v20_full.npz` | turbine | 1597 generated, 1248 with V_hub ≤ 20.2 m/s kept | soil-strength scale 1.0 |

Table 6 varies one setting per system over five levels, from L1 (mildest) to L5:
- oscillator: η ranges 0.70–0.98, 0.65–0.97, 0.60–0.96, 0.50–0.95 and 0.30–0.90 (1000 samples each);
- frame: upper limit of μ of 1.2, 1.6, 2.0, 2.7 and 3.4;
- tunnel: Q of 3.0, 1.8, 1.0, 0.6 and 0.35;
- turbine: soil-strength scale of 3.0, 1.8, 1.0, 0.6 and 0.35 (800 samples generated, about 625 kept after the V_hub cap).

## Reproducing the results

**Seeds and splits.** Every system is run with the seeds 42, 0, 1, 2 and 3. Each seed
sets both the random 70/15/15 train/validation/test split and the network initialisation.
For the oscillator, the Arm 2 script draws its own split, so its test samples differ from
those of Arms 1 and 3. The archived predictions in `database/` contain the split indices
of seed 42.

**Table 2 and Fig. 5.**
1. Copy the scripts of a system and its database into one folder.
2. Run that system's `release_webapp/0*/RUN.sh` there.
3. Run `python3 release_webapp/make_table2.py` to rebuild the table from the per-seed results.
4. Run `python3 figures/arms_table2.py` to draw Fig. 5.

`colab/table2_benchmark/benchmark_colab.ipynb` runs all 40 trainings on a Colab GPU.

**Hardware.** Results reproduce to within a hardware-dependent margin, not bit for bit:
- The oscillator and frame runs of Table 2 were trained on a CPU, as their logs show. The frame runner forces the CPU and uses a reduced configuration: window length and stride of 150 steps, hidden size 128, one recurrent layer, ten epochs.
- Re-running Arm 3 with the same scripts and seeds on an NVIDIA T4 gives mean R² of 0.945, 0.834, 0.862 and 0.852 for the oscillator, frame, turbine and tunnel. Table 2 reports 0.912, 0.836, 0.863 and 0.870.

**Metrics.** R² is pooled over channels and time for the oscillator and the frame. For
the tunnel and the turbine it is the unweighted mean over sensors, because the response
amplitude varies by two orders of magnitude along the tunnel.

**Post-processing.** The tunnel and turbine scripts (Arms 2 and 3) pass every predicted
history through `physics_lowpass` before scoring. This is a zero-phase three-band shelf
filter: unity gain below 1.1 Hz, gain 0.20 between 1.1 and 4.0 Hz, and gain 0.02 above
4.0 Hz. Arm 1 is not filtered. The oscillator and frame predictions are not filtered.

**Inputs.** For the frame, the tunnel and the turbine, the corrector also receives the
linear finite element response as an input channel, alongside the reduced-model response
and the excitation. The oscillator corrector receives two further static features: the
ratio η and the final drift of a cheap bilinear elastic–perfectly-plastic solver run on
the same excitation (`sdof_arm3_noepp_db.py`, printed as `net_drift` in the logs).

| result in the paper | how to reproduce it |
|---|---|
| Table 2, Fig. 5 | `release_webapp/0*/RUN.sh` or `colab/table2_benchmark`; `release_webapp/make_table2.py`; `figures/arms_table2.py` |
| MF-IA baseline (response to reviewers) | `colab/mfia` |
| Table 3, backbone comparison | `colab/backbone_comparison` |
| Table 4, computational cost | `colab/computational_cost` |
| Table 5, Fig. 6, component ablation | `colab/component_ablation`; `figures/ablation_table5.py` |
| Table 6, nonlinearity level | `regenerate_databases.sh levels`; `colab/nonlinearity_sensitivity` |
| context-length study | `colab/context_length`; `figures/context_table6.py` |
| Figs. 7–10, example predictions | archived seed-42 test predictions in `release_webapp/0*/database`, shown by the viewer |

The figure scripts plot the table values written in them. Trained network weights are not
distributed; the training scripts regenerate them.

## Numerical models

| system | model | solver |
|---|---|---|
| `01_sdof` | bilinear oscillator, Newmark integration | NumPy |
| `02_frame_v3` | two-bay three-storey frame, concentrated plastic hinges at the member ends | NumPy |
| `03_tunnel` | 200 m Euler–Bernoulli tube on an elastoplastic Winkler bed under a moving train | OpenSees (`openseespy`) |
| `04_turbine` | tapered tower and embedded monopile, rigid rotor–nacelle assembly, elastoplastic API-sand p–y springs, mudline plastic hinge, thrust from the wind speed relative to the moving hub | NumPy / SciPy |

## The viewer

`python3 webapp/mf_app.py` serves the archived predictions for the seed-42 test samples:
225 oscillator, 146 frame, 75 tunnel and 187 turbine records. For each sample it shows the
nonlinear finite element reference and one prediction per arm. It performs no inference
and needs no PyTorch. The metrics next to each plot are computed from the archived curves.
Use `PORT=8802 python3 webapp/mf_app.py` for another port. Because the oscillator's Arm 2
uses a different split, a sample held out from Arms 1 and 3 may be a training sample for
Arm 2; the split badge in the viewer says so.

Requirements: `requirements.txt` (viewer) and `release_webapp/REQUIREMENTS.txt` (training
and data generation).
