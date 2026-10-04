# MF-IA baseline (input-augmented multi-fidelity)

In MF-IA the low-fidelity response is an input and the network predicts the high-fidelity
response directly (Guo et al., 2022; Conti et al., 2023). The notebook runs the Arm 3 script
with `MFIA=1`, which keeps the same input channels (low-fidelity response included),
backbone, loss, epochs, seeds (42, 0, 1, 2, 3) and splits. The target becomes the
high-fidelity response, and the low-fidelity solution is not added back. The notebook then
re-runs Arm 3 on the same seeds, so the two can be paired. It uses the same Drive layout as
`../table2_benchmark`.

| file | contents |
|---|---|
| `benchmark_mfia_colab.ipynb` | the notebook; it patches the staged Arm 3 scripts with the `MFIA` switch and asserts each patch |
| `mfia_per_seed.csv` | every seed: Arm 1, Arm 3 (T4 re-run), MF-IA |
| `make_table2_mfia.py` | `python3 make_table2_mfia.py mfia_per_seed.csv` rebuilds the table and the figure |

Paired MF-IA − Arm 3 on the T4, n = 5 seeds each, same splits:

| system | MF-IA − Arm 3 | seeds with MF-IA higher |
|---|---|---|
| oscillator | −0.012 ± 0.026 | 1/5 |
| frame | −0.032 ± 0.015 | 0/5 |
| turbine | −0.113 ± 0.034 | 0/5 |
| tunnel | −0.118 ± 0.063 | 0/5 |

The tunnel and turbine MF-IA predictions pass through the same `physics_lowpass` filter as
Arms 2 and 3.
