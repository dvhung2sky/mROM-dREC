# Nonlinearity level (Table 6)

4 systems × 5 levels × 2 trained arms × 5 seeds = 200 runs. Every run also records Arm 1
on its own test split. L1 is the mildest level and L5 the most nonlinear. Build the 20
databases with `bash ../../regenerate_databases.sh levels`, which writes
`data/levels/<system>/L1.npz` … `L5.npz`. Upload the contents of `data/levels/` to `MyDrive/colab_sensitivity/data/`,
or edit the `BASE` line in the notebook. Each system varies one setting:

| system | setting | L1 → L5 |
|---|---|---|
| oscillator | η range (yield displacement / linear peak) | 0.70–0.98, 0.65–0.97, 0.60–0.96, 0.50–0.95, 0.30–0.90 |
| frame | upper limit of target ductility μ | 1.2, 1.6, 2.0, 2.7, 3.4 |
| tunnel | soil bearing-capacity scale Q | 3.0, 1.8, 1.0, 0.6, 0.35 |
| turbine | API-sand strength scale | 3.0, 1.8, 1.0, 0.6, 0.35 |

`tunnel_sensor1_colab.ipynb` needs no training. It reads the per-sensor R² stored by the sweep and reports the tunnel row
without sensor 1, which sits at the tube end where the response is two orders of magnitude smaller.
