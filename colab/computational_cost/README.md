# colab_cost — measure the computational-cost table on a GPU

Runs the four **production Arm-3 scripts** unmodified (they already pick
`cuda if torch.cuda.is_available()`) and reports Parameters, Model MB, Training s,
Inference s, Activation MB, Peak GPU MB and Database MB for each example. Seed 42.

## Upload

Drag this folder into Drive so it lands at `MyDrive/colab_cost`, keeping `data/` and
`scripts/` intact:

```
MyDrive/colab_cost/
  cost_measure_colab.ipynb
  scripts/  sdof_arm3_noepp_db.py  frame3_v3_colab.ipynb
            red_to_nl_tunnel_half.py  wt_arm3_v20.py
  data/     sdof_eta60_n1500.npz  frame3_v3_database.npz
            tunnel_moving_train_database.npz  wt_v20_full.npz
  results/                                   <- created by the notebook
```

Edit the single `BASE = ...` line in cell 2 if you put it elsewhere.

## Run

Open in Colab, **Runtime -> Change runtime type -> T4 GPU**, Run all. Roughly 8-20 min.
Each example writes `results/<example>.json` and is skipped on re-entry, so a disconnect
costs at most one example. The final cell writes `results/cost_table_gpu.csv`.

## Configurations

Published configs, not script defaults:

| example | config |
|---|---|
| sdof | as shipped, seed 42, DB = sdof_eta60_n1500.npz |
| frame3 | notebook cells + seed-sweep override: L=150 STEP=150 HIDDEN=128 LAYERS=1 EPOCHS=10 |
| tunnel | N_MAX=0 -> N=1000, 700/150/150, seed 42 |
| turbine | v20 database, FLIP=1 STEP=50 SAMPLE_W=0, seed 42 |

The frame override is the one that produced every reported frame number; the notebook's own
HIDDEN=192/LAYERS=2/EPOCHS=40 defaults give the 1,115,943-parameter model, which produced
none of them.

## Note on the frame

`frame3_v3_colab.ipynb` cell 0 would otherwise either mount Drive or **regenerate** the
database with `gen_3story_frame_v3.py`. The driver patches four string literals to make it
read the staged `.npz` instead, and leaves the CUDA device selection alone.

Rebuild this folder with `python3 make_colab_cost.py` from the project root.
