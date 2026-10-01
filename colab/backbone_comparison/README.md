# colab_arch — backbone comparison on Colab GPU

9 backbones x 4 examples, seed 42, 10 epochs, 100% of the 70% train split, metrics on the
held-out TEST split. 36 runs.

## Upload

Drag this whole folder into Google Drive so it lands at `MyDrive/colab_arch`, keeping the
`data/` subfolder intact:

```
MyDrive/colab_arch/
  arch_backbone_colab.ipynb
  data/  sdof_eta60_n1500.npz  frame3_v3_arm_database.npz
         tunnel_moving_train_database.npz  wt_v20_full.npz     (327 MB total)
  results/                                  <- created by the notebook
```

If you put it somewhere else, edit the single `BASE = ...` line in notebook cell 2.

## Run

Open the notebook in Colab, **Runtime -> Change runtime type -> T4 GPU**, then Run all.
Cell 1 stops you if no GPU is attached.

## If it does not finish in one session

The sweep is **resumable at per-run granularity**, which matters because Colab reclaims
GPUs and caps session length. After a disconnect:

1. re-run cells 1-4 (remount Drive, re-stage the databases locally, re-import the harness)
2. re-run cell 5

It skips every run whose JSON is already on disk and continues with the rest, so you lose at
most the single run that was in flight. Nothing else needs changing, and running the loop
again after it has finished is a no-op.

Safety details: JSON and `.pt` writes are atomic (`.tmp` then rename), and an existing JSON
that fails to parse is treated as unfinished and re-run -- a kill during a write cannot
leave a corrupt file that gets skipped forever. `results/progress.csv` is an append-only log
of what completed when.

Outputs land in `results/`: 36 JSONs, 36 `.pt` checkpoints (`SAVE_CKPT = True`),
`progress.csv`, `arch_table_pooled.csv` and `arch_table_unweighted.csv`. Each checkpoint
carries its own split indices and target scaling, so any row can be re-scored later without
retraining.

## What is in the databases

| example | file | N | T x dt | sensors | inputs |
|---|---|--:|---|--:|--:|
| sdof | sdof_eta60_n1500.npz | 1500 | 500 x 0.01 s | 1 | 5 |
| frame3 | frame3_v3_arm_database.npz | 970 | 2000 x 0.01 s | 3 | 33 |
| tunnel | tunnel_moving_train_database.npz | 1000 | 1001 x 0.02 s | 11 | 12 |
| turbine | wt_v20_full.npz | 1248 | 600 x 0.10 s | 5 | 17 |

These are the release databases copied unmodified, so hashes match `release_webapp/`.

## Provenance

The harness is a GPU port of `arch_compare.py`; every deliberate difference is listed in
the harness docstring and in the notebook's closing markdown cell. Rebuild this folder with
`python3 make_colab_arch.py` from the project root.
