# colab_ablation — component ablation (Table 5) on Colab GPU

4 examples x 4 variants x 5 seeds = **80 runs**, 10 epochs each, metrics on the held-out
TEST split.

| variant | what is removed |
|---|---|
| `full` | nothing — the full framework |
| `nofilm` | FiLM conditioning; the static parameters no longer modulate the backbone |
| `noskip` | the load-skip branch is not built at all |
| `nozeroinit` | the prediction head is not zero-initialised |

Arm 1 (`u_red`, no learning) is recorded on every run, so the reference column comes from
the same splits as the trained columns.

## Upload

Drag this folder into Google Drive so it lands at `MyDrive/colab_ablation`, keeping
`data/` intact:

```
MyDrive/colab_ablation/
  ablation_colab.ipynb
  data/  sdof_eta60_n1500.npz  frame3_v3_arm_database.npz
         tunnel_moving_train_database.npz  wt_v20_full.npz     (327 MB)
  results/                                  <- created by the notebook
```

Somewhere else is fine — edit the single `BASE = ...` line in cell 2.

## Run

Open in Colab, **Runtime -> Change runtime type -> T4 GPU**, then Run all. Cell 1 stops
you if no GPU is attached.

**Expect several hours, and expect not to finish in one session.** That is planned for.

## Resuming

The sweep is resumable at per-run granularity, which matters because Colab reclaims GPUs
and caps session length. After a disconnect:

1. re-run cells 1-4 (remount Drive, re-stage the databases locally, re-import the harness)
2. re-run cell 5

It skips every run whose JSON is already on disk and continues with the rest, so you lose
at most the single run that was in flight. Running the cell again after it has finished is
a no-op.

Verified behaviour of the skip test:

| state on disk | decision |
|---|---|
| no JSON | run |
| JSON with `R2` and `sec` | skip |
| JSON that parses but lacks the metric | re-run |
| truncated / unparseable JSON | re-run, with a warning |

JSON and `.pt` writes are atomic (`.tmp` then rename), so a kill during a write cannot
leave a corrupt file that gets skipped forever. `results/progress.csv` is an append-only
log of what completed when.

## Outputs

In `results/`: 80 JSONs, 80 `.pt` checkpoints, `progress.csv`, and
`table5_pooled.csv` + `table5_unweighted.csv`.

Use **pooled** for SDOF and the frame, **unweighted** for the tunnel and the turbine, as in
`release_webapp/` — those sensors span decades of energy and pooling hides the quiet ones.
Cells with fewer than five finished seeds print as `(n/5 seeds)` rather than being averaged
over a partial set.

Each checkpoint stores its own seed, split indices, target scaling and the three ablation
flags, so any cell of the table can be re-scored later without retraining.

## Provenance

The harness is lifted **verbatim** from `colab_arch/arch_backbone_colab.ipynb`, which
already carries the `no_film` / `no_skip` / `no_zeroinit` switches, and patched in exactly
one respect: `SEED` becomes a per-run argument threaded through `prep()` and `run_one()`.
Ten patches, each asserted to apply exactly once, so a change upstream fails the build
rather than silently emitting a harness whose `seed` argument does nothing.

Window construction, normalisation, the free-running overlap-add rollout and the metrics
are therefore identical to the backbone sweep, which is what makes the two tables
comparable. The databases are the `release_webapp/` files copied unmodified, so hashes
match.

Rebuild with `python3 make_colab_ablation.py` from the project root.

## One caveat for the paper

A seed changes the split **and** the initialisation together, as in the release runs. The
+/- therefore mixes split luck with initialisation sensitivity and is not a pure measure of
either. If the claim is about a component rather than about spread, quote the paired
difference between a variant and `full` at matched seed — the per-seed JSONs support that
directly.
