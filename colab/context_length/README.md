# colab_context — autoregressive context length (Tables 7 and 8)

4 examples x 5 context lengths `L in {0, 12, 25, 50, 100}` x 5 seeds = **100 runs**,
10 epochs each, metrics on the held-out TEST split.

`L` is how many of the 100 context steps of the autoregressive channel the model may see;
the rest are zeroed, in both the training windows and the free-running rollout. `L = 100`
(= `L_IN`) is the full context; `L = 0` removes the channel's content entirely. The channel
itself stays in place either way, so the input width is identical across all five columns
and only the information changes.

Both tables come from this one sweep:

| table | what it shows |
|---|---|
| **7** | absolute R², mean ± s.d. over the five seeds |
| **8** | paired change against `L = 100` **on the same seed**, so split luck cancels |

## Upload

Drag this folder into Google Drive so it lands at `MyDrive/colab_context`, keeping `data/`
intact:

```
MyDrive/colab_context/
  context_colab.ipynb
  data/  sdof_eta60_n1500.npz  frame3_v3_arm_database.npz
         tunnel_moving_train_database.npz  wt_v20_full.npz     (327 MB)
  results/                                  <- created by the notebook
```

Somewhere else is fine — edit the single `BASE = ...` line in cell 2.

## Run

Open in Colab, **Runtime -> Change runtime type -> T4 GPU**, then Run all. Cell 1 stops you
if no GPU is attached.

Rough cost: the 80-run ablation sweep took 20 min on a T4, so expect **25-35 min** here.
The resume machinery is in place regardless, because Colab reclaims GPUs without warning.

## Resuming

Resumable at per-run granularity. After a disconnect:

1. re-run cells 1-4 (remount Drive, re-stage the databases locally, re-import the harness)
2. re-run cell 5

It skips every run whose JSON is already on disk, so you lose at most the run that was in
flight. Re-running after completion is a no-op.

| state on disk | decision |
|---|---|
| no JSON | run |
| JSON with `R2` and `sec` | skip |
| JSON that parses but lacks the metric | re-run |
| truncated / unparseable JSON | re-run, with a warning |

JSON writes are atomic (`.tmp` then rename), so a kill mid-write cannot leave a corrupt file
that gets skipped forever. `results/progress.csv` is an append-only log.

## Outputs

In `results/`: 100 JSONs, `progress.csv`, `table7.csv` and `table8.csv`.

`SAVE_CKPT` is **off** by default — 100 checkpoints is a lot of Drive and the JSONs carry
every metric. Set it to `True` in cell 4 if you want the weights.

Metric per row follows the benchmark: pooled for SDOF and the frame, unweighted mean over
sensors for the tunnel and the turbine.

## Reading the two tables together

Table 7's ± is dominated by seed spread and will often exceed the gaps between columns —
that is expected, and it is exactly why Table 8 exists. Pairing against `L = 100` on the
same seed removes the split, leaving the effect of context length alone. **Quote Table 8 for
any claim about context length**; treat Table 7 as the level context. Table 8 marks
`+` p<0.10 and `*` p<0.05 from a paired t-test on 4 d.o.f.

Table 8 rows are ordered by the magnitude of the effect at `L = 0`, matching the published
layout.

## Provenance

The harness is lifted **verbatim** from `colab_arch/arch_backbone_colab.ipynb` and patched
only to thread a per-run seed through `prep()` and `run_one()` — ten patches, each asserted
to apply exactly once, shared with `make_colab_ablation.py`, which this builder imports.
`ar_len` is an upstream feature, not something added here.

Verified before shipping: `ar_len` in {0, 12, 25, 50, 100, None} maps to `AR_LEN`
{0, 12, 25, 50, 100, 100} with the channel count unchanged at every setting, and end-to-end
runs complete at both extremes on two seeds.

Databases are the `release_webapp/` files copied unmodified, so hashes match.

Rebuild with `python3 make_colab_context.py` from the project root.
