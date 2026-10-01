#!/bin/sh
# Turbine, V_hub-capped envelope: 1248 of the 1597 samples (V_hub <= 20.2 m/s), 5 seeds at
# 70/15/15, sign-flip on, overlapping windows (step 50), no per-sample loss weighting.
# Run from the project root with these scripts copied there and wt_v20_full.npz alongside
# (see numerical_model/RUN.sh or ../../regenerate_databases.sh).
# The table row is the unweighted mean of the 5 per-sensor R2 in each results CSV.
for S in 42 0 1 2 3; do
  SUF=$([ "$S" = 42 ] && echo "" || echo "_s$S")
  for A in 3 2; do
    DB_PATH=wt_v20_full.npz SEED=$S FLIP=1 STEP=50 TAG=wt_arm${A}_v20_full$SUF MPLBACKEND=Agg \
      python3 wt_arm${A}_v20.py
  done
done
