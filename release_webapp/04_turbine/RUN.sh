#!/bin/sh
# Turbine, V_hub-capped envelope.  1248 of the 1597 samples (V_hub <= 20.2 m/s), seed 42,
# sign-flip on, overlapping windows (step 50), no per-sample loss weighting.
# The database is built by filtering wt_dsize_n1600.npz on params[:, param_names.index('V_hub')].
for A in 3 2; do
  DB_PATH=webapp/artifacts/turbine_v20/database/wt_v20_full.npz \
  SEED=42 FLIP=1 STEP=50 TAG=wt_arm${A}_v20_full MPLBACKEND=Agg \
  python3 webapp/artifacts/turbine_v20/arm${A}/wt_arm${A}_v20.py
done
