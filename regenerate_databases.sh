#!/bin/bash
# Regenerate the high-fidelity databases from the finite element models in release_webapp/0*/numerical_model.
# The databases are not distributed (two exceed GitHub's 100 MB file limit); every generator uses a fixed
# random seed, so this script rebuilds the data sets behind the published results.
#
#   bash regenerate_databases.sh            # the four databases of Table 2 and of the Colab notebooks  -> data/
#   bash regenerate_databases.sh levels     # also the 20 databases of the nonlinearity study (Table 6) -> data/levels/
#
# Needs numpy, scipy and, for the tunnel only, openseespy. Runs on a CPU; the turbine and tunnel take the longest.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; M="$ROOT/release_webapp"; OUT="$ROOT/data"
export MPLBACKEND=Agg
mkdir -p "$OUT"
work() { rm -rf "$OUT/_work/$1"; mkdir -p "$OUT/_work/$1"; cd "$OUT/_work/$1"; }

sdof()    { N_SAMPLES=${N:-1500} ETA_LO=$1 ETA_HI=$2 OUT_DB=db.npz python3 "$M/01_sdof/numerical_model/gen_sdof_eta.py"; }
frame()   { N_SAMPLES=1000 MU_HI=$1 python3 "$M/02_frame_v3/numerical_model/gen_3story_frame_v3.py"
            python3 "$M/02_frame_v3/numerical_model/consolidate_frame3_v3.py"; }   # -> frame3_v3_database.npz + frame3_v3_arm_database.npz
tunnel()  { N_SAMPLES=1000 QSCALE=$1 OUT_DB=db.npz python3 "$M/03_tunnel/numerical_model/generate_tunnel_moving_train_database.py"; }
turbine() { mkdir -p wt_shards
            for K in 0 1 2 3 4 5 6 7; do
              FY_SOIL_SCALE=$2 python3 "$M/04_turbine/numerical_model/gen_wind_turbine_database.py" $1 --shard $K/8 --out wt_shards/wt_$K.npz &
            done; wait
            PYTHONPATH="$M/04_turbine/numerical_model" OUT_DB=raw.npz python3 "$M/04_turbine/numerical_model/consolidate_wt.py"
            python3 "$M/04_turbine/numerical_model/cap_vhub.py" raw.npz db.npz; }   # keeps V_hub <= 20.2 m/s

# ---- Table 2 databases (the names the training scripts and notebooks expect) ----
work sdof;    sdof 0.60 0.96;  mv db.npz "$OUT/sdof_eta60_n1500.npz"
work frame3;  frame 2.0;       mv frame3_v3_database.npz frame3_v3_arm_database.npz "$OUT/"
work tunnel;  tunnel 1.0;      mv db.npz "$OUT/tunnel_moving_train_database.npz"
work turbine; turbine 1600 1.0; mv db.npz "$OUT/wt_v20_full.npz"          # 1597 generated, 1248 kept
cd "$OUT"

# ---- Table 6: five nonlinearity levels per system, L1 mildest ... L5 most nonlinear ----
if [ "$1" = "levels" ]; then
  L=1; for e in "0.70 0.98" "0.65 0.97" "0.60 0.96" "0.50 0.95" "0.30 0.90"; do
    work sdof; N=1000 sdof $e; mkdir -p "$OUT/levels/sdof"; mv db.npz "$OUT/levels/sdof/L$L.npz"; L=$((L+1)); done
  L=1; for mu in 1.2 1.6 2.0 2.7 3.4; do
    work frame3; frame $mu; mkdir -p "$OUT/levels/frame3"; mv frame3_v3_arm_database.npz "$OUT/levels/frame3/L$L.npz"; L=$((L+1)); done
  L=1; for q in 3.0 1.8 1.0 0.6 0.35; do
    work tunnel; tunnel $q; mkdir -p "$OUT/levels/tunnel"; mv db.npz "$OUT/levels/tunnel/L$L.npz"; L=$((L+1)); done
  L=1; for fy in 3.0 1.8 1.0 0.6 0.35; do
    work turbine; turbine 800 $fy; mkdir -p "$OUT/levels/turbine"; mv db.npz "$OUT/levels/turbine/L$L.npz"; L=$((L+1)); done
fi
rm -rf "$OUT/_work"
ls -la "$OUT"
