#!/bin/bash
# SDOF (eta 0.60-0.96, N=1500)
# run from the project root with these scripts copied there
set -e

N_SAMPLES=1500 ETA_LO=0.60 ETA_HI=0.96 OUT_DB=sdof_eta60_n1500.npz python3 gen_sdof_eta.py
for s in 42 0 1 2 3; do   # Arm 1 + Arm 3, 15 epochs
  DB=sdof_eta60_n1500.npz SEED=$s MPLBACKEND=Agg python3 sdof_arm3_noepp_db.py
done
for s in 42 0 1 2 3; do   # Arm 2, nofb, 15 epochs
  DB=sdof_eta60_n1500.npz SEED=$s ARM2_FEEDBACK=0 MPLBACKEND=Agg python3 arm2_split15_db.py SDOF
done
