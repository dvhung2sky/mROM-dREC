#!/bin/sh
# Tunnel, moving-train.  5 seeds at N=500 (350/75/75), TEST split reported.
# The unweighted row of the table is the mean over the 11 per-sensor R2 in each results CSV.
for S in 42 0 1 2 3; do
  N_MAX=500 SEED=$S TAG=tunnel_half_arm3_s$S MPLBACKEND=Agg python3 red_to_nl_tunnel_half.py
  N_MAX=500 SEED=$S TAG=tunnel_half_arm2_s$S MPLBACKEND=Agg python3 red_to_nl_tunnel_arm2_half.py
done
# full-data anchor (seed 42, 700/150/150):
# TAG=red_to_nl_tunnel python3 red_to_nl_tunnel.py
