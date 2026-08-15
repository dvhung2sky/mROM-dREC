#!/bin/bash
# 3-story frame v3 (2D MRF, 2 bays, 15 SCWB hinges)
# run from the project root with these scripts copied there
set -e

N_SAMPLES=1000 MPLBACKEND=Agg python3 gen_3story_frame_v3.py
python3 consolidate_frame3_v3.py
for s in 42 0 1 2 3; do python3 run_frame3_seedsweep.py $s 3 10; done   # Arm 1 + Arm 3
for s in 42 0 1 2 3; do python3 run_frame3_seedsweep.py $s 2 10; done   # Arm 2
python3 summarize_seedsweep.py
