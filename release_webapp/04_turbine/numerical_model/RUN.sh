#!/bin/sh
# Wind turbine on a monopile: generate the high-fidelity database used in the paper.
#
#   1. gen_wind_turbine_database.py  the finite element model itself. Tapered tower and
#      embedded monopile in Euler-Bernoulli elements, rigid rotor-nacelle assembly,
#      elastoplastic API-sand p-y springs, a plastic hinge at the mudline, and thrust
#      formed from the wind speed relative to the moving hub. Sharded for parallelism.
#   2. consolidate_wt.py             merges the shards into the (N, S, T) layout with the
#      q_sensor load channel that the Arm 2 and Arm 3 scripts expect.
#   3. cap_vhub.py                   keeps the V_hub <= 20.2 m/s operating envelope.
#
# 1597 samples over 8 shards, roughly 3 s per sample per core.
set -e
cd "$(dirname "$0")"

rm -rf wt_shards && mkdir -p wt_shards
for K in 0 1 2 3 4 5 6 7; do
  python3 gen_wind_turbine_database.py 1600 --shard $K/8 --out wt_shards/wt_$K.npz &
done
wait

OUT_DB=wind_turbine_database.npz python3 consolidate_wt.py
python3 cap_vhub.py wind_turbine_database.npz wt_v20_full.npz
echo "wt_v20_full.npz is the database the published turbine runs use."
