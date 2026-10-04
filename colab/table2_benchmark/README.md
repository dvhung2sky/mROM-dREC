# Table 2 on Colab

`benchmark_colab.ipynb` trains Arms 2 and 3 for the four systems with seeds 42, 0, 1, 2
and 3 (40 runs), using the release scripts unchanged except for the patches the notebook
lists and asserts. Upload to `MyDrive/colab_benchmark/`:

```
data/     sdof_eta60_n1500.npz  frame3_v3_database.npz  tunnel_moving_train_database.npz  wt_v20_full.npz
scripts/  sdof_arm3.py   = release_webapp/01_sdof/arm3/sdof_arm3_noepp_db.py
          sdof_arm2.py   = release_webapp/01_sdof/arm2/arm2_split15_db.py
          frame_run.py   = release_webapp/02_frame_v3/arm3/run_frame3_seedsweep.py without the CPU override (lines 40-43)
          frame3_v3_colab.ipynb = release_webapp/02_frame_v3/arm3/frame3_v3_colab.ipynb
          tunnel_arm3.py = release_webapp/03_tunnel/arm3/red_to_nl_tunnel_half.py
          tunnel_arm2.py = release_webapp/03_tunnel/arm2/red_to_nl_tunnel_arm2_half.py
          turbine_arm3.py = release_webapp/04_turbine/arm3/wt_arm3_v20.py
          turbine_arm2.py = release_webapp/04_turbine/arm2/wt_arm2_v20.py
```

The data files are produced by `../../regenerate_databases.sh`. On a T4, Arm 3 gives mean R²
of 0.945, 0.834, 0.862 and 0.852 for the oscillator, frame, turbine and tunnel. Table 2
reports 0.912, 0.836, 0.863 and 0.870, so the reproduction holds within a hardware margin,
not bit for bit. The frame runs of Table 2 were made on a CPU.
