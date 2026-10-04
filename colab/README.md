# Colab notebooks

Each folder holds one self-contained notebook. The notebooks expect the databases built by
`../regenerate_databases.sh`, uploaded to the Google Drive folder named in the folder's
README or in the notebook's `BASE = ...` line. Use a T4 GPU runtime. Each notebook is
resumable: every run writes its own JSON, and re-running the sweep cell skips runs already
on disk.

| folder | result | runs |
|---|---|---|
| `table2_benchmark` | Table 2: Arms 1–3, four systems, seeds 42, 0, 1, 2, 3, using the release scripts | 40 |
| `mfia` | MF-IA baseline (low-fidelity response as an input, no residual) paired with Arm 3 | 40 |
| `backbone_comparison` | Table 3: nine backbones | 36 |
| `computational_cost` | Table 4: parameters, memory, training and inference time | 4 |
| `component_ablation` | Table 5 and Fig. 6 | 80 |
| `nonlinearity_sensitivity` | Table 6: five nonlinearity levels per system | 200 |
| `context_length` | context length of the residual-feedback channel | 100 |
