##Causal HTE + 3D Visualization

**Goal.** Estimate heterogeneous treatment effects (HTE) across timing arms `<4w`, `4–6w`, `>6w` for NAT==1 using a **pairwise DR-learner** (doubly-robust pseudo-outcomes) with GBM models, then visualize:
- 3D **τ** surfaces (e.g., `<4w` vs `>6w`),
- per-arm risk surfaces `m_s(X)`,
- patient-level 3D scatter for τ,
- τ histograms and ICE-like curves.

**Run:**
```bash
python scripts/driftnatx_hte3d.py \
  --input /path/to/Chort_total.csv \
  --output ./outputs \
  --config examples/hte3d_config.yaml
