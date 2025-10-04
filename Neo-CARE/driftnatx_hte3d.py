#!/usr/bin/env python
import argparse, os
from driftnatx.hte3d import run_hte3d, HTE3DConfig

def parse_args():
    p = argparse.ArgumentParser(description="Step 7 — HTE (DR-learner) + 3D visualization.")
    p.add_argument("--input", required=True, help="Path to input CSV (e.g., Chort_total.csv).")
    p.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs).")
    p.add_argument("--config", default=None, help="Optional YAML (see examples/hte3d_config.yaml).")
    # quick overrides
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--ice-n", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = HTE3DConfig()
    if args.seed is not None:
        cfg.random_state = args.seed
    if args.grid_n is not None:
        cfg.grid_n = args.grid_n
    if args.ice_n is not None:
        cfg.ice_n = args.ice_n
    os.makedirs(args.output, exist_ok=True)
    run_hte3d(args.input, args.output, cfg, config_yaml=args.config)

if __name__ == "__main__":
    main()
