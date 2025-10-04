#!/usr/bin/env python
import argparse, os
from driftnatx.config import Config
from driftnatx.pipeline import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Run DRIFT-NAT-X pipeline.")
    p.add_argument("--input", required=True, help="Path to input CSV (e.g., Chort_total.csv).")
    p.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs).")
    p.add_argument("--config", default=None, help="Optional YAML config (see examples/example_config.yaml).")
    p.add_argument("--seed", type=int, default=None, help="Random seed override.")
    p.add_argument("--boot", type=int, default=None, help="Bootstrap iterations for CIs.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    if args.seed is not None:
        cfg.random_state = args.seed
    if args.boot is not None:
        cfg.boot_b = args.boot
    os.makedirs(args.output, exist_ok=True)
    run_pipeline(args.input, args.output, cfg, config_yaml=args.config)

if __name__ == "__main__":
    main()
