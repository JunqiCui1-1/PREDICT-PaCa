#!/usr/bin/env python
import argparse, os
from driftnatx.explain import run_explainability, ExplainConfig

def parse_args():
    p = argparse.ArgumentParser(description="Step 8 — Explainability & Clinical Usability for DRIFT-NAT-X.")
    p.add_argument("--input", required=True, help="Path to input CSV (e.g., Chort_total.csv).")
    p.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs).")
    p.add_argument("--config", default=None, help="Optional YAML (see examples/explain_config.yaml).")
    # quick overrides
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cards", type=int, default=None, help="Number of patient cards (default: 20).")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = ExplainConfig()
    if args.seed is not None:
        cfg.random_state = args.seed
    if args.cards is not None:
        cfg.n_patient_cards = args.cards
    os.makedirs(args.output, exist_ok=True)
    run_explainability(args.input, args.output, cfg, config_yaml=args.config)

if __name__ == "__main__":
    main()
