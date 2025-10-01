# PREDICT-PaCa

![Figure 1](./Figure%201.png)

# PREDICT-PaCa
*Calibration-aware ensemble for personalized neoadjuvant therapy and perioperative risk management in resectable pancreatic cancer*

<p align="center">
  <img src="./Figure%201.png" alt="PREDICT-PaCa architecture" width="85%">
</p>

## Overview
**PREDICT-PaCa** is an end-to-end framework that unifies (i) preoperative risk stratification, (ii) personalized neoadjuvant therapy (NAT) policy learning, and (iii) counterfactual survival evaluation to support patient-specific decisions under feasibility constraints.

- **PyTabKit-Net** — a calibration-first tabular learner that combines teacher-ensemble distillation (OOF), sparse stacking, and temperature scaling to deliver reliable preoperative risk predictions (e.g., 1-year survival, complications).
- **DRIFT-NAT** — a multi-arm policy learning module for recommending the optimal NAT arm and *feasible* treatment window; evaluated with doubly robust estimators and equipped with two-stage **uncertainty guardrails** for deployable decisions.
- **Neo-CARE** — a multi-arm DR-Learner counterfactual engine that produces patient-level Δ-maps (survival gain/loss), heterogeneous treatment effect (HTE) landscapes, and feasibility-aware decision panels with Gantt-style timing views.

### What this repository enables
- Train a calibration-aware baseline for preoperative risk.
- Learn individualized NAT policies with feasibility-aware timing.
- Generate patient-level decision reports (counterfactual surfaces, Δ-maps, and local explanations).
- Visualize global/individual HTE patterns and strategy net-benefit curves.

---

## Quick Start
> Minimal skeleton—adapt paths and script names to your setup.

```bash
# 1) Clone
git clone https://github.com/<your-org-or-name>/PREDICT-PaCa.git
cd PREDICT-PaCa

# 2) Environment (Python 3.10+ recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip

# 3) Dependencies (example)
pip install scikit-learn==1.4.0 xgboost==2.0.0 lightgbm==4.0.0 catboost==1.2 \
           shap==0.45 lifelines==0.27 matplotlib==3.8 pandas==2.2

# 4) Run notebooks / scripts
jupyter lab
# Open notebooks in `notebooks/` or run scripts in `scripts/`
