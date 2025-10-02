<!-- Improved compatibility of back to top link -->
<a name="readme-top"></a>

<h3 align="center">PREDICT-PaCa</h3>
<p align="center"><em>A calibration-aware ensemble for personalized neoadjuvant therapy and perioperative risk management in resectable pancreatic cancer</em></p>
<p align="center">
  <img src="./Figure%201.png" alt="PREDICT-PaCa architecture" width="85%">
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#repository-layout">Repository Layout</a></li>
        <li><a href="#next-steps">Next Steps</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**PREDICT-PaCa** is an end-to-end framework that unifies (i) preoperative risk stratification, (ii) individualized neoadjuvant therapy (NAT) policy learning, and (iii) counterfactual survival evaluation to support patient-specific decisions under real-world feasibility constraints.

- **PyTabKit-Net** — a *calibration-first* tabular learner combining teacher-ensemble distillation (OOF), sparse stacking, and temperature scaling to deliver reliable preoperative risk predictions (e.g., 1-year survival, major complications).
- **DRIFT-NAT** — a multi-arm policy learner that recommends both the optimal NAT arm *and* a feasible treatment window; evaluated with doubly-robust estimators and equipped with two-stage **uncertainty guardrails** for deployable decisions.
- **Neo-CARE** — a multi-arm DR-Learner counterfactual engine producing patient-level Δ-maps (survival gain/loss), HTE landscapes, and feasibility-aware decision panels with Gantt-style timing views.

### Repository Layout
- `PyTabKit_net/` — preoperative risk modeling (calibration-aware tabular ensemble; calibration curves, decision curves, SHAP).  
- `DRIFT_NAT/` — multi-arm NAT policy learning + feasible timing window; off-policy value estimation & uncertainty guardrails.  
- `Neo_CARE/` — counterfactual survival engine；patient-level Δ-maps、Pareto （ΔSurvival vs ΔMinDays）。  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Next Steps
- **External/temporal validation** across multi-institutional cohorts; assess transportability & calibration drift.
- **Policy value at scale** via doubly-robust estimators and Pareto analysis (ΔSurvival vs. ΔFeasibility/ΔMinDays).
- **Safety & calibration**: periodic recalibration, conformal safeguards, decision-time guardrails.
- **Workflow integration** with perioperative EHR to generate feasibility-aware patient-specific decision panels.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
- Python (3.10+)  
- Tabular ML: scikit-learn, XGBoost, LightGBM, CatBoost  
- Survival/causal: lifelines/pycox (optional), doubly-robust estimators, DR/DR-Learner toolkits  
- Explainability/plots: SHAP, Matplotlib  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
The framework expects (a) preoperative tabular features and (b) curated labels for survival and perioperative outcomes. Typical inputs include cohort splits such as `Chort_train.csv` / `Chort_valid.csv`, along with treatment indicators and timing variables for NAT policy learning.  


> **Environment notes**
> - Fix seeds across all libraries to ensure reproducible calibration.
> - Causal/policy modules may require additional dependencies (see `requirements-optional.txt`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
High-level pipelines and examples are provided per module:
- **PyTabKit_net/** — train & evaluate preoperative risk models; export calibration/decision curves and SHAP plots.  
- **DRIFT_NAT/** — learn individualized NAT arm + timing; export policy value estimates and uncertainty guardrail diagnostics.  
- **Neo_CARE/** — generate patient-level Δ-maps、feasibility-aware decision panels.

Outputs include calibration plots, decision curves, policy value estimates, counterfactual Δ-maps, and patient-level decision panels.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing
Contributions are welcome!
1. Fork the project  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License
Distributed under the MIT License. See `LICENSE` for details.  
*Research use only. Not intended for clinical decision-making.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact
Maintainers: Junqi Cui (TBD contact)  
Project Link: [https://github.com/JunqiCui1-1/PREDICT-PaCa](https://github.com/JunqiCui1-1/PREDICT-PaCa)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
