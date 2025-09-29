---
number: 1
title: "QuSOP: a quantum learning–optimization framework for robust dose selection in early clinical development"
topic: "Quantum Machine Learning for Pharmaceutical Problems"

team_leads:
  - "Louis Chen (Jij Europe) <louis.chen@j-ij.com>"
  - "Ross Grassie (Jij Inc.) <ross.grassie@j-ij.com>"
  - "Lily Lee (Imperial College London) <l.lee23@imperial.ac.uk>"

contributors:
  - "Felix Burt (Imperial College London) <f.burt23@imperial.ac.uk>"
  - "Hasan Almatrouk (Imperial College London) <hasan.almatrouk23@imperial.ac.uk>"

# Optional fields; set to null or remove if unused
github: null
youtube_video: null

abstract: |-
  QuSOP is a hardware-agnostic, regulator-aligned framework that pairs a quantum LSTM (QLSTM) predictor with quantum Monte Carlo (QMC) to enable uncertainty-aware, chance-constrained dose selection in PK/PD. QLSTM learns 24-h and 168-h biomarker trajectories from sparse data; QMC propagates covariate priors to estimate attainment p(d)=Pr[max_t PD(t)≤τ] across dose grids. The minimal dose is chosen to meet preset success thresholds. On a Phase-1-like dataset, QuSOP yields calibrated dose–success curves and consistent once-daily/weekly recommendations, with lower doses under relaxed criteria (e.g., 90%→75%). The framework supports covariate-shift stress tests and runs on CPUs/GPUs while remaining QPU-ready.
  References:
---
