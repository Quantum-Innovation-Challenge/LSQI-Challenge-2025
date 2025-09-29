---
number: 1 # leave as-is, maintainers will adjust
title: QUSOP: A QUANTUM LEARNING–OPTIMIZATION FRAME- WORK FOR ROBUST DOSE SELECTION IN EARLY CLINICAL DEVELOPMENT
topic: Quantum Machine Learning for Pharmaceutical Problems
team_leads:
  - Louis Chen (Jij Europe) louis.chen@j-ij.com
  - Ross Grassie (Jij Inc.) ross.grassie@j-ij.com
  - Lily Lee (Imperial College London) l.lee23@imperial.ac.uk

# Comment these lines by prepending the pound symbol (#) to each line to hide these elements

contributors:

- Felix Burt (Imperial College London) f.burt23@imperial.ac.uk
- Hasan Almatrouk (Imperial College London) hasan.almatrouk23@imperial.ac.uk

# github: louis-chen-jij/LSQI-Challenge-2025

# youtube_video: <your-video-id>

---

Project 1 description

QuSOP is a hardware-agnostic, regulator-aligned framework that integrates a **quantum long short–term memory (QLSTM)** predictor with **quantum Monte Carlo (QMC)** methods to enable uncertainty-aware, chance-constrained dose selection for PK/PD decision-making. The QLSTM component learns steady-state biomarker trajectories over 24-hour and 168-hour windows from sparse and irregular data, while QMC propagates covariate priors (such as body weight or concomitant medications) to compute attainment probabilities p(d) = Pr[max_t PD(t) ≤ τ] across discrete dose grids. The minimal effective dose is then chosen to satisfy predefined success thresholds (e.g., ≥90% or ≥75%). Quantum resources are confined to training and estimation subroutines (e.g., variational circuits and amplitude-estimation–style samplers), while inference remains deterministic on CPU/GPU with complete audit trails.

On a Phase 1–like dataset, the combined QLSTM+QMC pipeline produces well-calibrated sigmoidal dose–success curves and consistent once-daily and once-weekly dose recommendations. Relaxing the attainment criterion from 90% to 75% reduces mean minimal doses by approximately 25% for once-daily regimens (19.0 → 14.3 mg) and 13% for once-weekly regimens (116.9 → 101.3 mg), with similar trends observed when concomitant medication effects are excluded. Beyond dose optimization, QuSOP supports covariate-shift stress testing and multi-endpoint extensions, and it is immediately deployable in sovereign GPU environments while maintaining compatibility with emerging quantum processing units (QPUs). By combining QLSTM’s sample-efficient temporal modeling with QMC’s calibrated uncertainty propagation, QuSOP provides a pragmatic, transparent, and clinically relevant pathway to quantum-enhanced dose policy development in early drug research.

References:
