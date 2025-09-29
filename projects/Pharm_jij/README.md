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
We introduce \textbf{QuSOP}, a hardware-agnostic, regulator-aligned framework that couples a \emph{quantum long short–term memory} predictor (QLSTM) with \emph{quantum Monte Carlo} (QMC) to deliver uncertainty-aware, chance-constrained dose selection for PK/PD decision-making. QLSTM learns steady-state biomarker trajectories over 24\,h and 168\,h windows from sparse, irregular data, while QMC propagates covariate priors (e.g., body weight, concomitant medication) to compute attainment probabilities \(p(d)=\Pr[\max_t \mathrm{PD}(t)\le \tau]\) on discrete dose grids; the minimal dose is then chosen to satisfy prespecified success levels (e.g., \(\ge 90\%\) or \(\ge 75\%\)). Quantum resources are confined to training/estimation subroutines (e.g., variational circuits, amplitude-estimation–style samplers); inference is deterministic on CPU/GPU with full audit trails. On a Phase~1–like dataset, the QLSTM+QMC pipeline yields well-calibrated sigmoidal dose–success curves and internally consistent once-daily/once-weekly recommendations; relaxing the attainment criterion from 90\% to 75\% reduces the mean minimal doses by \(\sim\)25\% for once-daily (19.0\(\rightarrow\)14.3\,mg) and \(\sim\)13\% for once-weekly (116.9\(\rightarrow\)101.3\,mg), with analogous trends when concomitant medication is excluded. QuSOP supports covariate-shift stress tests and multi-endpoint extensions, and is immediately deployable in sovereign GPU environments while remaining compatible with emerging QPUs. By combining QLSTM’s sample-efficient temporal modeling with QMC’s calibrated uncertainty propagation, QuSOP provides a pragmatic pathway to quantum-enhanced, transparent, and clinically relevant dose policies in early development.

References:
