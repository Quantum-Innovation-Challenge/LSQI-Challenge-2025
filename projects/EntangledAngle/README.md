---

number: 1 # leave as-is, maintainers will adjust
title: Quantum Circuit Simulation of Nonlinear Population Pharmacokinetics Using Variational Algorithms
topic: Quantum Circuit Simulation of Compartmental Drug Dynamics by Leveraging Variational Algorithms for Nonlinear Mixed Effects Population Pharmacokinetics

Team Leads:
  - Dr. Reena Monica (Vellore Institute of Technology, Chennai) @reenamonica-design
  - Isshaan Singh (Vellore Institute of Technology, Chennai) @IsshaanSingh2701
  - Nandan Patel (Vellore Institute of Technology, Chennai) @NandanPatel24

Contributors:

  - Dr. Reena Monica (Vellore Institute of Technology, Chennai) @reenamonica-design
  - Isshaan Singh (Vellore Institute of Technology, Chennai) @IsshaanSingh2701
  - Nandan Patel (Vellore Institute of Technology, Chennai) @NandanPatel24

# github: "NandanPatel24/LSQI-Challenge-2025"

---

Abstract

Population pharmacokinetic/pharmacodynamic modeling traditionally employs
classical ordinary differential equations for drug dynamics simulation. This work
reformulates compartmental models as open quantum systems governed by Hamiltonian 
dynamics, achieving superior statistical performance. Four pharmacolog-
ical compartments (central, peripheral, effect-site, response) are encoded using
twelve qubits, evolved through PennyLane-implemented quantum circuits where
inter-compartmental transfers manifest as quantum jump operators via controlled
rotation gates. Applied to Phase 1 clinical data, quantum-enhanced stochastic approximation 
expectation-maximization achieves log-likelihood of −1366.81 versus
classical −8403.60, representing sixfold improvement in model fit. Parameter estimates 
converge identically (CL = 2.0 L/h, V1 = 10.0 L, V2 = 20.0 L, Imax = 0.8,
IC50 = 2.0 ng/mL), validating quantum accuracy while demonstrating enhanced
residual error modeling. SAEM optimization converges 42% faster (26 versus 44.74
minutes), though PennyLane API overhead increases total runtime 53% across 242
million circuit evaluations. Dose optimization recommends 20.0 mg daily or 15
mg weekly for standard populations targeting 90% biomarker suppression, with
quantum methods showing greater sensitivity to population heterogeneity (25-33%
dose reductions in specific scenarios). The framework successfully simulates 28,488
subjects with 100% stability, establishing quantum computing as viable for population 
pharmacometrics despite current computational overhead requiring future
optimization.

The link to the code file is: [EntangledAngle_Code_Submission](https://github.com/NandanPatel24/LSQI-Challenge-2025/blob/main/projects/EntangledAngle/Quantum_Innovation_Challenge_EntangledAngle_Code_submission.ipynb)

The link to the report file is: [EntangledAngle_Report_File](https://github.com/NandanPatel24/LSQI-Challenge-2025/blob/main/projects/EntangledAngle/Quantum_Innovation_Challenge_EntangledAngle_Report_submission.pdf.pdf)

References

[1] Gibiansky, L., & Gibiansky, E. (2009). Target-mediated drug disposition model: re-
lationships with indirect response models and application to population PK–PD
analysis. Journal of Pharmacokinetics and Pharmacodynamics, 36 (4), 341–351.
https://doi.org/10.1007/S10928-009-9125-9

[2] Mody, H., Kowthavarapu, V. K., & Betts, A. (2025). Recent Advances in PK/PD
and Quantitative Systems Pharmacology (QSP) Models for Biopharmaceuticals.
307–343. https://doi.org/10.1201/9781003300311-12

[3] Danhof, M., van der Graaf, P. H., Jonker, D. M., Visser, S. A. G., & Zuideveld, K.
P. (2007). 5.38 – Mechanism-Based Pharmacokinetic–Pharmacodynamic Modeling
for the Prediction of In Vivo Drug Concentration–Effect Relationships – Applica-
tion in Drug Candidate Selection and Lead Optimization. Elsevier, Vol. 5, 885–908.
https://doi.org/10.1016/B0-08-045044-X/00154-1

[4] Colburn, W. A. (1981). Simultaneous pharmacokinetic and pharmacodynamic
modeling. Journal of Pharmacokinetics and Biopharmaceutics, 9 (3), 367–388.
https://doi.org/10.1007/BF01059272

[5] Saha, G. (2024). Computer Simulations in Pharmacokinetics and Pharmacodynam-
ics. 392–425. https://doi.org/10.69613/x2158555

[6] Gherman, I., Abdallah, Z. S., Pang, W.-Z., Gorochowski, T. E., Grierson, C. S.,
& Marucci, L. (2023). Bridging the gap between mechanistic biological models
and machine learning surrogates. PLOS Computational Biology, 19 (4), e1010988.
https://doi.org/10.1371/journal.pcbi.1010988

[7] Losada, I. B., & Terranova, N. (2024). Bridging pharmacology and neural networks:
A deep dive into neural ordinary differential equations. CPT: Pharmacometrics &
Systems Pharmacology. https://doi.org/10.1002/psp4.13149

[8] Poulinakis, K., Drikakis, D., Kokkinakis, I., & Spottswood, S. M. (2023).
Machine-Learning Methods on Noisy and Sparse Data. Mathematics, 11 (1), 236.
https://doi.org/10.3390/math11010236

[9] Caiafa, C. F., Sun, Z., Tanaka, T., Marti-Puig, P., & Sol´e-Casals, J. (2021). Ma-
chine Learning Methods with Noisy, Incomplete or Small Datasets. Applied Sciences,
11 (9), 4132. https://doi.org/10.3390/APP11094132

[10] Goryanin, I., Goryanin, I., & Demin, O. (2025). Revolutionizing drug discovery:
Integrating artificial intelligence with quantitative systems pharmacology. Drug Dis-
covery Today, 104448. https://doi.org/10.1016/j.drudis.2025.104448

[11] Sivakumar, N., Mura, C., & Peirce, S. M. (2022). Innovations in integrating machine
learning and agent-based modeling of biomedical systems. Frontiers in Systems Bi-
ology, 2. https://doi.org/10.3389/fsysb.2022.959665

[12] Combining Machine Learning and Agent-Based Modeling to Study Biomedical Sys-
tems (2022).

[13] Heard, D., Dent, G., Schifeling, T., & Banks, D. (2015). Agent-Based Mod-
els and Microsimulation. Social Science Research Network, 2 (1), 259–272.
https://doi.org/10.1146/ANNUREV-STATISTICS-010814-020218

[14] Holcombe, M., Adra, S. F., Bicak, M., Chin, S., Coakley, S., Graham, A. I.,
Green, J., Greenough, C., Jackson, D. E., Kiran, M., MacNeil, S., Maleki-Dizaji,
A., McMinn, P., Pogson, M., Poole, R. K., Qwarnstrom, E. E., Ratnieks, F. L.
W., Rolfe, M. D., Smallwood, R., . . . Worth, D. (2012). Modelling complex bi-
ological systems using an agent-based approach. Integrative Biology, 4 (1), 53–64.
https://doi.org/10.1039/C1IB00042J
