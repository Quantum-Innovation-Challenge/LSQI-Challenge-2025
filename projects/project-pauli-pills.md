---
number: 1 # maintainers adjust
title: Validating Few‑Sample Quantum ML for PK/PD Dose Selection
topic: pkpd
team_leads:
  - Artemiy Burov (FHNW) @tyrolize
  - Leena Anthony (FHNW) @Anthony-Leena

contributors:
  - Johannes Mosbacher (FHNW)
  - Martin Kuentz (FHNW)
  - Abdullah Kahraman (FHNW / SIB)
  - Nicolas Piro (Sony Advanced Visual Sensing AG (Zurich, Switzerland))

github: tyrolize/paulis-pills
# youtube_video: <add-id-when-ready>
---

**QCNN-only plan.** A mechanistic PK core simulates regimen-aware exposure curves \(C(t)\) (daily/weekly). A parameter-efficient **Quantum Convolutional Neural Network (QCNN)** with mid-circuit pooling predicts the probability that all PD values remain below a clinical threshold across a steady-state window. We emphasise small-\(N\) validation: patient-level k-fold, learning curves, bootstrap CIs, calibration, and ablations.

**Official project participants:** Leena Anthony (FHNW), Artemiy Burov (FHNW).  
**Links:** Code & docs — https://github.com/tyrolize/paulis-pills
