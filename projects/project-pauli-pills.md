**Team:** pauli's pills

**Update (approach):** We switched to a **pure QCNN** Phase‑1 plan. A mechanistic PK core simulates regimen‑aware exposure curves \(C(t)\), and a parameter‑efficient **Quantum Convolutional Neural Network (QCNN)** with mid‑circuit pooling predicts the probability that all PD values stay below a clinical threshold across a steady‑state window. We focus on **few‑sample generalization** with patient‑level k‑fold, learning curves, bootstrap CIs, and calibration.

**Team leads / official participants (2):**
- Artemiy Burov (FHNW) — @tyrolize
- Leena Anthony (FHNW) — @Anthony-Leena

**Contributors:** Johannes Mosbacher (FHNW), Martin Kuentz (FHNW), Abdullah Kahraman (FHNW/SIB), Nicolas Piro (Independent)

**Repository (code):** https://github.com/tyrolize/paulis-pills

**Project summary (QCNN-only):**
- **PK core**: one‑compartment with first‑order absorption/elimination; allometric scaling and random effects; covariates (BW, COMED).
- **QCNN**: amplitude‑encodes daily (64→6 qubits) / weekly (128→7 qubits) exposure sequences, re‑uploads covariates with \(R_y(\cdot)\); two conv–pool blocks with mid‑circuit measurements; (~45) parameters; SU(4) head.
- **Evaluation**: patient‑level k‑fold (k=5), small‑N learning curves \(N\in\{4,8,16,32\}\), bootstrap CIs, calibration (reliability/ECE), and ablations (qubits, depth, parameters, shots).
- **Dose selection**: select minimal dose s.t. \(\hat\pi(d,s)\ge q\), with \(q\in\{0.90,0.75\}\), over daily/weekly grids.

**Computational resource estimate (QCNN):**
- Parameters \(P\approx45\), parameter‑shift ⇒ \(2P\) circuits/step. With 1024 shots, 150 steps, 5 folds:  
  circuits ≈ \(90\times 150\times 5 = 67{,}500\); shots ≈ \(67{,}500\times 1{,}024 = 69{,}120{,}000\).  
- CPU 8–16 vCPU, 20–32 GB RAM, (~2–6) h depending on parallelism. (Can halve shots or steps if needed.)

**License confirmation:** Code under **Apache‑2.0**; documentation/report under **CC BY 4.0**.

**Linking the initialization issue:** Refs #2

@tyrolize @Anthony-Leena
