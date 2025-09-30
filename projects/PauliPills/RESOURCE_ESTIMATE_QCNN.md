# Computational Resource Estimate — QCNN

Let parameters P≈45; parameter-shift uses 2P evaluations/step.
With S=1024 shots, T=150 steps, and k=5 folds:
- circuits ≈ (2P) × T × k = 90 × 150 × 5 = 67,500
- shots ≈ circuits × S = 67,500 × 1,024 = 69,120,000

**Hardware:** CPU (8–16 vCPU), RAM (20–32 GB).  
**Wall-time:** ~2–6 h depending on simulator and parallelism.  
**Notes:** Weekly model uses 7 qubits; per-circuit cost rises, not shot count. We may halve shots (512) or steps (≤100) if needed.
