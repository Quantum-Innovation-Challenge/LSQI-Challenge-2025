# Evaluation Plan — Pure QCNN

- **Splits:** patient-level k-fold (k=5), no patient in both train/test.
- **Few-sample curves:** N ∈ {4, 8, 16, 32} per fold.
- **Metrics:** RMSE/MAE (if regressing PD); AUC (if classifying success).
- **Uncertainty:** 200× bootstrap 95% CIs; seed variance reported.
- **Calibration:** reliability curves and ECE.
- **Ablations:** qubits (6→7 + feature re-uploading), depth (1 vs 2), parameters (~45), shots (512 vs 1024).
