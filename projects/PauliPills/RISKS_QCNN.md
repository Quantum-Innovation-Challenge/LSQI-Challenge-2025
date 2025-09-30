# Risks & Mitigations — QCNN

| Risk | Impact | Mitigation |
|------|--------|------------|
| Runtime spikes | Missed timelines | Cap steps/shots; parallelize CV; cache encodings |
| Barren plateaus | Training stagnates | Shallow depth (≤2); good init; re-uploading; early stopping |
| Overfitting (low-N) | Inflated scores | Patient-level splits; bootstrap CIs; calibration reporting |
| Hardware mismatch | Repro issues | Lock versions; seed control; config-driven runs |
