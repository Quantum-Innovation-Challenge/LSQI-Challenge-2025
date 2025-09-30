# Quantum Enhanced Machine Learning Model for Dose Optimization in Early Drug Development

This repository contains our submission for the **Quantum Innovation Challenge 2025** by **Team PNU**.  
We propose a **quantum-enhanced machine learning framework** for optimizing drug dosing strategies in the early stages of pharmaceutical development.

---

## 🚀 Quick Start

### Training a Model
```bash
python main.py --mode {mode} --encoder {encoder} 
```
### Optional Arguments
--use_fe → enable feature engineering
--use_mc_dropout → apply Monte Carlo Dropout for uncertainty estimation
--use_pt_contrast → activate contrastive learning between PK/PD tasks

### Key Features
#### Training Modes
- independent → PK and PD models trained separately
- cascade → PK and PD models trained jointly with shared information
- multitask → PK first, then PD conditioned on PK outputs

#### Model Architectures
- mlp → Multi-Layer Perceptron
- resmlp → Residual MLP
- moe → Mixture of Experts
- resmlp_moe → Residual MLP + MoE
- qnn → Quantum Neural Network-enhanced MLP
- resqnn_moe → Quantum-enhanced Residual MLP + MoE

#### Advanced Features
- Feature Engineering → time windows, per-kg dosing, future dose information
- PK/PD Contrastive Learning → domain-specific positive/negative pair generation
- Uncertainty Estimation → Monte Carlo Dropout

## 📄 **License**

This project is licensed under the MIT License. See the LICENSE file for details. Developed and maintained by Team PNU for the Quantum Innovation Challenge 2025.

 ---

