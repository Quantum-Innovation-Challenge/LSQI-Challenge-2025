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
  - "Sofia Moliner Bobo (Imperial College London) <sofia.moliner23@imperial.ac.uk>"

# Optional fields; set to null or remove if unused
github: "louis-chen-jij/LSQI-Challenge-2025"
---

Abstract: 
QuSOP is a hardware-agnostic, regulator-aligned framework that pairs a quantum long short-term memory predictor (QLSTM) with quantum Monte Carlo (QMC) to deliver uncertainty-aware, chance-constrained dose selection for PK/PD decision-making. QLSTM learns steady-state biomarker trajectories over 24-hour and 168-hour windows from sparse, irregular data, while QMC propagates covariate priors (such as body weight and concomitant medication) to estimate attainment probabilities on discrete dose grids; the minimal dose is then chosen to meet prespecified success levels (for example, 90% or 75%). Quantum resources are confined to training and estimation routines, and inference is deterministic on CPU/GPU with full audit trails. On a Phase-1–like dataset, the QLSTM+QMC pipeline yields well-calibrated sigmoidal dose–success curves and consistent once-daily and once-weekly recommendations; relaxing the attainment criterion from 90% to 75% reduces mean minimal doses by about 25% for once-daily (19.0→14.3 mg) and about 13% for once-weekly (116.9→101.3 mg), with similar trends when concomitant medication is excluded. QuSOP supports covariate-shift stress tests and multi-endpoint extensions, and is deployable in sovereign GPU environments while remaining compatible with emerging QPUs. By combining QLSTM’s sample-efficient temporal modeling with QMC’s calibrated uncertainty propagation, QuSOP offers a pragmatic path to quantum-enhanced, transparent, clinically relevant dose policies in early development.

References:
[1] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. “Quantum machine learning.” Nature 549(7671):195–202, 2017.

[2] Sergey Bravyi, Oliver Dial, Jay M. Gambetta, Darío Gil, and Zaira Nazario. “The future of quantum computing with superconducting qubits.” Journal of Applied Physics 132(16), 2022.

[3] Matthias C. Caro, Hsin-Yuan Huang, Marco Cerezo, Kunal Sharma, Andrew Sornborger, Łukasz Cincio, and Patrick J. Coles. “Generalization in quantum machine learning from few training data.” Nature Communications 13:4919, 2022.

[4] Marco Cerezo, Guillaume Verdon, Hsin-Yuan Huang, Łukasz Cincio, and Patrick J. Coles. “Challenges and opportunities in quantum machine learning.” Nature Computational Science 2(9):567–576, 2022.

[5] Samuel Yen-Chi Chen, Chao-Han Huck Yang, Jun Qi, Pin-Yu Chen, Xiaoli Ma, and Hsi-Sheng Goan. “Variational quantum circuits for deep reinforcement learning.” IEEE Access 8:141007–141024, 2020.

[6] Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang. “Quantum long short-term memory.” In ICASSP 2022 – IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 8622–8626. IEEE, 2022a.

[7] Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang. “Quantum long short-term memory.” In ICASSP 2022 – IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 8622–8626. IEEE, 2022b.

[8] Hakan Doga, Aritra Bose, M. Emre Sahin, Joao Bettencourt-Silva, Anh Pham, Eunyoung Kim, Alan Andress, Sudhir Saxena, Laxmi Parida, Jan Lukas Robertus, Hideaki Kawaguchi, Radwa Soliman, and Daniel Blankenberg. “How can quantum computing be applied in clinical trial design and optimization?” Trends in Pharmacological Sciences, 2024.

[9] Jay Gambetta. “Quantum-centric supercomputing: The next wave of computing.” IBM Research Blog, 2022.

[10] Sepp Hochreiter and Jürgen Schmidhuber. “Long short-term memory.” Neural Computation 9(8):1735–1780, 1997.

[11] Yu-Chao Hsu, Nan-Yow Chen, Tai-Yu Li, Po-Heng Henry Lee, and Kuan-Cheng Chen. “Quantum kernel-based long short-term memory for climate time-series forecasting.” In 2025 International Conference on Quantum Communications, Networking, and Computing (QCNC), 421–426. IEEE, 2025a.

[12] Yu-Chao Hsu, Jiun-Cheng Jiang, Chun-Hua Lin, Wei-Ting Chen, Kuo-Chung Peng, Prayag Tiwari, Samuel Yen-Chi Chen, and En-Jui Kuo. “Federated quantum kernel-based long short-term memory for human activity recognition.” arXiv:2508.06078, 2025b.

[13] Yu-Chao Hsu, Tai-Yu Li, and Kuan-Cheng Chen. “Quantum kernel-based long short-term memory.” In 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), 1–5. IEEE, 2025c.

[14] Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R. McClean. “Power of data in quantum machine learning.” Nature Communications 12:2631, 2021.

[15] Hsin-Yuan Huang, Michael Broughton, Jordan Cotler, Sitan Chen, Jerry Li, Masoud Mohseni, Hartmut Neven, Ryan Babbush, Richard Kueng, John Preskill, et al. “Quantum advantage in learning from experiments.” Science 376(6598):1182–1186, 2022.

[16] Chu-Hsuan Abraham Lin, Chen-Yu Liu, and Kuan-Cheng Chen. “Quantum-train long short-term memory: Application on flood prediction problem.” arXiv:2407.08617, 2024.

[17] Chen-Yu Liu, En-Jui Kuo, Chu-Hsuan Abraham Lin, Jason Gemsun Young, Yeong-Jar Chang, Min-Hsiu Hsieh, and Hsi-Sheng Goan. “Quantum-train: Rethinking hybrid quantum-classical machine learning in the model compression perspective.” arXiv:2405.11304, 2024a.

[18] Chen-Yu Liu, Chu-Hsuan Abraham Lin, Chao-Han Huck Yang, Kuan-Cheng Chen, and Min-Hsiu Hsieh. “QTRL: Toward practical quantum reinforcement learning via Quantum-Train.” arXiv:2407.06103, 2024b.

[19] Junhua Liu, Kwan Hui Lim, Kristin L. Wood, Wei Huang, Chu Guo, and He-Liang Huang. “Hybrid quantum-classical convolutional neural networks.” Science China Physics, Mechanics & Astronomy 64(9):290311, 2021a.

[20] Xiangyu Liu, Chao Liu, Ruihao Huang, Hao Zhu, Qi Liu, Sunanda Mitra, and Yaning Wang. “Long short-term memory recurrent neural network for pharmacokinetic–pharmacodynamic modeling.” International Journal of Clinical Pharmacology and Therapeutics 59(2):138, 2021b.

[21] Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, and Nathan Killoran. “Transfer learning in hybrid classical–quantum neural networks.” Quantum 4:340, 2020.

[22] S. S. Negus and M. L. Banks. “Pharmacokinetic–pharmacodynamic (PK/PD) analysis with drug discrimination.” In The Behavioral Neuroscience of Drug Discrimination. Springer, 2016.

[23] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. “Data re-uploading for a universal quantum classifier.” Quantum 4:226, 2020.

[24] John Preskill. “Quantum computing in the NISQ era and beyond.” Quantum 2:79, 2018.

[25] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. “Effect of data encoding on the expressive power of variational quantum machine-learning models.” Physical Review A 103(3):032430, 2021.

[26] Joseph Standing. “Understanding and applying pharmacometric modelling and simulation in clinical practice and research.” British Journal of Clinical Pharmacology, 2016.

[27] Mizuki Uno, Yuta Nakamaru, and Fumiyoshi Yamashita. “Application of machine learning techniques in population pharmacokinetics/pharmacodynamics modeling.” Drug Metabolism and Pharmacokinetics, 2024.
