
# Efficient Client Selection in Federated Learning

📍 *3-Day Systems for Machine Learning Workshop*  
🗓️ *June 13–15, 2025 — ACM Student Chapter, IISc*

---

## Overview

This project explores **client selection strategies in federated learning (FL)** with a focus on client heterogeneity, profiling, and optimization. It was completed as part of a hands-on workshop organized by the ACM Student Chapter at IISc.

---

## 🚀 Key Highlights

- **Client Profiling**
  - Measured average training time and memory usage per batch size
  - Profiled Raspberry Pi and desktop performance across batch sizes 8–1900

- **Label Distribution Analysis**
  - Calculated label entropy for each client to assess data diversity

- **Baseline FL (Random Selection)**
  - Trained with 3 randomly selected clients over 20 rounds using FedAvg
  - Tracked accuracy, label coverage, and training efficiency

- **Smart Selection Strategy**
  - Developed a scoring heuristic:
    ```python
    score = entropy / minibatch_time
    ```
  - Prioritized fast and diverse clients for selection

---

##  Results

| Metric                    | Random         | Smart (Entropy/Time) |
|---------------------------|----------------|-----------------------|
| Final Accuracy (20 rounds)| ~0.33          | **~0.36**             |
| Jain’s Index              | **0.71**       | 1.00 (unfair)         |
| Target Accuracy Speed     | 20 rounds      | **18 rounds**         |

Visualizations include:
- Accuracy vs. Round & Time
- Moving Average Accuracy
- Client Selection Fairness
- Throughput vs Batch Size

---

## Technologies Used

- Python, PyTorch
- Jupyter Notebooks
- matplotlib, seaborn, pandas
- Raspberry Pi 4
- YAML-configured simulation framework

---

##  Folder Structure

```
├── data_analysis/           # Label distribution and entropy plots
├── profile_on_rbpi/         # Batch size profiling on Raspberry Pi
├── simulation/              # Random and Smart FL runs
├── configs/config.yaml      # Experiment configuration
├── models/CNN.py            # Model used in FL (enhanced CNN)
└── results/                 # Output logs and plots
```


---

## 📎 Acknowledgments

Thanks to the **ACM Student Chapter, IISc** and all organizers for hosting the workshop.
