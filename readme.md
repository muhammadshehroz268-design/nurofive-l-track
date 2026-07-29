# Model Evaluation & Hyperparameter Tuning
## 1. Why Accuracy is Misleading for Imbalanced Data
Accuracy calculates the proportion of total correct predictions out of all predictions. In imbalanced datasets (e.g., 90% class 0 and 10% class 1), a model that predicts class 0 every time achieves **90% accuracy**, yet fails to detect a single positive case. 
Metrics like **Precision**, **Recall**, and **F1-Score** provide a clearer evaluation:
- **Precision:** Measures how many of the positively predicted instances were actually positive.
- **Recall:** Measures how many actual positive instances were correctly identified.
- **F1-Score:** The harmonic mean of Precision and Recall, balancing both metrics.
---
## 2. Performance Comparison: Baseline vs Tuned Model
We used `GridSearchCV` with 5-fold cross-validation to tune key hyperparameters (`max_depth`, `min_samples_split`, and `criterion`).

| Metric | Baseline Model | Tuned Model (GridSearchCV) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7413 | 0.7832 | +4.19% |
| **Precision** | 0.6808 | 0.7500 | +6.92% |
| **Recall** | 0.6667 | 0.6875 | +2.08% |
| **F1-Score** | 0.6737 | 0.7174 | +4.37% |

> **Best Parameters Found:** `{'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 5}`
---
## 3. Summary of Improvements
Hyperparameter tuning helped control decision tree depth and prevented overfitting on noisy training samples. This boosted general accuracy while improving the **F1-Score**, ensuring higher reliability on unseen test samples.
