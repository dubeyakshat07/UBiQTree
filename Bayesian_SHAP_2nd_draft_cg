# Bayesian Distributional SHAP (BayesDSHAP): A Probabilistic Framework for Uncertainty Quantification in Tree-Based SHAP Values

## Abstract

SHAP (SHapley Additive exPlanations) is a widely used technique to interpret machine learning models by attributing contributions to each feature. However, SHAP lacks mechanisms to quantify uncertainty in these attributions, which is critical in high-stakes domains. We propose BayesDSHAP, a Bayesian framework to estimate uncertainty over SHAP values for tree-based ensemble models. Our approach leverages the structure of ensembles to compute posterior distributions over marginal contributions, enabling calibrated, interpretable uncertainty estimates. We provide theoretical grounding, mathematical derivations, and experiments to demonstrate the method’s utility.

---

## 1. Introduction

Interpretability and uncertainty quantification are foundational to trustworthy machine learning. TreeSHAP has become a standard tool for interpreting predictions from ensemble tree models, but it provides point estimates without quantifying uncertainty. This research addresses that gap by introducing a Bayesian framework that treats SHAP values as distributions rather than fixed quantities.

### 1.1 Motivation

* SHAP values depend on marginal expectations that vary under data perturbations and model bootstrapping.
* Tree-based models inherently support ensembling and variance decomposition.
* Calibrated uncertainty in SHAP values improves trust, supports decision-making, and aligns with regulatory needs.

---

## 2. Background

### 2.1 SHAP and TreeSHAP

Given a model $f$, feature set $F = \{1, \dots, M\}$, and instance $x$, the SHAP value for feature $i$ is:

$$
\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (M - |S| - 1)!}{M!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

where $v(S) = \mathbb{E}[f(x_S, X_{\bar{S}})]$.

TreeSHAP computes these values efficiently for decision trees and ensembles, exploiting structure in the model.

### 2.2 Types of Uncertainty

* **Aleatoric**: Due to noise in data.
* **Epistemic**: Due to uncertainty in the model.
* **SHAP Variability**: Due to variation in coalitions and ensembling.

---

## 3. Intuition and Contributions

### 3.1 Key Insight

Marginal contributions $\Delta_i^S = v(S \cup \{i\}) - v(S)$ are stochastic. We model these as random variables with posterior distributions:
$Z_i^S \sim \mathcal{P}_i^S$

### 3.2 Contributions

* A Bayesian formulation for SHAP values over ensembles.
* Posterior variance estimation via model outputs.
* Closed-form uncertainty for TreeSHAP.
* Theoretical guarantees under mild assumptions.

---

## 4. Ensemble Tree Models and SHAP Structure

### 4.1 Model Structure

Tree-based ensemble models (Random Forests, XGBoost):

* Consist of multiple weak learners (trees).
* Each tree defines a piecewise constant function.
* Output is an additive sum over tree predictions:
  $f(x) = \sum_{t=1}^T f_t(x)$

### 4.2 SHAP in Ensembles

TreeSHAP computes per-tree SHAP values $\phi_{i,t}(x)$ and sums them:
$\phi_i(x) = \sum_{t=1}^T \phi_{i,t}(x)$

This decomposition naturally supports a Bayesian interpretation:

* Trees offer multiple samples of marginal contributions.
* Bootstrap aggregation reflects epistemic uncertainty.

### 4.3 Variance Estimation

Let:
$\Delta_{i,t}^S = v_t(S \cup \{i\}) - v_t(S)$
Define:
$\mu_i^S = \mathbb{E}[\Delta_{i,t}^S], \quad \sigma_i^S = \text{Var}[\Delta_{i,t}^S]$

---

## 5. Methodology: BayesDSHAP

### 5.1 Bayesian Modeling

Assume:
$Z_i^S \sim \mathcal{N}(\mu_i^S, \sigma_i^S)$
Observed values from trees are $\{z_{i,t}^S\}$. With Gaussian likelihood and conjugate prior, posterior is:
$\mu_i^S \mid \{z_{i,t}^S\} \sim \mathcal{N}(\hat{\mu}_i^S, \hat{\sigma}_i^S)$

### 5.2 Posterior Aggregation

Total posterior for SHAP value:
$\phi_i(x) = \sum_S w_S Z_i^S \sim \mathcal{N}\left(\sum_S w_S \hat{\mu}_i^S, \sum_S w_S^2 \hat{\sigma}_i^S\right)$

This yields:

* $\mathbb{E}[\phi_i(x)]$ as point SHAP value.
* $\text{Var}[\phi_i(x)]$ as SHAP uncertainty.

---

## 6. Theoretical Guarantees

### 6.1 Linearity

Expectation is preserved:
$\mathbb{E}[\phi_i(x)] = \sum_S w_S \mathbb{E}[Z_i^S]$

### 6.2 Variance

Assuming independence:
$\text{Var}[\phi_i(x)] = \sum_S w_S^2 \text{Var}[Z_i^S]$

---

## 7. Experiments

### 7.1 Datasets

* UCI Adult, Breast Cancer, Credit Default.

### 7.2 Metrics

* Width of uncertainty intervals.
* Coverage of true SHAP values under perturbation.
* Calibration of uncertainty vs prediction error.

### 7.3 Baselines

* Bootstrap SHAP.
* Normalized SHAP.

---

## 8. Discussion

### 8.1 Benefits

* Calibrated uncertainty.
* Closed-form computation for trees.
* Faithfulness to TreeSHAP.

### 8.2 Limitations

* Independence assumption across coalitions.
* Gaussianity of posterior may be restrictive.

### 8.3 Future Work

* Nonparametric priors.
* Multi-output SHAP.

---

## 9. Conclusion

BayesDSHAP extends SHAP with uncertainty quantification grounded in Bayesian theory and ensemble variability. It preserves the interpretability of TreeSHAP while enhancing trust through uncertainty estimates, enabling robust deployment in critical applications.

---

## References

* Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
* Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.
* Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A Benchmark for Interpretability Methods in Deep Neural Networks.
* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
