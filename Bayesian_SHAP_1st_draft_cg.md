
---

## 🔍 Motivation

While SHAP provides local explanations for model predictions using feature attributions, it lacks **direct uncertainty quantification**, which is critical in high-stakes domains like healthcare or finance. This absence:

* Hinders trust when the input lies in low-density data regions.
* Ignores epistemic and aleatoric uncertainty during explanation.
* Fails to reflect the variability induced by data sampling and model instability in ensemble settings.

---

## 💡 Intuition

SHAP values are expectations over marginal contributions of features across coalitions. This marginal contribution is inherently **stochastic**, especially in ensembles. Our key idea is:

> **Treat each SHAP value not as a point estimate but as a posterior distribution** under a Bayesian framework over coalition outcomes.

Instead of returning a single SHAP value, return:

$$
\phi_i(x) \sim \mathcal{P}_i
$$

where $\mathcal{P}_i$ is a **posterior distribution** over the contribution of feature $i$ to prediction $f(x)$.

---

## 🧠 Theoretical Foundation

### Definitions

Let:

* $f(x)$ be a prediction from a tree ensemble.
* $S \subseteq \{1, \dots, M\}$ be a subset of features.
* $x_S$ be input values with only features in $S$.
* $v(S) = \mathbb{E}[f(x_S, X_{\bar{S}})]$, where $\bar{S}$ is the complement.

The SHAP value of feature $i$ is:

$$
\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(M - |S| - 1)!}{M!} [v(S \cup \{i\}) - v(S)]
$$

We reinterpret:

$$
\Delta_i^S = v(S \cup \{i\}) - v(S)
$$

as a **random variable**, denoted $Z_i^S$, due to sampling, tree variation, and feature correlations.

### Assumptions

1. **Exchangeability**: Coalitions of the same size are exchangeable (per Shapley axioms).
2. **Posterior Additivity**: The expectation over SHAP distributions equals the original SHAP value:

   $$
   \mathbb{E}_{Z_i^S}[\phi_i(x)] = \phi_i(x)
   $$
3. **Bayesian Belief Update**: Prior over marginal contribution $Z_i^S$ is updated via Bayesian inference using ensemble variability.

---

## 📐 Methodology: BayesDSHAP

### Step 1: Prior Definition

Model the marginal contributions $Z_i^S$ as Gaussian (or Student-t for robustness):

$$
Z_i^S \sim \mathcal{N}(\mu_i^S, \sigma_i^S)
$$

The prior mean $\mu_i^S$ can be initialized via TreeSHAP estimates on bootstrapped subsamples.

### Step 2: Likelihood from Ensemble Models

Let each tree $t$ in ensemble produce contribution $z_{i,t}^S$. Assume:

$$
p(z_{i,t}^S | \mu_i^S, \sigma_i^S) = \mathcal{N}(\mu_i^S, \sigma_i^2 / T)
$$

### Step 3: Posterior Computation

Use conjugate priors (Gaussian-Gaussian):

$$
\mu_i^S | \{z_{i,t}^S\}_{t=1}^T \sim \mathcal{N}\left(\hat{\mu}_i^S, \hat{\sigma}_i^S\right)
$$

where:

* $\hat{\mu}_i^S = \frac{\sum_t z_{i,t}^S}{T}$
* $\hat{\sigma}_i^S = \sigma^2 / T$

### Step 4: Posterior Aggregation

Combine posteriors across all subsets $S$:

$$
\phi_i(x) \sim \sum_{S \subseteq F \setminus \{i\}} w_S \cdot \mathcal{N}(\hat{\mu}_i^S, \hat{\sigma}_i^S)
$$

where $w_S$ is the Shapley kernel weight.

### Step 5: Closed-form SHAP Uncertainty

Assuming independence across subsets (approximation):

$$
\phi_i(x) \sim \mathcal{N}\left( \sum_S w_S \hat{\mu}_i^S,\ \sum_S w_S^2 \hat{\sigma}_i^S \right)
$$

* Mean gives the traditional SHAP value.
* Variance gives the **uncertainty** over SHAP.

---

## 📊 Advantages

| Problem in SHAP                       | How BayesDSHAP Solves It                            |
| ------------------------------------- | --------------------------------------------------- |
| No uncertainty info                   | Returns credible intervals over $\phi_i(x)$         |
| Inseparability of data/model variance | Bayesian framework separates priors and likelihoods |
| Instability under correlated features | Posterior regularizes high variance coalitions      |

---

## 🧪 Experimental Protocol

1. **Datasets**: Use UCI, healthcare risk prediction, fraud detection.
2. **Models**: Random Forests, XGBoost, CatBoost.
3. **Metrics**:

   * SHAP uncertainty width vs prediction error.
   * Calibration of SHAP intervals.
   * Robustness to feature correlation (simulate collinearity).
   * Agreement with domain expert confidence.
4. **Baselines**:

   * Bootstrap SHAP (non-Bayesian).
   * Gaussian Process SHAP (if available).
   * Normalized SHAP (z-score only).

---

## 📐 Mathematical Proof Sketch

Let $\phi_i(x)$ be a weighted sum of Gaussians:

$$
\phi_i(x) = \sum_S w_S Z_i^S, \quad Z_i^S \sim \mathcal{N}(\hat{\mu}_i^S, \hat{\sigma}_i^S)
$$

By linearity of expectation and independence:

$$
\mathbb{E}[\phi_i(x)] = \sum_S w_S \hat{\mu}_i^S, \quad
\text{Var}[\phi_i(x)] = \sum_S w_S^2 \hat{\sigma}_i^S
$$

This proves BayesDSHAP preserves the expected SHAP values, while providing uncertainty as variance.

---

## 🔍 Novelty

* Combines **Bayesian inference with Shapley theory**.
* Grounded in **probabilistic modeling** of feature coalitions.
* Applicable **directly on top of TreeSHAP**.
* Provides **mathematically provable** and **calibrated uncertainty**.

---
