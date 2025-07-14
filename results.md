
---

## **3.2 Epistemic Uncertainty Analysis for Class "No PCOS"**

To systematically understand the **epistemic uncertainty** embedded in the model’s explanations, we performed an ensemble-based SHAP analysis for the class "No PCOS". This allows us to go beyond average SHAP values and assess the **reliability, variability, and directional consistency** of feature attributions. Our findings are summarized in three complementary visualizations (Figure X), each designed to dissect different facets of uncertainty.

---

### **3.2.1 Aggregate Feature Importance with Uncertainty Quantification**

The first plot (Figure X-a) displays the **mean SHAP values** across multiple sub-models alongside **epistemic uncertainty** represented as ±2 standard deviations (σ) using dark red whiskers. Features are ranked in descending order of their mean contribution to the class prediction (No PCOS). The color intensity of the bars corresponds to the absolute magnitude of their SHAP value, aiding visual emphasis on dominant contributors.

Most notably, **Antral Follicle Count** emerges as the most influential feature, contributing positively to the prediction of the No PCOS class. However, the associated uncertainty whiskers (2σ) are relatively long, indicating considerable variability across different model instantiations (i.e., different sub-ensembles or random seeds). This suggests that while the model consistently relies on this feature, it does so with **substantial epistemic ambiguity** regarding its precise impact magnitude.

In contrast, features such as **Testosterone Level**, **Menstrual Irregularity**, and **BMI** exhibit low average SHAP values and narrow uncertainty whiskers, reflecting both **low importance and high confidence** in their negligible contribution. Features like **Age**, though showing a modest average impact, also exhibit relatively lower uncertainty, hinting at **stable but marginal roles** in class prediction.

---

### **3.2.2 SHAP Distributional Analysis: Feature-Wise Uncertainty and Directionality**

To further investigate the **stability and credibility** of the most influential features, we examined the **distribution of SHAP values** for `Antral Follicle Count` (Figure X-b). The KDE plot reveals a **right-skewed but unimodal** distribution of SHAP values collected from different model samples. The red dashed vertical line marks the mean SHAP value, while the shaded region represents the 95% credible interval.

Three key indicators were derived:

* **Standard Deviation (σ = 0.169)**: High dispersion, reinforcing that the precise magnitude of influence is variable across model instances.
* **Sign Stability (100%)**: Despite variability, all sampled SHAP values for this feature retained the same **positive sign**, indicating consistent directional contribution to the "No PCOS" class.
* **Entropy (not shown here)**: Suggests a relatively dispersed distribution, reinforcing the model's uncertainty about precise attribution.

These findings reveal that while the magnitude of effect is **epistemically uncertain**, the **direction of effect is highly stable**, thus supporting this feature’s interpretability and trustworthiness in clinical inference.

---

### **3.2.3 Global Uncertainty Metrics: Comparative Feature Assessment**

The third panel (Figure X-c) systematically compares three orthogonal measures of epistemic uncertainty across all features:

#### a) **Standard Deviation of SHAP Values (Top Subplot)**

This metric reflects **model variance** in attribution for each feature. A high standard deviation suggests inconsistent SHAP values across the ensemble, reflecting uncertainty about the feature’s influence. `Antral Follicle Count` again appears as the feature with the highest variance, while features such as `Testosterone Level` and `Menstrual Irregularity` show minimal variance, indicating robust and reliable attributions.

#### b) **Explanation Entropy (Middle Subplot)**

Explanation entropy captures the **distributional uncertainty** of SHAP values across the model ensemble. High entropy indicates a **flat or dispersed** distribution of SHAP values, signaling **low information certainty** about the feature's impact. Features such as `Antral Follicle Count` and `BMI` exhibit higher entropy, implying less concentrated explanations. Conversely, features like `Testosterone Level` have low entropy, indicating consistent and peaked SHAP value distributions.

#### c) **Directional Stability or Sign Consistency (Bottom Subplot)**

This measure quantifies how consistently the **sign of the SHAP value** remains positive or negative across sub-models. Features with high sign stability (> 0.9) are directionally reliable, while those with low stability (< 0.67) are flagged as **low-confidence explanations**, given the sign of their contribution fluctuates. `Antral Follicle Count` maintains 100% sign stability, again reinforcing its interpretive consistency despite high magnitude variance. In contrast, features like `Age` exhibit directional instability, casting doubt on the validity of their explanatory power in any given model realization.

---

### **3.2.4 Implications for Interpretability and Clinical Trust**

Together, these visualizations offer a **multifaceted view of epistemic uncertainty**, a crucial consideration when deploying interpretable models in safety-critical domains like healthcare. The analyses affirm that while some features (e.g., `Antral Follicle Count`) are both impactful and directionally stable, others suffer from magnitude uncertainty (high variance) or sign inconsistency, both of which challenge their interpretive reliability.

By explicitly quantifying epistemic uncertainty along three axes—**magnitude variability**, **distributional dispersion**, and **directional consistency**—our approach enhances trust in model explanations and allows domain experts to discern **which features can be trusted** in downstream decisions and which should be treated with caution.

---

