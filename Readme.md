-----

# UbiQTree: Uncertainty Quantification in XAI for Tree Ensembles

**UbiQTree** is a Python library for quantifying the **epistemic uncertainty** in SHAP-based explanations for tree ensemble models. Go beyond average feature importances and understand the *reliability* of your model's explanations.

-----

## Key Features 🎯

  * **Quantify Model Uncertainty**: Directly measure the uncertainty (or confidence) the model has in its own feature attributions.
  * **Multi-Faceted Analysis**: Decomposes uncertainty into three key aspects: magnitude, distributional spread, and directional consistency.
  * **Intuitive Visualizations**: Generates clear, publication-ready plots to interpret uncertainty, including violin plots of SHAP distributions and global uncertainty comparisons.
  * **Robust Foundation**: Built on rigorous statistical methods, including **Dempster-Shafer Theory** and **Dirichlet Processes**.
  * **Build Trust**: Helps you distinguish between trustworthy explanations and those that are unreliable, which is critical for high-stakes applications.

-----

## Methodology

`UbiQTree` quantifies uncertainty in machine learning explanations by integrating two core statistical frameworks: **Dempster-Shafer Theory (DST)** and **Dirichlet Processes (DP)**. DST provides a powerful way to reason about evidence by defining a **Belief-Plausibility interval**. **Belief** represents the conservative lower bound of evidence supporting an explanation, while **Plausibility** is the optimistic upper bound. The gap between these two values explicitly measures our ignorance or uncertainty. A Dirichlet Process, a flexible Bayesian method, is then used to model the distribution of these explanations across an ensemble of models. This process is governed by a key **concentration parameter, alpha (α)**, which controls how focused or spread out the belief is across different explanations.

This foundation allows `UbiQTree` to provide a multi-faceted view of explanation uncertainty.

### 1\. Aggregate Feature Importance with Distributional Uncertainty

First, visualize the **mean contribution (SHAP value)** of each feature alongside its **epistemic uncertainty**. This uncertainty is shown using a **violin plot**, which reveals the full distribution of how its importance varies across different versions of the model.

  * **High-Impact, High-Uncertainty Features**: A feature might be influential on average, but a **wide violin plot** shows a large spread in importance scores, signaling **substantial epistemic ambiguity**.
  * **Low-Impact, High-Confidence Features**: Other features might have low importance and a **very narrow or "spiky" violin plot**, reflecting the model's high confidence that their contribution is negligible.
  * **Stable but Marginal Features**: Some features may have a modest impact but also a narrow violin plot, hinting at a **stable but marginal role**.

### 2\. Distributional Analysis: A Deeper Look at Top Features

To investigate the **stability and credibility** of influential features, you can examine the full distribution of their importance scores.

This reveals key stability indicators:

  * **Standard Deviation (σ)**: A high standard deviation shows that the exact magnitude of the feature's influence is highly variable.
  * **Sign Stability**: This checks if the feature's contribution is always positive or always negative. A 100% stability means its **directional effect is perfectly consistent**, even if its magnitude varies.
  * **Epistemic Uncertainty**: Quantifying the uncertainty from the data and the model itself.
### 3\. Global Uncertainty Comparison: A Feature-by-Feature Showdown

Finally, systematically compare all features across three orthogonal measures of uncertainty in one comprehensive view.

  * **Standard Deviation of SHAP Values**: Directly reflects the **model's variance** in attributing importance. High variance signals an inconsistent explanation.
  * **Explanation Entropy**: Captures the **distributional uncertainty**. High entropy means the importance scores are spread out, signaling low information certainty.
  * **Directional Stability (Sign Consistency)**: Quantifies how consistently a feature's effect is positive or negative. High stability (\>90%) means it is **directionally reliable**, while low stability makes an explanation untrustworthy.

-----

## Quickstart: Example Usage 


```python
from UBiQTree import ExplainerClassification
import numpy as np

# Assume 'model' is your trained RandomForest classifier (e.g., RandomForest)
# and 'X_train', 'y_train', 'X_test' are your data.
# 'feature_names' and 'class_names' are lists of strings.

explainer = ExplainerClassification(model, X_train, y_train, beta=3.0, random_state=42)

# Explain the first instance from the test set
instance = X_test[0:1]

# Generate explanations and plots for each class
for class_idx, class_name in enumerate(class_names):
    print(f"\nExplaining for class: {class_name}")
    results = explainer.explain(instance, n_samples=300, class_idx=class_idx, alpha=0.5)

    # 1. Plot aggregate uncertainty with violin plots
    # Note: Assuming plot_uncertainty_bars now renders violins or similar
    explainer.plot_uncertainty_bars(
        results,
        feature_names,
        class_name=class_name
    )

    # 2. Plot uncertainty distribution for the top 5 features
    top5_indices = np.argsort(np.abs(results["mean"]))[-5:]
    for i in top5_indices:
        explainer.plot_uncertainty_distribution(
            results,
            feature_names,
            i,
            class_name=class_name,
            ylim=(0, 20),
            xlim=(-0.4, 0.4)
        )

    # 3. Plot the global comparison of uncertainty metrics
    explainer.plot_uncertainty_comparison(
        results,
        feature_names,
        class_name=class_name
    )
```

-----

## Citation 📜

If you use this work in your research, please cite the original paper:

```bibtex
@article{dubey2025ubiqtree,
  title={UbiQTree: Uncertainty Quantification in XAI with Tree Ensembles},
  author={Akshat Dubey and Aleksandar Anžel and Bahar İlgen and Georges Hattab},
  journal={arXiv preprint arXiv:2508.09639},
  year={2025},
  doi={10.48550/arXiv.2508.09639},
  url={https://arxiv.org/abs/2508.09639}
}
```

-----

## License

This project is licensed under the [Creative Commons BY-NC-SA 4.0 License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

-----
