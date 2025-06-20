
## Usefulness

- Existing methods mostly focus on predictive uncertainty or require costly retraining, and do not directly quantify uncertainty in SHAP values while respecting key structures like the combinatorial nature of Shapley values and dependencies in tree-based ensembles[3][6].
- Our approach addresses this gap by explicitly modeling uncertainty in SHAP values, which is critical for trustworthy explanations and practical deployment in real-world scenarios where computational constraints exist[3][6].
- The decomposition of SHAP variance into aleatoric and epistemic components can provide deeper insights into the sources of uncertainty, which is valuable for model interpretation, feature selection, and robustness analysis[1][7].
- Stability-constrained computation to ensure path-dependent feature attribution under perturbation aligns with concerns about explanation robustness and reliability, which have been highlighted as important challenges[3][6].

## Novelty

- Hypothesis Space Sampling using Dirichlet-weighted tree sampling from existing ensembles is a distinctive innovation that respects the dependency structure of tree ensembles, which is not commonly addressed by existing methods that often rely on bootstrap or Bayesian approximations without such structural considerations[3][6].
- The theoretical variance decomposition separating aleatoric and epistemic uncertainty in SHAP values is a novel theoretical contribution that extends beyond prior work focusing mainly on point estimates or single uncertainty measures[1][3].
- Stability-constrained computation introduces a new mechanism to maintain consistent feature attributions under input perturbations, which is a less-explored area in SHAP uncertainty quantification literature[3][6].

## Supporting Literature

- Research efforts have begun to quantify uncertainty in SHAP values using bootstrap sampling and confidence intervals, but these often do not fully capture the combinatorial and dependency structures our method targets[3].
- Other works adapt Shapley values to explain predictive uncertainty from an information-theoretic perspective but do not explicitly quantify SHAP value uncertainty respecting tree ensemble dependencies or computational constraints[1][4][7].
- Efficient algorithms for explanation uncertainty estimation exist but typically focus on stochastic approximations without the comprehensive variance decomposition and stability constraints you propose[6].

E-SHAP framework fills a notable gap by integrating structural, theoretical, and computational innovations to quantify SHAP value uncertainty in a way that is both theoretically grounded and practically feasible. This makes it a useful and novel contribution to the explainable AI and uncertainty quantification literature.

[1] https://proceedings.neurips.cc/paper_files/paper/2023/file/16e4be78e61a3897665fa01504e9f452-Paper-Conference.pdf
[2] https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Explaining%20quantitative%20measures%20of%20fairness.html
[3] https://scholarship.tricolib.brynmawr.edu/bitstream/handle/10066/23538/2021LiR.pdf?sequence=1&isAllowed=y
[4] https://arxiv.org/abs/2306.05724
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC8299327/
[6] https://papers.phmsociety.org/index.php/phmap/article/download/3694/2161
[7] https://openreview.net/forum?id=6rabAZhCRS
[8] https://www.sciencedirect.com/science/article/pii/S0377221724004715
