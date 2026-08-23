# Algorithms

One package per algorithm under `dimma.algorithms`, one page per algorithm
here. Every algorithm is a choice at each stage of the
[pipeline](../library/pipeline.md) — including the choice to leave a stage
out — plus the loop that runs the choices, and its own accounting where
the standard accountants do not cover its mechanism.

| Algorithm | Paper |
|---|---|
| [DP-SGD](dp-sgd.md) | Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016 |
| [Private SpiderBoost](spiderboost.md) | Arora et al., *Faster Rates of Convergence to Stationary Points in Differentially Private Optimization*, ICML 2023 |
| [Bias-reduced sparse SGD](bias-reduced-sparse-sgd.md) | Ghazi et al., *Differentially Private Optimization with Sparse Gradients*, NeurIPS 2024 |
| [Non-private baselines](baselines.md) | — |

Each page names the notebook that evaluates its algorithm; the executed
comparisons themselves are collected under
[evaluation on Criteo](../evaluation.md).
