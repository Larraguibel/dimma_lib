"""Classical DP-SGD (Abadi et al., CCS 2016), Algorithm 1.

The reference every non-classical method in dimma is measured against,
and the shortest path through the pipeline: Poisson subsampling, one
gradient per example, clip, sum, add Gaussian noise, scale, step.

Algorithm 1 in the paper's notation, against the primitive that
implements it:

=====================================  ============================
Algorithm 1                            `dimma.core`
=====================================  ============================
sample ``L_t`` at rate ``q = L/N``     `sampling.poisson`
``g_t(x_i) <- grad L(theta_t, x_i)``   `gradients.per_sample_grads`
``.../max(1, |g_t(x_i)|_2 / C)``       `clipping.per_sample_clip`
``sum_i ...``                          `aggregation.sum_over_batch`
``... + N(0, sigma^2 C^2 I)``          `noise.add_gaussian`
``(1/L) ...``                          `pytree.scale`
``theta_{t+1} <- theta_t - eta_t g~``  `updates.apply`
=====================================  ============================

The first line is stage 1 and lives in
:mod:`~dimma.algorithms.dp_sgd.train`, because a Poisson draw has
data-dependent cardinality and cannot be compiled. The rest is
:mod:`~dimma.algorithms.dp_sgd.step`.

Nothing is re-exported; import from those two modules.
"""

__all__: list[str] = []
