"""Non-private SGD: the baseline DP-SGD is measured against.

Not an algorithm from a paper. It is `dimma.algorithms.dp_sgd` with the
privacy taken out and nothing else changed — ADR-0005.

Stage by stage, against Algorithm 1:

=====  ==============================  ===========================
Stage  DP-SGD                          here
=====  ==============================  ===========================
1      `sampling.poisson`              `sampling.shuffled`
3      `gradients.per_sample_grads`    `gradients.batch_grads`
4      `clipping.per_sample_clip`      dropped
5      `aggregation.sum_over_batch`    inside `batch_grads`
6      `noise.add_gaussian`            dropped
7      `updates.apply`                 `updates.apply`
=====  ==============================  ===========================

Stage 5 survives as the mean inside `batch_grads`. Stage 7 is
`updates.sgd`, the same optimizer object DP-SGD is given and not
`optax.sgd`, which exists and would compute the same thing — ADR-0002.

Two things follow from dropping stage 6. The loop carries one random
stream rather than DP-SGD's two, so a run is reproducible from one
seed; and `step` is a single function — ADR-0006.

Nothing is re-exported; import from the two modules.
"""

__all__: list[str] = []
