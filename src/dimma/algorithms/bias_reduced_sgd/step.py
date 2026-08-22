"""One bias-reduced iteration: Algorithm 3, and the update it feeds.

Stage 1 is not here. The scale is drawn, the batch and its halves and
the single record are drawn at that scale, and the rows are gathered,
all on the host in `dimma.core.sampling.dyadic`; this module takes the
gathered rows. From the per-sample gradients to the perturbation
everything compiles.

Two mechanisms, so four functions, per ADR-0006. `batch_release` and
`single_release` return what their mechanism makes public, and so all
an accountant accounts for; `debiased_gradient` and `step` apply what
they returned, which is post-processing and free. The apply side is
split by post-processing layer rather than by mechanism, because the
two releases are combined into one estimate before anything is applied
at all — there is no "apply the batch release" that could stand alone.

The triplet is one release, not three. The whole batch and its two
halves are three quantities out of a *single* draw of ``B``, so they
are amplified once, jointly, at rate ``2 ** (N + 1) / n``, which is
what Lemma 5.3 states. Returning them in one `BatchRelease` is how the
code says so; three release functions would invite an accountant to
compose three independent amplifications, which is not what ran.

A step is two compiled calls plus a host-side combine rather than one
compiled call, which bends ADR-0006. The batch release has a different
shape at every scale while the single release has one shape for the
whole run, so binding them separately is what keeps the constant shape
from being retraced once per scale; and the combine runs in host
`numpy` float64, for the reason below.

Numerics. The releases are float32, as the device computed them. The
combine is float64 on the host, because ``G+ - 0.5 * (G-_O + G-_E)``
is a near-cancellation that ``1 / p_N`` then multiplies by about
``2 ** (N + 1)``. Doing it in float64 stops the combine from adding
rounding of its own; it cannot remove the rounding already inside each
float32 release, which the package docstring states as a ceiling and a
test pins as a scale law.

Binding. `grad_fn`, `estimator` and `optimizer` are *static*
`jax.jit` arguments rather than traced ones — one is a function, one
holds a function, the third is a pair of them. Bind them once outside
the loop and let `jax.jit`'s own cache be the per-scale table of
compiled programs::

    from functools import partial

    grad_fn = gradients.per_sample_grads(loss)
    estimator = estimators.projection_estimator(
        clip_norm=1.0, radius=5.0, noise_multiplier=3.2)
    releases = step.Releases(
        batch=jax.jit(partial(step.batch_release, grad_fn, estimator),
                      static_argnames=("batch_size",)),
        single=jax.jit(partial(step.single_release, grad_fn, estimator)),
    )

``batch_size`` is static, so the batch release is traced once per size
that comes up — at most ``max_scale + 1`` programs, populated lazily,
and the large ones only if their scale is ever drawn. The single
release has one shape and is traced once.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dimma.algorithms.bias_reduced_sgd import estimators
from dimma.core import aggregation, clipping, pytree, updates

__all__ = [
    "BatchRelease",
    "Releases",
    "batch_release",
    "single_release",
    "debiased_gradient",
    "step",
]


class BatchRelease(NamedTuple):
    """Everything the jointly-subsampled mechanism makes public.

    One object because it is one mechanism: the three inner calls run
    on a single draw of ``B``, are amplified once at rate
    ``2 ** (N + 1) / n``, and are basic-composed with each other
    conditionally on that draw.

    Leaves are float32, the dtype they were released in. Converting
    them is `debiased_gradient`'s job and nobody else's.
    """

    whole: Any
    """``G+_{N+1}(x, B)``, the private mean over the whole batch."""

    odd: Any
    """``G-_N(x, O)``, the private mean over the first half."""

    even: Any
    """``G-_N(x, E)``, the private mean over the second half."""


class Releases(NamedTuple):
    """The two compiled release callables, bound once outside the loop.

    Holds functions, so it is a static `jax.jit` argument and never a
    traced one. The module docstring shows how to build it.
    """

    batch: Callable
    """``(params, x_batch, y_batch, key, *, batch_size) -> BatchRelease``."""

    single: Callable
    """``(params, x_single, y_single, key) -> pytree``."""


def _rows(per_sample_pytree: Any, start: int, stop: int) -> Any:
    """Rows ``start:stop`` of every per-sample leaf."""
    return jax.tree.map(lambda leaf: leaf[start:stop], per_sample_pytree)


def batch_release(
    grad_fn: Callable,
    estimator: estimators.MeanEstimator,
    params: Any,
    x_batch: jax.Array,
    y_batch: jax.Array,
    key: jax.Array,
    *,
    batch_size: int,
) -> BatchRelease:
    """Algorithm 3's three batch lines, on one draw::

        G+_{N+1} = Estimator(mean over B, |B| = 2 ** (N + 1))
        G-_N     = Estimator(mean over O, |O| = 2 ** N)
        G-_N     = Estimator(mean over E, |E| = 2 ** N)

    Stages 3 through 6, over ``B`` once. The per-sample gradients are
    computed for the whole batch and the halves are slices of them,
    which is identical arithmetic to three separate calls — ``O`` and
    ``E`` are disjoint halves of ``B``, and every example is
    differentiated at the same parameters either way — for a third of
    the gradient work.

    Each inner call gets its own perturbation, from
    ``jax.random.split(key, 3)``. Three releases of one mechanism, not
    one release read three times: Lemma 5.3 composes them at
    ``(eps/32, delta/16)`` each before amplifying the composition.

    Parameters
    ----------
    grad_fn
        From `dimma.core.gradients.per_sample_grads`. Built once, so
        `jax.jit` sees a stable compilation key.
    estimator
        The inner mean estimator. Its ``claim.clip_norm`` is the norm
        stage 4 enforces here, and it appears nowhere else in this
        signature, so the bound and the noise calibrated against it
        cannot disagree.
    x_batch, y_batch
        The gathered rows of ``B``, leading axis exactly
        ``batch_size``. Nothing is padded and there is no mask.
    batch_size
        ``2 ** (N + 1)``, the exact cardinality the coin fixed before
        any data was touched — CONTEXT.md's expected batch size in the
        form the glossary admits. Never read off ``x_batch.shape``: a
        divisor taken from the array agrees with it by construction,
        and would hide a mismatch rather than expose it. Static, so it
        drives compilation along with the axis it equals.

    Returns
    -------
    BatchRelease
        Three private means, one draw.

    Raises
    ------
    ValueError
        If ``batch_size`` is not an even number of at least two. It is
        ``2 ** (scale + 1)`` and splits into halves of ``2 ** scale``,
        so an odd size describes no draw on the ladder.
    """
    if batch_size < 2 or batch_size % 2:
        raise ValueError(
            f"batch_size={batch_size} must be even and at least 2; it "
            f"is 2 ** (scale + 1), and it splits into two exact halves "
            f"of 2 ** scale."
        )
    half = batch_size // 2
    clipped = clipping.per_sample_clip(
        grad_fn(params, x_batch, y_batch), estimator.claim.clip_norm
    )
    whole_key, odd_key, even_key = jax.random.split(key, 3)
    whole = estimator.estimate(
        aggregation.average_over_batch(clipped, batch_size),
        whole_key,
        batch_size,
    )
    odd = estimator.estimate(
        aggregation.average_over_batch(_rows(clipped, 0, half), half),
        odd_key,
        half,
    )
    even = estimator.estimate(
        aggregation.average_over_batch(_rows(clipped, half, batch_size), half),
        even_key,
        half,
    )
    return BatchRelease(whole=whole, odd=odd, even=even)


def single_release(
    grad_fn: Callable,
    estimator: estimators.MeanEstimator,
    params: Any,
    x_single: jax.Array,
    y_single: jax.Array,
    key: jax.Array,
) -> Any:
    """Algorithm 3's ``G_0``: the private gradient at one record.

    The second mechanism, and accounted separately. Its record is drawn
    uniformly and independently of ``B``, so it is amplified at rate
    ``1 / n`` rather than at the batch's rate, and Lemma 5.3 adds its
    cost to the triplet's rather than folding it in.

    Parameters
    ----------
    x_single, y_single
        One gathered row each, leading axis 1. That shape is the same
        at every scale, which is why this release compiles once for a
        whole run.

    Notes
    -----
    A batch of one is not a special case. The sensitivity
    ``2 * clip_norm / batch_size`` is simply largest here and the noise
    scale follows it, with no branch anywhere in the estimator.
    """
    clipped = clipping.per_sample_clip(
        grad_fn(params, x_single, y_single), estimator.claim.clip_norm
    )
    return estimator.estimate(
        aggregation.average_over_batch(clipped, 1), key, 1
    )


def _as_host_float64(tree: Any) -> Any:
    """Every floating leaf as a host `numpy` float64 array; others as-is."""

    def _convert(leaf):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float64)
        return array

    return jax.tree.map(_convert, tree)


def _as_device_float32(tree: Any) -> Any:
    """Every floating leaf as a device float32 array; others as they are."""

    def _convert(leaf):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.floating):
            return array.astype(jnp.float32)
        return array

    return jax.tree.map(_convert, tree)


def debiased_gradient(
    batch: BatchRelease,
    single: Any,
    *,
    scale_probability: float,
) -> Any:
    """Algorithm 3's Return line, in float64 on the host::

        G(x) = (1 / p_N) * [G+ - 0.5 * (G-_O + G-_E)] + G_0

    Post-processing of four already-released quantities, so it costs
    nothing an accountant sees and is free to run wherever the
    arithmetic is best — which is not the device. The bracket nearly
    cancels and ``1 / p_N`` then multiplies it by about
    ``2 ** (N + 1)``, so combining in float32 would amplify the
    combine's own rounding along with the releases'.

    Parameters
    ----------
    batch, single
        The two releases as returned: device float32 arrays, or
        anything `numpy` converts.
    scale_probability
        ``p_N``, read from
        `dimma.core.sampling.dyadic.scale_probabilities` at the scale
        that was drawn, rather than recomputed here — the debias weight
        and the coin then cannot disagree about the law.

    Returns
    -------
    Any
        A pytree of host `numpy` float64 arrays, in the releases'
        structure.

    Raises
    ------
    ValueError
        If ``scale_probability`` is outside ``(0, 1]``. It is a
        probability whose reciprocal is the debias weight.
    """
    if not 0.0 < scale_probability <= 1.0:
        raise ValueError(
            f"scale_probability={scale_probability} must lie in "
            f"(0, 1]; it is p_N, the probability of the scale that was "
            f"drawn, and the debias weight is its reciprocal."
        )
    whole = _as_host_float64(batch.whole)
    halves = pytree.scale(
        pytree.add(_as_host_float64(batch.odd), _as_host_float64(batch.even)),
        0.5,
    )
    debiased = pytree.scale(
        pytree.sub(whole, halves), 1.0 / scale_probability
    )
    return pytree.add(debiased, _as_host_float64(single))


def step(
    releases: Releases,
    optimizer: updates.Optimizer,
    params: Any,
    opt_state: updates.OptState,
    x_batch: jax.Array,
    y_batch: jax.Array,
    x_single: jax.Array,
    y_single: jax.Array,
    batch_key: jax.Array,
    single_key: jax.Array,
    *,
    batch_size: int,
    scale_probability: float,
) -> tuple[Any, updates.OptState]:
    """One full iteration: release twice, combine, descend.

    Algorithm 4's ``x^{t+1} = Pi_X(x^t - eta G(x^t))`` without the
    ``Pi_X``: a caller-side projection composes at the optimizer seam
    per ADR-0014, so it is `dimma.transforms.projection.l1_projected`
    wrapped around ``optimizer``, not an argument here.

    Not itself compiled. Both releases already are, each bound outside
    the loop; what remains is the host-side combine and the update,
    which run in float64 over `numpy` leaves. ``params`` may therefore
    be a float64 `numpy` pytree: it is converted to float32 for the
    release calls and left alone for the update.

    Parameters
    ----------
    releases
        The two compiled release callables.
    batch_size, scale_probability
        ``2 ** (N + 1)`` and ``p_N`` for the scale this step drew. Both
        come from the coin, which is public and touches no data — which
        is what lets the accountant price the step before it runs.

    Returns
    -------
    tuple
        ``(params, opt_state)``, and nothing else. The releases are the
        step's whole public output and were made public already by the
        mechanisms that produced them.

    Notes
    -----
    The float64 apply side survives a dtype-agnostic optimizer such as
    `dimma.core.updates.sgd` at a constant learning rate. A schedule
    returns a `jax.numpy` scalar and an optax transformation runs
    `jax.numpy` operations, either of which pulls the update back down
    to float32.
    """
    device_params = _as_device_float32(params)
    batch = releases.batch(
        device_params, x_batch, y_batch, batch_key, batch_size=batch_size
    )
    single = releases.single(device_params, x_single, y_single, single_key)
    estimate = debiased_gradient(
        batch, single, scale_probability=scale_probability
    )
    return updates.apply(optimizer, params, estimate, opt_state)
