"""The two mechanisms and the two applies, against Algorithm 3.

The releases are where the mechanism is observable: by the time `step`
returns, four private means have been combined into one update and none
of them is recoverable from it.

Every draw here is built by hand from `dimma.core.sampling.dyadic` or
from explicit index arrays, so a failure points at this chunk rather
than at the sampler.
"""

from __future__ import annotations

import itertools
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.algorithms.bias_reduced_sgd import estimators, step as brs_step
from dimma.core import aggregation, clipping, gradients, updates
from dimma.core.sampling import dyadic

from ...helpers import tree_allclose
from .conftest import D, N, S, squared_error

EPS32 = float(np.finfo(np.float32).eps)


@pytest.fixture
def grad_fn():
    return gradients.per_sample_grads(squared_error)


def identity_estimator(clip_norm: float = 1e6) -> estimators.MeanEstimator:
    """No noise and a ball nothing reaches: `estimate` is the identity.

    The degenerate instantiation of the seam, and the one every exact
    identity below is checked through — what it isolates is the
    debiasing arithmetic, with the mechanism switched off.
    """
    return estimators.projection_estimator(
        clip_norm=clip_norm, radius=1e9, noise_multiplier=0.0
    )


def logging_estimator(log: list) -> estimators.MeanEstimator:
    """A stub recording the batch size of every slot it is called for."""

    def estimate(mean, key, batch_size):
        del key
        log.append(batch_size)
        return mean

    return estimators.MeanEstimator(
        name="stub",
        claim=estimators.GaussianMeanClaim(
            clip_norm=1e6, noise_multiplier=0.0
        ),
        estimate=estimate,
    )


def rows(x, y, indices):
    """The gathered rows, as `train` will hand them to a release."""
    indices = np.asarray(indices, dtype=np.int64)
    return x[indices], y[indices]


def mean_of(grad_fn, params, x, y, indices, clip_norm, divisor):
    """The clipped per-sample gradient mean over ``indices``, by hand."""
    x_rows, y_rows = rows(x, y, indices)
    clipped = clipping.per_sample_clip(
        grad_fn(params, x_rows, y_rows), clip_norm
    )
    return aggregation.average_over_batch(clipped, divisor)


def test_the_batch_release_is_three_means_of_one_draw(grad_fn, sparse_problem,
                                                      moved_params, key, rng):
    """The three quantities Algorithm 3 releases from ``B``, at zero
    noise: the mean over the whole batch and the mean over each half."""
    x, y, _ = sparse_problem
    estimator = identity_estimator()
    draw = dyadic.subsample(rng, N, scale=3)
    x_batch, y_batch = rows(x, y, draw.whole)

    release = brs_step.batch_release(
        grad_fn, estimator, moved_params, x_batch, y_batch, key,
        batch_size=16,
    )
    clip = estimator.claim.clip_norm
    assert tree_allclose(release.whole, mean_of(
        grad_fn, moved_params, x, y, draw.whole, clip, 16), atol=1e-6)
    assert tree_allclose(release.odd, mean_of(
        grad_fn, moved_params, x, y, draw.odd, clip, 8), atol=1e-6)
    assert tree_allclose(release.even, mean_of(
        grad_fn, moved_params, x, y, draw.even, clip, 8), atol=1e-6)


def test_gradients_computed_once_and_sliced_equal_three_separate_calls(
        grad_fn, sparse_problem, moved_params, key, rng):
    """The one optimization in the release, pinned as arithmetic.

    ``O`` and ``E`` are disjoint halves of ``B`` and every example is
    differentiated at the same parameters, so differentiating ``B``
    once and slicing is the same computation as three calls — at a
    third of the cost. Checked with a *binding* clip, since clipping is
    the one stage that could couple the rows if it were done wrong.
    """
    x, y, _ = sparse_problem
    clip = 0.05
    estimator = identity_estimator(clip_norm=clip)
    draw = dyadic.subsample(rng, N, scale=2)
    x_batch, y_batch = rows(x, y, draw.whole)

    release = brs_step.batch_release(
        grad_fn, estimator, moved_params, x_batch, y_batch, key, batch_size=8,
    )
    unclipped = grad_fn(moved_params, x_batch, y_batch)
    assert float(jnp.max(clipping.per_sample_norms(unclipped))) > clip

    for got, indices, divisor in [(release.whole, draw.whole, 8),
                                  (release.odd, draw.odd, 4),
                                  (release.even, draw.even, 4)]:
        separate = mean_of(grad_fn, moved_params, x, y, indices, clip, divisor)
        assert tree_allclose(got, separate, atol=1e-7)


def test_at_zero_noise_the_whole_is_the_mean_of_the_halves(
        grad_fn, sparse_problem, moved_params, key, rng):
    """The exact-halves identity the debiasing rests on.

    ``|O| = |E|``, so the mean over ``B`` is the average of the two
    half-means with no weights. If the halves were ever unequal — a
    truncated draw, an odd batch — the bracket would stop cancelling in
    expectation and the estimator would be biased rather than noisy.
    """
    x, y, _ = sparse_problem
    draw = dyadic.subsample(rng, N, scale=4)
    x_batch, y_batch = rows(x, y, draw.whole)
    release = brs_step.batch_release(
        grad_fn, identity_estimator(), moved_params, x_batch, y_batch, key,
        batch_size=32,
    )
    assert tree_allclose(
        release.whole,
        jax.tree.map(lambda o, e: 0.5 * (o + e), release.odd, release.even),
        atol=1e-6,
    )


def test_the_halves_are_divided_by_half_the_batch_size(
        grad_fn, sparse_problem, moved_params, key, rng):
    """``2 ** N``, not ``2 ** (N + 1)``. Dividing both by the batch size
    is the mistake this catches: it halves the bracket rather than
    cancelling it, and the run would still look plausible."""
    x, y, _ = sparse_problem
    draw = dyadic.subsample(rng, N, scale=2)
    x_batch, y_batch = rows(x, y, draw.whole)
    release = brs_step.batch_release(
        grad_fn, identity_estimator(), moved_params, x_batch, y_batch, key,
        batch_size=8,
    )
    clip = 1e6
    assert tree_allclose(
        release.odd, mean_of(grad_fn, moved_params, x, y, draw.odd, clip, 4),
        atol=1e-7,
    )
    over_the_whole = mean_of(
        grad_fn, moved_params, x, y, draw.odd, clip, 8)
    assert not tree_allclose(release.odd, over_the_whole, atol=1e-4)


def test_the_three_inner_calls_get_independent_noise(grad_fn):
    """One key in, three independent perturbations out.

    On a batch whose rows are identical the three means coincide, so
    anything separating the three releases is the noise. Reusing one
    key for all three would make them equal here, and would make an
    accountant's basic composition of three releases describe one.
    """
    x = jnp.tile(jnp.array([[1.0, -2.0, 0.5]]), (8, 1))
    y = jnp.full((8,), 0.75)
    params = {"w": jnp.zeros(3)}
    estimator = estimators.projection_estimator(
        clip_norm=10.0, radius=1e9, noise_multiplier=1.0
    )
    release = brs_step.batch_release(
        grad_fn, estimator, params, x, y, jax.random.key(0), batch_size=8)
    assert not tree_allclose(release.whole, release.odd, atol=1e-3)
    assert not tree_allclose(release.odd, release.even, atol=1e-3)

    draws = jax.vmap(lambda k: brs_step.batch_release(
        grad_fn, estimator, params, x, y, k, batch_size=8))(
            jax.random.split(jax.random.key(1), 2000))
    whole = np.asarray(draws.whole["w"])
    odd = np.asarray(draws.odd["w"])
    even = np.asarray(draws.even["w"])
    for a, b in [(whole, odd), (whole, even), (odd, even)]:
        correlation = np.corrcoef(a[:, 0] - a[:, 0].mean(),
                                  b[:, 0] - b[:, 0].mean())[0, 1]
        assert abs(correlation) < 0.08


def test_every_per_example_gradient_is_clipped_to_the_claims_norm(
        grad_fn, sparse_problem, moved_params, key, rng):
    """Stage 4 is present here, unlike in SpiderBoost: ``L`` is enforced
    rather than assumed (ADR-0012), and the number enforced is the one
    the claim carries, so the bound and the calibration cannot differ.
    """
    x, y, _ = sparse_problem
    clip = 0.02
    estimator = identity_estimator(clip_norm=clip)
    draw = dyadic.subsample(rng, N, scale=3)
    x_batch, y_batch = rows(x, y, draw.whole)

    release = brs_step.batch_release(
        grad_fn, estimator, moved_params, x_batch, y_batch, key,
        batch_size=16)
    unclipped = aggregation.average_over_batch(
        grad_fn(moved_params, x_batch, y_batch), 16)
    assert float(jnp.linalg.norm(unclipped["w"])) > clip
    for released in (release.whole, release.odd, release.even):
        assert float(jnp.linalg.norm(released["w"])) <= clip + 1e-6

    single = brs_step.single_release(
        grad_fn, estimator, moved_params, x[:1], y[:1], key)
    assert float(jnp.linalg.norm(single["w"])) <= clip + 1e-6


def test_the_combine_is_the_papers_return_line():
    """``G = (1/p_N)[G+ - 0.5 (G-_O + G-_E)] + G_0``, against a hand
    computation on releases that are numbers rather than gradients."""
    release = brs_step.BatchRelease(
        whole={"w": jnp.array([1.0, -2.0])},
        odd={"w": jnp.array([0.5, -1.0])},
        even={"w": jnp.array([-0.5, 3.0])},
    )
    single = {"w": jnp.array([0.25, 0.75])}
    combined = brs_step.debiased_gradient(
        release, single, scale_probability=0.25)
    bracket = np.array([1.0, -2.0]) - 0.5 * (np.array([0.5, -1.0])
                                             + np.array([-0.5, 3.0]))
    want = 4.0 * bracket + np.array([0.25, 0.75])
    assert np.allclose(combined["w"], want, atol=0.0)


@pytest.mark.parametrize("scale_probability", [0.0, -0.5, 1.5])
def test_the_combine_rejects_a_weight_that_is_not_a_probability(
        scale_probability):
    """``1 / p_N`` is the reciprocal of the drawn scale's probability;
    anything outside ``(0, 1]`` is not one."""
    release = brs_step.BatchRelease(
        whole={"w": jnp.zeros(2)}, odd={"w": jnp.zeros(2)},
        even={"w": jnp.zeros(2)},
    )
    with pytest.raises(ValueError, match="scale_probability="):
        brs_step.debiased_gradient(
            release, {"w": jnp.zeros(2)},
            scale_probability=scale_probability)


@pytest.mark.parametrize("batch_size", [1, 3, 0, -2])
def test_a_batch_that_is_not_a_rung_of_the_ladder_is_rejected(
        grad_fn, moved_params, key, batch_size):
    """Every batch is ``2 ** (scale + 1)`` and splits into exact halves,
    so an odd size describes no draw the debiasing identity holds for."""
    x = jnp.zeros((4, D))
    y = jnp.zeros((4,))
    with pytest.raises(ValueError, match="batch_size="):
        brs_step.batch_release(
            grad_fn, identity_estimator(), moved_params, x, y, key,
            batch_size=batch_size)


def _exact_problem():
    """Eight examples whose gradients are exactly representable.

    Every value is a small dyadic rational, so each per-sample gradient,
    each batch mean over a power-of-two batch, and the combine are exact
    in float32 *and* in float64. That is what lets the enumeration below
    assert an identity rather than a tolerance.
    """
    x = jnp.asarray(np.array([
        [1.0, 0.0, 2.0], [0.0, -1.0, 1.0], [2.0, 2.0, 0.0], [-1.0, 0.0, 1.0],
        [0.0, 1.0, -2.0], [1.0, -2.0, 0.0], [-2.0, 0.0, -1.0], [0.0, 2.0, 1.0],
    ], dtype=np.float32))
    y = jnp.asarray(np.array(
        [0.5, -1.0, 1.5, 0.25, -0.5, 2.0, -0.25, 1.0], dtype=np.float32))
    params = {"w": jnp.asarray(np.array([0.5, -0.25, 0.75], np.float32))}
    return x, y, params


def _full_gradient(x, y, params):
    """``grad F_S(x)`` in float64 on the host, from the same numbers."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(params["w"], dtype=np.float64)
    residual = x64 @ w64 - y64
    return (residual[:, None] * x64).mean(axis=0)


def test_the_debiased_estimator_is_unbiased_under_the_identity_estimator():
    """Algorithm 3's whole point, as an identity rather than a bound.

    With the inner estimator switched off — no noise, a ball nothing
    reaches — Lemma 5.4's telescoping is exact: the bracket has mean
    zero at every scale because ``B``, ``O`` and ``E`` are each uniform
    draws of their own size, and ``G_0`` supplies the mean itself. So

        E[G(x)] = grad F_S(x)

    with no error term at all. On ``n = 8`` (hence ``M = 2``) every
    outcome is enumerable: three scales, every subset of the right
    size, every equal split of it, and every single record, each
    weighted by its own probability.

    The strongest pin available on the debiasing identity. It fails
    loudly if the halves are ever not an exact equal partition, if the
    weight is not ``1 / p_N``, or if ``G_0`` is dropped.
    """
    x, y, params = _exact_problem()
    n = 8
    assert dyadic.max_scale(n) == 2
    probabilities = dyadic.scale_probabilities(dyadic.max_scale(n))
    grad_fn = gradients.per_sample_grads(squared_error)
    estimator = identity_estimator()
    release = jax.jit(
        partial(brs_step.batch_release, grad_fn, estimator),
        static_argnames=("batch_size",),
    )
    single_release = jax.jit(
        partial(brs_step.single_release, grad_fn, estimator))
    key = jax.random.key(0)

    singles = [single_release(params, x[i:i + 1], y[i:i + 1], key)
               for i in range(n)]

    expectation = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for scale, probability in enumerate(probabilities):
        batch_size = 1 << (scale + 1)
        half = batch_size // 2
        subsets = list(itertools.combinations(range(n), batch_size))
        splits = list(itertools.combinations(range(batch_size), half))
        weight = float(probability) / (len(subsets) * len(splits) * n)
        for subset in subsets:
            for split in splits:
                odd = [subset[i] for i in split]
                even = [i for i in subset if i not in set(odd)]
                x_batch, y_batch = rows(x, y, odd + even)
                batch = release(params, x_batch, y_batch, key,
                                batch_size=batch_size)
                for single in singles:
                    combined = brs_step.debiased_gradient(
                        batch, single, scale_probability=float(probability))
                    expectation += weight * combined["w"]
                    total_weight += weight

    assert math.isclose(total_weight, 1.0, rel_tol=1e-12)
    assert np.allclose(expectation, _full_gradient(x, y, params), atol=1e-12)


def test_the_near_unbiasedness_survives_noise_and_projection(sparse_problem):
    """The same identity with the mechanism switched back on.

    Noise is zero-mean but the projection is not affine, so the exact
    identity becomes Lemma 5.4's bias bound. The telescoping leaves the
    bias of the *largest* slot alone, which is why a bound written at
    the full batch covers an estimator most of whose slots are tiny::

        ||E[G] - grad F_S||  <=  L (s ln d ln(1/delta))^(1/4)
                                 / sqrt(n eps)

    with the note's ``ln d`` where the paper writes ``ln(d/s)``.
    """
    x, y, _ = sparse_problem
    clip_norm, delta, draws = 1.0, 1e-5, 2000
    multiplier = 0.02
    epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) / multiplier
    grad_fn = gradients.per_sample_grads(squared_error)
    estimator = estimators.projection_estimator(
        clip_norm=clip_norm, radius=clip_norm * math.sqrt(S),
        noise_multiplier=multiplier,
    )
    release = jax.jit(partial(brs_step.batch_release, grad_fn, estimator),
                      static_argnames=("batch_size",))
    single_release = jax.jit(
        partial(brs_step.single_release, grad_fn, estimator))

    params = {"w": jnp.zeros(D)}
    rng = np.random.default_rng(11)
    keys = jax.random.split(jax.random.key(5), draws)
    probabilities = dyadic.scale_probabilities(dyadic.max_scale(N))

    total = np.zeros(D, dtype=np.float64)
    for index in range(draws):
        scale = dyadic.draw_scale(rng, dyadic.max_scale(N))
        draw = dyadic.subsample(rng, N, scale)
        batch_key, single_key = jax.random.split(keys[index])
        x_batch, y_batch = rows(x, y, draw.whole)
        x_one, y_one = rows(x, y, draw.single)
        total += brs_step.debiased_gradient(
            release(params, x_batch, y_batch, batch_key,
                    batch_size=1 << (scale + 1)),
            single_release(params, x_one, y_one, single_key),
            scale_probability=float(probabilities[scale]),
        )["w"]

    clipped = clipping.per_sample_clip(grad_fn(params, x, y), clip_norm)
    reference = np.asarray(
        aggregation.average_over_batch(clipped, N)["w"], dtype=np.float64)
    bias = float(np.linalg.norm(total / draws - reference))
    bound = (clip_norm
             * (S * math.log(D) * math.log(1.0 / delta)) ** 0.25
             / math.sqrt(N * epsilon))
    assert bias <= bound


def test_the_releases_are_float32_and_the_combine_is_float64(
        grad_fn, sparse_problem, moved_params, key):
    """Where the dtypes change, and where they do not.

    A release is float32 because that is what the device computed and
    what the mechanism made public. The combine amplifies a
    near-cancellation by ``2 ** (N + 1)``, so it runs in float64 on the
    host — post-processing, and free.
    """
    x, y, _ = sparse_problem
    estimator = identity_estimator()
    release = brs_step.batch_release(
        grad_fn, estimator, moved_params, *rows(x, y, np.arange(8)), key,
        batch_size=8)
    single = brs_step.single_release(
        grad_fn, estimator, moved_params, x[:1], y[:1], key)

    for released in (release.whole, release.odd, release.even, single):
        for leaf in jax.tree_util.tree_leaves(released):
            assert leaf.dtype == jnp.float32

    combined = brs_step.debiased_gradient(
        release, single, scale_probability=0.5)
    for leaf in jax.tree_util.tree_leaves(combined):
        assert isinstance(leaf, np.ndarray)
        assert leaf.dtype == np.float64


def _project_l1_ball_float64(v: np.ndarray, radius: float) -> np.ndarray:
    """Duchi et al. (2008) in float64, the reference the device's
    float32 projection is measured against."""
    if np.sum(np.abs(v)) <= radius:
        return v.copy()
    u = np.sort(np.abs(v))[::-1]
    cssv = np.cumsum(u)
    k = np.arange(1, v.size + 1, dtype=np.float64)
    rho = max(int(np.sum(u * k > (cssv - radius))), 1)
    theta = max((cssv[rho - 1] - radius) / rho, 0.0)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def _float64_reference(x, y, params, indices, key, *, clip_norm, radius,
                       multiplier, divisor):
    """One inner release, computed entirely in float64 on the host, with
    the very noise `jax.random` handed the float32 path."""
    x64 = np.asarray(x, dtype=np.float64)[np.asarray(indices)]
    y64 = np.asarray(y, dtype=np.float64)[np.asarray(indices)]
    w64 = np.asarray(params["w"], dtype=np.float64)
    grads = ((x64 @ w64 - y64)[:, None] * x64)
    norms = np.linalg.norm(grads, axis=1)
    grads *= np.minimum(1.0, clip_norm / (norms + 1e-12))[:, None]
    mean = grads.sum(axis=0) / divisor
    noise = np.asarray(
        jax.random.normal(jax.random.split(key, 1)[0], (w64.size,),
                          dtype=jnp.float32),
        dtype=np.float64,
    )
    scale = multiplier * 2.0 * clip_norm / divisor
    return _project_l1_ball_float64(mean + scale * noise, radius)


def test_the_combine_matches_a_float64_reference_up_to_the_scale_law(
        sparse_problem):
    """float64 on the apply side has a ceiling, and this is where it is.

    Each float32 release carries rounding of order ``eps32 * clip_norm``
    from its own sum, perturbation and projection, which converting to
    float64 afterwards cannot remove. The signal in
    ``G+ - 0.5 (G-_O + G-_E)`` cancels exactly while that rounding does
    not, and ``1 / p_N`` then multiplies what is left by about
    ``2 ** (N + 1)``::

        relative error of G  <=  2 ** (N + 1) * eps32
                                 / (4 * noise_multiplier)

    A law over the whole ladder rather than a flat tolerance, because a
    flat tolerance would hide exactly the regime dependence that
    matters: at the top of a full-size ladder this reaches order one,
    which is the ceiling the package docstring states and the reason
    `max_scale` interacts with float32 at all.

    Measured against a reference that repeats the *whole* step in host
    float64 — gradients, clipping, means, projection — on the very noise
    `jax.random` handed the float32 path, so the difference is rounding
    and nothing else. Run at a small multiplier, where a release is
    dominated by its mean rather than by its noise: that is the regime
    the rounding floor ``eps32 * clip_norm`` describes, and the one in
    which the amplification is visible rather than divided out.
    """
    x, y, _ = sparse_problem
    clip_norm, radius, multiplier = 1.0, float(math.sqrt(S)), 1e-3
    params = {"w": jnp.zeros(D)}
    grad_fn = gradients.per_sample_grads(squared_error)
    estimator = estimators.projection_estimator(
        clip_norm=clip_norm, radius=radius, noise_multiplier=multiplier)
    probabilities = dyadic.scale_probabilities(dyadic.max_scale(N))
    rng = np.random.default_rng(4)

    deviations = []
    for scale in range(dyadic.max_scale(N) + 1):
        batch_size = 1 << (scale + 1)
        half = batch_size // 2
        probability = float(probabilities[scale])
        measured = []
        for trial in range(4):
            draw = dyadic.subsample(rng, N, scale)
            batch_key, single_key = jax.random.split(
                jax.random.key(100 * scale + trial))
            got = brs_step.debiased_gradient(
                brs_step.batch_release(
                    grad_fn, estimator, params, *rows(x, y, draw.whole),
                    batch_key, batch_size=batch_size),
                brs_step.single_release(
                    grad_fn, estimator, params, *rows(x, y, draw.single),
                    single_key),
                scale_probability=probability,
            )["w"]

            whole_key, odd_key, even_key = jax.random.split(batch_key, 3)
            reference = partial(
                _float64_reference, x, y, params, clip_norm=clip_norm,
                radius=radius, multiplier=multiplier)
            bracket = (
                reference(draw.whole, whole_key, divisor=batch_size)
                - 0.5 * (reference(draw.odd, odd_key, divisor=half)
                         + reference(draw.even, even_key, divisor=half))
            )
            want = (bracket / probability
                    + reference(draw.single, single_key, divisor=1))
            measured.append(
                float(np.linalg.norm(got - want) / np.linalg.norm(want)))

        deviation = float(np.mean(measured))
        deviations.append(deviation)
        assert deviation <= batch_size * EPS32 / (4.0 * multiplier)
        assert deviation <= 4.0 * batch_size * EPS32

    # And the law is not a formality: the deviation grows with the
    # ladder, by about the 2 ** (N + 1) the debias weight multiplies by.
    assert deviations[-1] >= 8.0 * deviations[0]


def test_the_estimator_seam_records_four_calls_with_the_papers_batch_sizes(
        grad_fn, sparse_problem, moved_params, key, rng):
    """Four slots a step, at ``2 ** (N + 1)``, ``2 ** N``, ``2 ** N``,
    ``1`` — Algorithm 3's own cardinalities, read off a stub estimator
    rather than inferred from the arithmetic."""
    x, y, _ = sparse_problem
    log: list = []
    estimator = logging_estimator(log)
    draw = dyadic.subsample(rng, N, scale=3)
    brs_step.batch_release(
        grad_fn, estimator, moved_params, *rows(x, y, draw.whole), key,
        batch_size=16)
    brs_step.single_release(
        grad_fn, estimator, moved_params, *rows(x, y, draw.single), key)
    assert log == [16, 8, 8, 1]


def test_a_scale_of_zero_uses_a_batch_of_two_and_halves_of_one(
        grad_fn, sparse_problem, moved_params, key, rng):
    """The bottom rung, which about half of all steps draw. Both halves
    are a single example, and nothing special-cases them."""
    x, y, _ = sparse_problem
    log: list = []
    draw = dyadic.subsample(rng, N, scale=0)
    assert draw.whole.shape == (2,)
    brs_step.batch_release(
        grad_fn, logging_estimator(log), moved_params,
        *rows(x, y, draw.whole), key, batch_size=2)
    assert log == [2, 1, 1]


def test_each_scale_traces_once(sparse_problem, moved_params):
    """The jit strategy, as a count.

    Per-shape compilation and no padding cap: a scale costs one trace
    the first time it is drawn and none afterwards, so a whole run
    compiles at most ``max_scale + 1`` batch programs, and the largest
    only if it is ever drawn at all. The single release has one shape
    for the run and traces once, which is why it is bound separately
    from the per-scale table.
    """
    x, y, _ = sparse_problem
    batch_traces = single_traces = 0

    def counted_batch_loss(params, x_one, y_one):
        nonlocal batch_traces
        batch_traces += 1
        return squared_error(params, x_one, y_one)

    def counted_single_loss(params, x_one, y_one):
        nonlocal single_traces
        single_traces += 1
        return squared_error(params, x_one, y_one)

    estimator = identity_estimator()
    releases = brs_step.Releases(
        batch=jax.jit(
            partial(brs_step.batch_release,
                    gradients.per_sample_grads(counted_batch_loss), estimator),
            static_argnames=("batch_size",)),
        single=jax.jit(
            partial(brs_step.single_release,
                    gradients.per_sample_grads(counted_single_loss),
                    estimator)),
    )

    key = jax.random.key(0)
    for _ in range(3):
        releases.batch(moved_params, *rows(x, y, np.arange(4)), key,
                       batch_size=4)
        releases.single(moved_params, x[:1], y[:1], key)
    assert (batch_traces, single_traces) == (1, 1)

    for _ in range(3):
        releases.batch(moved_params, *rows(x, y, np.arange(16)), key,
                       batch_size=16)
        releases.single(moved_params, x[1:2], y[1:2], key)
    assert (batch_traces, single_traces) == (2, 1)


def test_the_step_descends_along_the_debiased_gradient(sparse_problem):
    """The apply side end to end, and the float64 it runs in.

    ``x^{t+1} = x^t - eta G(x^t)``, with ``params`` a host float64
    pytree converted to float32 for the release calls and left in
    float64 for the update — which `updates.sgd` at a constant rate
    supports, being dtype-agnostic pytree arithmetic.
    """
    x, y, _ = sparse_problem
    grad_fn = gradients.per_sample_grads(squared_error)
    estimator = identity_estimator()
    releases = brs_step.Releases(
        batch=jax.jit(partial(brs_step.batch_release, grad_fn, estimator),
                      static_argnames=("batch_size",)),
        single=jax.jit(partial(brs_step.single_release, grad_fn, estimator)),
    )
    learning_rate = 0.05
    optimizer = updates.sgd(learning_rate)
    params = {"w": np.zeros(D, dtype=np.float64)}
    batch_key, single_key = jax.random.split(jax.random.key(3))

    x_batch, y_batch = rows(x, y, np.arange(8))
    stepped, state = brs_step.step(
        releases, optimizer, params, updates.init(optimizer, params),
        x_batch, y_batch, x[:1], y[:1], batch_key, single_key,
        batch_size=8, scale_probability=0.5,
    )
    estimate = brs_step.debiased_gradient(
        releases.batch(params, x_batch, y_batch, batch_key, batch_size=8),
        releases.single(params, x[:1], y[:1], single_key),
        scale_probability=0.5,
    )
    assert stepped["w"].dtype == np.float64
    assert np.allclose(stepped["w"], -learning_rate * estimate["w"], atol=0.0)
    assert int(state.count) == 1
