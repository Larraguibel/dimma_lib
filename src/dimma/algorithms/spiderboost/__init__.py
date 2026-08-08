"""Private SpiderBoost (Arora et al., ICML 2023), Algorithm 2.

A variance-reduced private method, and the first non-classical
algorithm here. It composes *two* mechanisms rather than one. Every
``anchor_interval`` steps an **anchor** step draws a batch and releases
a privatized gradient. Every other step is a **variation** step, which
draws a batch, evaluates per-sample gradients at the current and the
previous parameters, and releases the privatized difference; the
running estimate is the previous estimate plus that release, and the
update descends along it.

Algorithm 2 in the paper's notation, against the parameter that carries
it. The paper's symbols appear here and nowhere else in the package, so
a signature can be read without the paper open:

=====================================  ==============================
Algorithm 2                            this package
=====================================  ==============================
``T``, iterations                      ``steps - 1``
``q``, phase size                      ``anchor_interval``
``b_1``, ``b_2``, batch sizes          ``anchor_expected_batch_size``,
                                       ``variation_expected_batch_size``
``eta``, learning rate                 the `Optimizer`
``sigma_1``                            ``anchor_noise_scale``
``sigma_2``                            ``variation_noise_rate``
``sigma-hat_2``                        ``variation_noise_cap``
``w_t``                                ``params``
``w_{t-1}``                            ``previous_params``
``nabla_t``                            the running estimate
``Delta_t``                            a variation release
``w-bar``                              the output-rule iterate
``L_0``, ``L_1``                       absent; see below
=====================================  ==============================

and line by line against the primitive that implements it:

=====================================  ==============================
Algorithm 2                            `dimma.core`
=====================================  ==============================
``Sample batch S_t of size b_1``       `sampling.poisson`
``grad f(w_t; x)``                     `gradients.per_sample_grads`
*(no clipping line)*                   stage 4 is absent
``[grad f(w_t;x) - grad f(w_{t-1};x)]``  `pytree.sub`
``(1/b) sum_{x in S_t} ...``           `aggregation.average_over_batch`
``... + g_t``                          `noise.add_gaussian`
``||w_t - w_{t-1}||``                  `pytree.global_norm`
``nabla_t = nabla_{t-1} + Delta_t``    `pytree.add`
``w_{t+1} = w_t - eta nabla_t``        `updates.apply`
=====================================  ==============================

The first line is stage 1 and lives in
:mod:`~dimma.algorithms.spiderboost.train`, because a Poisson draw has
data-dependent cardinality and cannot be compiled. The rest is
:mod:`~dimma.algorithms.spiderboost.step`.

Note where the noise goes. Lines 9 and 13 add ``g_t`` to a quantity
already divided by the batch size, so the noise scales named above are
scales on the *released estimate*. Classical DP-SGD noises the sum
instead, at ``sigma * C``, because there the sensitivity bound belongs
to the sum. Both are written the way their paper writes them.

Our words, not the paper's
--------------------------
The paper has no name for either branch: it writes ``mod(t, q) = 0``
and an ``else``. **Anchor** and **variation** are ours, chosen because
the branches are two mechanisms and an accountant has to name them
apart. ``anchor_interval`` is the paper's *phase size* ``q``, renamed
because `dimma`'s glossary reserves *phase* against *stage*. *Release*,
*expected batch size* and *padding cap* are the glossary's, not the
paper's. Do not look for any of them in Algorithm 2.

The absent stage
----------------
There is no clipping, and none was added. The privacy rests on the
function class the paper assumes — ``f(.; x)`` is ``L_0``-Lipschitz and
``L_1``-smooth — which bounds a per-sample gradient by ``L_0`` and a
per-sample gradient difference by ``min(L_1 ||w_t - w_{t-1}||, 2 L_0)``
without any operation making it so. Stage 4 is therefore absent by
decision, not by oversight; ADR-0009 and ADR-0001. ``L_0`` and ``L_1``
appear nowhere in this package: they are the accountant's inputs, and
the loop takes noise scales.

Sampling departs from the paper
-------------------------------
Lines 7 and 11 draw a batch of *fixed* size. This package draws by
Poisson subsampling at rates ``anchor_expected_batch_size / n`` and
``variation_expected_batch_size / n``, which is what `dimma.core`
implements, what DP-SGD uses, and what any accountant plugged in will
assume. The estimates are divided by the expected batch size, fixed
before the run, and never by the length of the draw. An oversize draw
raises rather than truncating; ADR-0007.

The output rule and its support
-------------------------------
Line 18 returns ``w-bar`` uniformly from ``{w_1, ..., w_T}``, while the
loop on line 5 runs ``t = 0, ..., T`` and so produces ``w_1, ...,
w_{T+1}``. The last iterate produced is excluded. Both are correct as
printed, and neither is an off-by-one:

- a gradient estimate is released at every ``w_t`` the loop visits,
  that is at ``w_0`` through ``w_T``;
- ``w_{T+1}`` is the one iterate at which no estimate was ever taken,
  so the convergence proof's final inequality has nothing to say about
  it.

Two plausible-looking repairs each break that correspondence in one
direction: extending the support to ``w_{T+1}`` returns an iterate no
bound covers, and shortening the loop by one removes the release that
produced the last iterate the bound does cover.

``steps`` here is the number of optimizer updates, which for this
algorithm is also the number of releases, so ``steps = T + 1``. The
support is then ``{w_1, ..., w_{steps-1}}`` — every iterate the loop
produced except the last. `train` returns that iterate first and the
final parameters second; the final parameters carry no stationarity
bound and are not the algorithm's result.

Preconditions this package cannot check
---------------------------------------
Stated out loud because a run violating them fails silently, with a
number rather than a crash:

- the supplied per-sample loss is ``L_0``-Lipschitz and ``L_1``-smooth
  in the parameters, for every example, with the same constants used to
  calibrate the noise scales passed in;
- the optimizer is descent at the constant step size Theorem B.3 fixes
  at ``1/(2 L_1)``. Any other rule leaves the privacy intact but
  removes the convergence guarantee, and additionally changes how far
  the parameters move each step, which feeds the variation branch's
  noise scale — so it changes what the mechanism does, not only what
  can be proved about it.

Nothing is re-exported; import from `step` and `train`.
"""

__all__: list[str] = []
