# dimma

A library for differentially private optimization, where every method is
expressed as the same fixed sequence of stages so that two methods — and a
method against its privacy-free counterpart — can be compared rather than
merely described.

## What belongs here

Only language shared across algorithms. Two tests, and a term needs both.

*Delete any one algorithm — does the term still have a referent?* If the
concept dies with the algorithm, it was that algorithm's, not the project's.

*Who else says it?* A term earns a place when at least two algorithms, or an
algorithm and a layer, would both reach for it. One user is a paper's notation
wearing general clothes.

A paper's own symbols and names stay with the algorithm that implements it, in
its package docstring, mapped to the general term. That is the right home for
them, not a demotion: `dimma/algorithms/dp_sgd/__init__.py` tabulates Abadi et
al.'s notation against the primitives it calls, and a reader of that algorithm
wants the paper's words.

A word already carried by a public API means what the API means. The glossary
sharpens vocabulary; it does not overrule code that a caller has to type.

## Language

### The pipeline

**Pipeline**:
The fixed seven-stage sequence every method in dimma is expressed as.
_Avoid_: training loop, workflow, recipe

**Stage**:
One of the seven positions in the pipeline. A method is a choice at each
stage, including the choice to leave one out.
_Avoid_: phase, layer, step

**Step**:
One optimizer update. The unit privacy composes over, and so the unit a run's
length is measured in.
_Avoid_: iteration, epoch, round

**Batch**:
The examples one step draws. Under random sampling its size varies from step
to step, so it is not what an estimate is divided by.
_Avoid_: minibatch, sample

**Expected batch size**:
The constant a step's aggregate is divided by, fixed before the run rather
than read off the draw. Dividing by what the draw actually produced would make
the divisor depend on the data.
_Avoid_: realized batch size, actual batch size

**Padding cap**:
The fixed length a variable-size draw is padded out to. A memory bound, never
a privacy parameter.
_Avoid_: max batch size, batch limit, capacity

**Sampling rate**:
The probability an individual example is included in a given batch.
_Avoid_: batch ratio, selection probability

**Per-sample loss**:
The loss of a single example under a single set of parameters. The caller's
contribution to the pipeline, and the only place a model appears.
_Avoid_: loss function, objective, criterion

**Per-sample gradient**:
One gradient per example, before anything has been aggregated.
_Avoid_: individual gradient, microbatch gradient, example gradient

**Clipping norm**:
The ℓ₂ bound each per-sample gradient is rescaled to lie within, and so what
bounds the ℓ₂ sensitivity of the sum that follows.
_Avoid_: clip threshold, max norm, sensitivity

**Sensitivity**:
The most that adding or removing one example can change an aggregate,
measured in a named norm. The norm is part of the quantity, not a detail of
it: a Gaussian mechanism is calibrated against ℓ₂ sensitivity and a Laplace
mechanism against ℓ₁. The clipping norm *bounds* a sensitivity; the two are
not interchangeable words.
_Avoid_: influence, max contribution

**Noise scale**:
A noise distribution's own dispersion parameter, carrying the units of the
quantity being perturbed — the standard deviation of a Gaussian, the `b` of a
Laplace. Every distribution has one.
_Avoid_: sigma, noise level, magnitude, noise multiplier

**Noise multiplier**:
A noise scale divided by the sensitivity it is calibrated against, in the norm
that mechanism uses. Dimensionless, and the quantity an accountant takes.
Named per mechanism rather than shared: a Gaussian's and a Laplace's are not
the same ratio and are not interchangeable.
_Avoid_: noise scale, sigma, standard deviation

**Privatized gradient**:
The gradient estimate after perturbation, and the direction stage 7 descends
along. In the simplest methods it is also the whole of what the step released;
in a variance-reduced one it is not, being formed from this step's release and
earlier ones.
_Avoid_: noisy gradient, private gradient, sanitized gradient

### Privacy

**Mechanism**:
The randomized map whose privacy is analyzed: sampling, clipping, aggregation
and perturbation taken together. Two mechanisms that differ anywhere are
different mechanisms, however similar the code. When the literature names one
— the Gaussian mechanism, the exponential mechanism, the projection mechanism
— it means a complete map carrying an analysis of its own, not a component of
one.
_Avoid_: algorithm, method, procedure

**Release**:
Everything a mechanism makes public, and so the only thing an accountant
accounts for. The boundary: what a step releases is accounted, and everything
computed from it afterwards is post-processing. A release need not be a
gradient — a variance-reduced method releases an increment and forms its
estimate from that and earlier releases.
_Avoid_: output, result, noisy quantity

**Accountant**:
The conversion between a run's parameters and its privacy cost, in either
direction. What it computes depends on the mechanism, not on the algorithm
that ran it — where the mechanism is a standard one, the standard accountant
applies whatever surrounds it.
_Avoid_: privacy calculator, budget tracker, estimator

**Epsilon**:
The bound on how much more likely any outcome becomes when one example is
added or removed. Smaller is stronger, and the scale is multiplicative rather
than linear.
_Avoid_: privacy level, privacy loss, leakage

**Delta**:
The probability that the epsilon bound does not hold at all. Kept well below
the reciprocal of the dataset size, because at or above it a mechanism that
publishes a few records outright can still qualify.
_Avoid_: slack, tolerance, failure rate

**Privacy budget**:
The (ε, δ) pair a run is permitted to spend. A cost, not a setting: it is
spent by accessing the data, and composes over steps.
_Avoid_: privacy parameter, privacy level

**Post-processing**:
Anything computed from an already-released quantity. It costs no budget, and
saying something is post-processing is a claim about the mechanism, not about
the code.
_Avoid_: downstream, cleanup

**Adjacency**:
The relation between the two datasets a guarantee is stated over. dimma
assumes add-or-remove-one.
_Avoid_: neighbouring definition, dataset distance

**Amplification**:
The reduction in privacy cost that comes from each step seeing a random subset
rather than everything.
_Avoid_: subsampling bonus, dilution

### What dimma contains

**Algorithm**:
A choice at every stage, the loop that runs them, and its own accountant where
the standard ones do not cover it.
_Avoid_: model, optimizer, method, solver

**Baseline**:
The privacy-free counterpart of an algorithm dimma implements privately, built
from the same stages so the difference between them is the privacy and nothing
else.
_Avoid_: control, non-DP version, reference implementation

**Optimizer**:
The rule that turns a gradient estimate into a parameter update — stage 7, and
the one stage both sides of a comparison have to choose identically for the
comparison to mean anything.
_Avoid_: solver, descent rule, step rule

**Transform**:
A change to a quantity that is not itself an algorithm and composes across
several, such as a projection applied to an already-privatized gradient.
Whether a given transform is free is a claim about the mechanism it sits
inside, not a property the transform carries around with it.
_Avoid_: mechanism — in the literature that names a complete analyzed map, so
it would promise a guarantee a transform does not have
