# Non-private baselines live in the library, built from the same stages

SGD, Adam and non-private SpiderBoost are part of dimma rather than scripts
kept beside it. A baseline is the same pipeline with stages 4 and 6 dropped,
stage 3 changed to a per-batch gradient, and stage 1 relaxed to ordinary
sampling — expressed in the same primitives, taking the same per-sample loss
its private counterpart would be given.

The alternative — comparing against whatever reference implementation is handy
— is how most DP papers do it, and it makes "what did privacy cost here" an
uncontrolled question: the private and non-private runs differ in the
optimizer, the data loading, and the loss as well as in the privacy. Keeping
the baseline inside the library means the difference between the two runs is
the privacy and nothing else.

## Consequences

Classical DP-SGD is subject to the same rule for the opposite reason. It is the
reference every non-classical method is measured against, so it gets the same
primitives, tests and documentation as anything else rather than being a
strawman written to lose.
