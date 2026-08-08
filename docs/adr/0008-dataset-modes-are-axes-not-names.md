# Dataset loading modes are independent axes, not mode names

`load_criteo` takes two independent parameters — which columns, and whether
they are preprocessed — giving four combinations, rather than a single mode
name selecting among presets.

An earlier version of this loader had a mode called `"integer"`, which named
the columns it returned and said nothing about the median-fill, `log1p` and
standardization it also applied. That is the failure mode of preset names: the
half of the behaviour that does not fit in the name goes missing, and it is
reliably the half that matters for interpreting a result. Splitting the axes
makes preprocessing unmissable at the call site, and whatever it did is
recorded in the returned metadata.

## Consequences

Every statistic — medians, category frequencies, means, standard deviations —
is fitted on the training split alone. This is the standard benchmark
convention and is *not* a private operation: those statistics depend on the
training data and are accounted for in no privacy budget. Any (ε, δ) reported
for a run over this data inherits that caveat and should state it.

Loaders are a convenience and no algorithm imports them, so the dataframe stack
stays out of the base install.
