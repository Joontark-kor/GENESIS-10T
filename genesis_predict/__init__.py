"""Research track: prediction engines around the polarbear pipeline (opt-in, off by default).

The polarbear rule is "the model writes the formula, the calculator calculates". This package
extends it to quantities a calculator cannot produce:

* ``event``   (pre-processing): forecast a future value of a time series, with quantiles and samples,
  so the formula can be evaluated over the whole distribution.
* ``physics`` (post-processing): the model fills a scenario *spec* for a fixed template solver; the
  solver recomputes the answer and checks the model's claimed value and units.

Nothing in ``genesis`` (the core) imports this package, and nothing here is wired into
``genesis.harness.evaluate``. A result only moves to the core after it passes the yield check
described in ``docs/13_predict_ko.md``.
"""
