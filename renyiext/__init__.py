"""02-ext-research: distributional shape and algorithmic structure in bot detection.

Design documents are the source of truth and live in ``02-ext-research/docs/``.
Read ``00-CHARTER.md`` (hypotheses) and ``06-STANDING-RULES.md`` (the datasaurus
gate) before changing anything here.

Layering is strict, following the convention of the three replications:

    events -> spectrum / dna -> features -> models -> evaluate -> pipeline -> experiments/plots

No scientific logic lives in a notebook; notebooks import from here so the same
code is testable outside Jupyter.
"""

__all__ = ["config", "events", "spectrum", "features", "evaluate",
           "behaviour", "textfront", "tailstats", "dna", "ait",
           "generators", "lid", "checks"]
