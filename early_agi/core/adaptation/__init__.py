"""Single adaptation boundary behind the historical core façades.

``core/plasticity.py`` and ``core/adaptability.py`` remain public compatibility
modules. Implementation and shared state live here so surprise, structural
novelty, policy evolution, and offline adaptation curves have one source of
truth. The package intentionally exposes no eager imports: the benchmark curves
reach into Gauge lazily and must not create an import cycle.
"""
