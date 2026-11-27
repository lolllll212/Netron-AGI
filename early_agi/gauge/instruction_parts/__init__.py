"""Single-responsibility opcode groups behind :mod:`early_agi.gauge.instructions`.

``instructions.py`` remains the public compatibility façade and owns the
``GaugeOpcodeRunner`` class identity; ``vm.py`` still imports only that façade.
"""
