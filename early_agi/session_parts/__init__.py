"""Single-responsibility pieces behind :mod:`early_agi.session`.

``session.py`` remains the public compatibility façade and owns the composed
``AgentSession`` class identity; these modules hold the turn loop, learning,
archiving, and workspace methods without changing their source semantics.
"""
