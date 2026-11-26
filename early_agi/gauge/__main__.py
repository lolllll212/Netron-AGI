"""Allow ``python -m early_agi.gauge`` to run the Gauge CLI."""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
