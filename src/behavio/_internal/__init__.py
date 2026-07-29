"""Private implementation helpers shared across Behavio modules.

Nothing in this package is public API. Modules here must stay dependency-free with
respect to the rest of ``behavio`` so that any module, including ``behavio.contracts``,
can import them without creating a cycle.
"""

from __future__ import annotations
