"""Registry of all compliance rules."""

from __future__ import annotations

from . import dependency, descope

RULES = (descope, dependency)
