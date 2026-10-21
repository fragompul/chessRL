"""Registration point for the CLI subcommands added by later phases (train, eval, vision...)."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    return None
