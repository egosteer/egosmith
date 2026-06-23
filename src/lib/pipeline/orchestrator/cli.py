"""CLI entrypoints for the dataset pipeline orchestrator."""

from __future__ import annotations

import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official dataset pipeline. Preferred stages: prepare, annotate, infer, filter, build, validate"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML pipeline config")
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stage list. Preferred: prepare,infer,filter,build,validate. "
            "When omitted, annotate is included only if annotation.command is configured."
        ),
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag override")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Resume compatible stages (prepare/infer/build), skipping work that "
            "already produced outputs. Defaults to the config 'resume' value, "
            "which defaults to disabled. Pass --no-resume to force off."
        ),
    )
    parser.add_argument(
        "--descriptor_manifest",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser
