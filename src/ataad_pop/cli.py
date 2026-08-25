"""Command-line utilities for validation, privacy checks, and synthetic examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AnalysisConfig
from .privacy import scan_public_tree
from .schema import derive_bentall, load_feature_metadata, read_table, validate_input
from .synthetic import write_synthetic


def _parser() -> argparse.ArgumentParser:
    config = AnalysisConfig.from_environment()
    parser = argparse.ArgumentParser(prog="ataad-pop")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-config", help="Print resolved paths and frozen constants")
    show.set_defaults(func=cmd_show_config)

    synthetic = sub.add_parser("make-synthetic", help="Create schema-compatible synthetic data")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--feature-metadata", type=Path, default=config.feature_metadata)
    synthetic.add_argument("--n", type=int, default=360)
    synthetic.add_argument("--seed", type=int, default=config.seed)
    synthetic.set_defaults(func=cmd_make_synthetic)

    validate = sub.add_parser("validate-data", help="Validate an input workbook or CSV")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--sheet", default=config.sheet_name)
    validate.add_argument("--feature-metadata", type=Path, default=config.feature_metadata)
    validate.set_defaults(func=cmd_validate_data)

    privacy = sub.add_parser("privacy-check", help="Scan a proposed public repository")
    privacy.add_argument("--root", type=Path, default=Path.cwd())
    privacy.set_defaults(func=cmd_privacy_check)
    return parser


def cmd_show_config(_args: argparse.Namespace) -> int:
    config = AnalysisConfig.from_environment()
    print(json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=2))
    return 0


def cmd_make_synthetic(args: argparse.Namespace) -> int:
    path = write_synthetic(args.output, args.feature_metadata, args.n, args.seed)
    print(f"Synthetic dataset written to {path}")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    metadata = load_feature_metadata(args.feature_metadata)
    frame = read_table(args.data, sheet_name=args.sheet)
    issues = validate_input(frame, metadata)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    outcome = derive_bentall(frame)
    print(
        f"OK: rows={len(frame)}, candidate_features={len(metadata)}, "
        f"surgeons={frame['Surgeon'].nunique()}, Bentall_events={int(outcome.sum())}"
    )
    return 0


def cmd_privacy_check(args: argparse.Namespace) -> int:
    findings = scan_public_tree(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"BLOCK: {finding.path}: {finding.reason}")
        return 1
    print("OK: no prohibited clinical-data artifacts detected")
    return 0


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

