"""Stage CLI — the surface the agent protocol (SKILL.md) drives.

    .venv/bin/python -m pipeline.cli <cmd> ...   (from repo root)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backfill, build, ingest, template, trace, verify

STAGE_REPORTS = {"template": "template.json", "ingest": None, "trace": "trace.json",
                 "build": "build.json", "verify": "verify.json"}


def _save_report(project: Path, name: str, report: dict) -> None:
    rdir = Path(project) / "reports"
    rdir.mkdir(exist_ok=True)
    (rdir / name).write_text(json.dumps(report, indent=1, ensure_ascii=False))


def cmd_init(args) -> dict:
    project = Path(args.project)
    for d in ("sheets", "samples", "crops", "out"):
        (project / d).mkdir(parents=True, exist_ok=True)
    cfg = project / "config.toml"
    if not cfg.exists():
        cfg.write_text(f'family = "{args.family}"\npsname = "{args.family}"\n')
    return {"project": str(project), "family": args.family}


def cmd_template(args) -> dict:
    only = set(args.only) if args.only else None
    report = template.generate(Path(args.project) / "sheets", only=only)
    _save_report(args.project, "template.json", report)
    return report


def cmd_ingest_digital(args) -> dict:
    report = ingest.ingest_digital(Path(args.project), Path(args.sheet))
    _save_report(args.project, f"ingest_digital_{Path(args.sheet).stem}.json", report)
    return report


def cmd_ingest_paper(args) -> dict:
    report = ingest.ingest_paper(Path(args.project), Path(args.photo),
                                 sheet=args.sheet)
    _save_report(args.project, f"ingest_paper_{args.sheet}.json", report)
    return report


def cmd_ingest_extract(args) -> dict:
    crops = {}
    for pair in args.crops:
        gid, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"crops must be gid=path, got {pair!r}")
        crops[gid] = Path(path)
    report = ingest.ingest_extract(Path(args.project), crops,
                                   preset=args.preset,
                                   scale_to_body=not args.no_scale_to_body)
    _save_report(args.project, "ingest_extract.json", report)
    return report


def cmd_backfill(args) -> dict:
    report = backfill.backfill(Path(args.project), args.font)
    _save_report(args.project, "backfill.json", report)
    return report


def cmd_trace(args) -> dict:
    report = trace.trace(Path(args.project))
    _save_report(args.project, "trace.json", report)
    return report


def cmd_build(args) -> dict:
    report = build.build(Path(args.project))
    _save_report(args.project, "build.json", report)
    return report


def cmd_verify(args) -> dict:
    report = verify.verify(Path(args.project))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tamil-font-maker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("project")
    s.add_argument("--family", default=build.DEFAULT_FAMILY)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("template")
    s.add_argument("project")
    s.add_argument("--only", nargs="*", default=None,
                   help="completion mode: explicit glyph ids")
    s.set_defaults(fn=cmd_template)

    s = sub.add_parser("ingest-digital")
    s.add_argument("project")
    s.add_argument("sheet")
    s.set_defaults(fn=cmd_ingest_digital)

    s = sub.add_parser("ingest-paper")
    s.add_argument("project")
    s.add_argument("photo")
    s.add_argument("--sheet", required=True)
    s.set_defaults(fn=cmd_ingest_paper)

    s = sub.add_parser("ingest-extract")
    s.add_argument("project")
    s.add_argument("crops", nargs="+", metavar="gid=path")
    s.add_argument("--preset", default="faithful", choices=["faithful", "clean"])
    s.add_argument("--no-scale-to-body", action="store_true",
                   help="keep extracted letters at crop scale (default: "
                        "scale to match backfilled body letters; run "
                        "backfill first)")
    s.set_defaults(fn=cmd_ingest_extract)

    s = sub.add_parser("backfill")
    s.add_argument("project")
    s.add_argument("--font", required=True,
                   help="base font (e.g. Noto Sans Tamil) for missing cells")
    s.set_defaults(fn=cmd_backfill)

    s = sub.add_parser("trace")
    s.add_argument("project")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("build")
    s.add_argument("project")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("verify")
    s.add_argument("project")
    s.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    report = args.fn(args)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.cmd == "verify":
        return 0 if report["verdict"] != "fail" else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
