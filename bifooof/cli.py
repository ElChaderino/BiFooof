"""Command line for BiFooof.

Examples::

    bifooof info recording.edf
    bifooof run recording.edf --out results
    bifooof compare --ec eyes_closed.edf --eo eyes_open.edf --out results
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from . import __version__
from .batch import compare_files, describe, format_description, run_file
from .io.load import load_recording
from .io.sniff import FILE_FILTER


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        if args.command == "formats":
            print(FILE_FILTER.replace(";;", "\n"))
            return 0
        if args.command == "info":
            return _info(args.files)
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            return _compare(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1
    parser.print_help()
    return 2


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Print the full traceback when a command fails.")
    parser = argparse.ArgumentParser(
        prog="bifooof",
        description="Fit specparam and bycycle on EEG recordings, from the command line or the window.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"BiFooof {__version__}")
    commands = parser.add_subparsers(dest="command")

    info = commands.add_parser("info", parents=[common], help="Load files and print channels, rate, and duration.")
    info.add_argument("files", nargs="+")

    run = commands.add_parser("run", parents=[common], help="Fit one or more recordings and write CSV and PNG files.")
    run.add_argument("files", nargs="+")
    run.add_argument("--condition", choices=("auto", "ec", "eo"), default="auto",
                     help="Eyes closed, eyes open, or guess from the filename. Default: auto.")
    run.add_argument("--out", type=Path, default=Path("bifooof-out"), help="Output folder.")
    run.add_argument("--channel", help="Channel for the spectrum highlight and the cycle fit.")
    run.add_argument("--band", default="alpha", help="Cycle band: delta, theta, alpha, beta, gamma, or 8-12.")
    run.add_argument("--no-cycles", action="store_true", help="Skip the bycycle fit.")
    run.add_argument("--no-plots", action="store_true", help="Write tables only.")

    compare = commands.add_parser("compare", parents=[common], help="Fit an eyes-closed and an eyes-open recording.")
    compare.add_argument("--ec", required=True, help="Eyes-closed file.")
    compare.add_argument("--eo", required=True, help="Eyes-open file.")
    compare.add_argument("--out", type=Path, default=Path("bifooof-out"), help="Output folder.")
    compare.add_argument("--channel", help="Channel for the overlay and the cycle comparison.")
    compare.add_argument("--band", default="alpha", help="Cycle band: delta, theta, alpha, beta, gamma, or 8-12.")
    compare.add_argument("--no-cycles", action="store_true")
    compare.add_argument("--no-plots", action="store_true")

    commands.add_parser("formats", help="List the file types BiFooof opens.")
    commands.add_parser("gui", help="Open the desktop window.")
    return parser


def _info(files: list[str]) -> int:
    failed = 0
    for path in files:
        try:
            print(format_description(describe(load_recording(path))))
            print()
        except Exception as exc:
            failed += 1
            print(f"{Path(path).name}\n  error: {exc}\n", file=sys.stderr)
    return 1 if failed else 0


def _run(args: argparse.Namespace) -> int:
    failed = 0
    for path in args.files:
        folder = args.out / _stem(path)
        print(f"Fitting {Path(path).name} -> {folder}")
        try:
            written = run_file(
                path, folder,
                condition=None if args.condition == "auto" else args.condition,
                channel=args.channel,
                band=args.band,
                cycles=not args.no_cycles,
                plots=not args.no_plots,
            )
        except Exception as exc:
            failed += 1
            print(f"  error: {exc}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()
            continue
        for item in written:
            print(f"  {item.name}")
    return 1 if failed else 0


def _compare(args: argparse.Namespace) -> int:
    folder = args.out
    print(f"Comparing {Path(args.ec).name} and {Path(args.eo).name} -> {folder}")
    written = compare_files(
        args.ec, args.eo, folder,
        channel=args.channel,
        band=args.band,
        cycles=not args.no_cycles,
        plots=not args.no_plots,
    )
    for item in written:
        print(f"  {item.name}")
    return 0


def _stem(path: str) -> str:
    stem = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return cleaned.strip("._") or "recording"
