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
from .analysis.settings import BURST_PRESETS, FitSettings
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
    _add_fit_args(run)

    compare = commands.add_parser("compare", parents=[common], help="Fit an eyes-closed and an eyes-open recording.")
    compare.add_argument("--ec", required=True, help="Eyes-closed file.")
    compare.add_argument("--eo", required=True, help="Eyes-open file.")
    compare.add_argument("--out", type=Path, default=Path("bifooof-out"), help="Output folder.")
    compare.add_argument("--channel", help="Channel for the overlay and the cycle comparison.")
    compare.add_argument("--band", default="alpha", help="Cycle band: delta, theta, alpha, beta, gamma, or 8-12.")
    compare.add_argument("--no-cycles", action="store_true")
    compare.add_argument("--no-plots", action="store_true")
    _add_fit_args(compare)

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
                settings=_settings(args),
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
        settings=_settings(args),
    )
    for item in written:
        print(f"  {item.name}")
    return 0


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tmin", type=float, default=0.0, help="Start of the fit window, in seconds.")
    parser.add_argument("--tmax", type=float, default=None, help="End of the fit window, in seconds. Default: end of the file.")
    parser.add_argument("--fmin", type=float, default=1.0, help="Lowest frequency in the spectrum fit.")
    parser.add_argument("--fmax", type=float, default=40.0, help="Highest frequency in the spectrum fit.")
    parser.add_argument("--segment", type=float, default=4.0, help="Welch segment length in seconds. 4 s is 0.25 Hz.")
    parser.add_argument("--aperiodic", choices=("fixed", "knee"), default="fixed",
                        help="Fixed 1/f slope, or a knee when the slope bends.")
    parser.add_argument("--peak-width", nargs=2, type=float, default=(0.5, 12.0), metavar=("LOW", "HIGH"),
                        help="Narrowest and widest peak, in Hz.")
    parser.add_argument("--max-peaks", type=int, default=6)
    parser.add_argument("--min-height", type=float, default=0.05, help="Minimum peak height in log10 power.")
    parser.add_argument("--peak-threshold", type=float, default=2.0,
                        help="Peak height in standard deviations. Papers leave this at 2.")
    parser.add_argument("--alpha", nargs=2, type=float, default=(8.0, 12.0), metavar=("LOW", "HIGH"),
                        help="Band used for alpha reactivity.")
    parser.add_argument("--bursts", choices=("library", "resting", "paper"), default="library",
                        help="Burst rule: library, resting alpha, or the paper alpha thresholds.")


def _settings(args: argparse.Namespace) -> FitSettings:
    length = None if args.tmax is None else float(args.tmax) - float(args.tmin)
    bursts = {"library": 0, "resting": 1, "paper": 2}[args.bursts]
    amp, amp_c, period_c, mono, min_cycles = BURST_PRESETS[bursts][1:]
    low, high = args.peak_width
    return FitSettings(
        t_start=float(args.tmin),
        t_length=length,
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        segment_s=float(args.segment),
        aperiodic_mode=args.aperiodic,
        peak_width_low=float(low),
        peak_width_high=float(high),
        max_n_peaks=int(args.max_peaks),
        min_peak_height=float(args.min_height),
        peak_threshold=float(args.peak_threshold),
        alpha_low=float(args.alpha[0]),
        alpha_high=float(args.alpha[1]),
        amp_fraction=float(amp),
        amp_consistency=float(amp_c),
        period_consistency=float(period_c),
        monotonicity=float(mono),
        min_n_cycles=int(min_cycles),
    )


def _stem(path: str) -> str:
    stem = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return cleaned.strip("._") or "recording"
