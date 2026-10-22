"""Launch the BiFooof window, or a headless command."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if _wants_gui(args):
        if args and args[0] in {"gui", "--gui"}:
            args = args[1:]
        sys.argv = [sys.argv[0], *args]
        from bifooof.ui.app import main as gui_main
        gui_main()
        return 0
    os.environ["BIFOOF_HEADLESS"] = "1"
    import matplotlib
    matplotlib.use("Agg", force=True)
    from bifooof.cli import main as cli_main
    return cli_main(args)


def _wants_gui(args: list[str]) -> bool:
    """No arguments opens the window. Subcommands stay on the command line."""
    if args and args[0] in {"gui", "--gui"}:
        return True
    return not args


if __name__ == "__main__":
    raise SystemExit(main())
