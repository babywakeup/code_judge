"""Launch with python -m periodic_template_matcher.app --result OUTPUT_DIR."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import argparse


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Stage-2 input/predicted-mask desktop viewer')
    parser.add_argument('--result', type=Path, help='Saved result directory or result.json; omit to choose in app')
    args = parser.parse_args(argv)
    # Import Tk only here: matching/saving works without desktop dependencies.
    try:
        import tkinter as tk
        from .viewer import PredictionViewer
    except ImportError as error:
        parser.exit(1, f'The desktop viewer needs Python with Tkinter support: {error}\n')
    app = None
    try:
        app = PredictionViewer()
        if args.result is not None:
            app.load(args.result)
        app.mainloop()
    except (tk.TclError, OSError, ValueError, KeyError) as error:
        if app is not None:
            app.destroy()
        parser.exit(1, f'Could not open viewer: {error}\n')


if __name__ == '__main__':
    main()
