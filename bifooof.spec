# -*- mode: python ; coding: utf-8 -*-
"""One-file BiFooof executable for a clean Windows machine.

Build from F:\\BiFooof:

    pip install pyinstaller
    pyinstaller bifooof.spec

The result is dist\\BiFooof.exe. Copy that file by itself.
Double-click it for the window. From a command prompt the same exe runs
info, run, and compare. The first launch unpacks into a temporary folder, so
it takes longer to appear than a later launch in the same session.
WinEEG .eeg reading is part of the app.
"""

from pathlib import Path
import os

import mne
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

def _needed_mne_modules() -> list[str]:
    """Private and lazy MNE modules the readers import.

    PyInstaller skips underscore names, and MNE's lazy loader hides the
    implementations behind stubs. Walking every MNE subpackage also imports
    TensorFlow and Sphinx, so this stays on the file-reading path.
    """
    base = Path(mne.__file__).resolve().parent
    allow = {"_fiff", "html_templates", "utils", "channels", "io"}
    singles = {"_freesurfer", "_ola"}
    names: list[str] = []
    for path in base.rglob("*.py"):
        rel = path.relative_to(base).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or "tests" in parts:
            continue
        if not any(part.startswith("_") for part in parts):
            continue
        if parts[0] not in allow and not (len(parts) == 1 and parts[0] in singles):
            continue
        names.append("mne." + ".".join(parts))
    # Modules actually imported while opening EDF, BDF, GDF, BrainVision,
    # EEGLAB, FIF, and CNT. MNE loads most of these through stubs, so a
    # static analysis of launch.py never sees them.
    names += [
        "mne.annotations",
        "mne.baseline",
        "mne.bem",
        "mne.channels",
        "mne.channels.channels",
        "mne.channels.layout",
        "mne.channels.montage",
        "mne.cuda",
        "mne.defaults",
        "mne.epochs",
        "mne.event",
        "mne.evoked",
        "mne.filter",
        "mne.fixes",
        "mne.html_templates",
        "mne.io",
        "mne.io.base",
        "mne.io.brainvision",
        "mne.io.brainvision.brainvision",
        "mne.io.cnt",
        "mne.io.cnt.cnt",
        "mne.io.edf",
        "mne.io.edf.edf",
        "mne.io.eeglab",
        "mne.io.eeglab.eeglab",
        "mne.io.fiff",
        "mne.io.fiff.raw",
        "mne.parallel",
        "mne.rank",
        "mne.surface",
        "mne.time_frequency",
        "mne.time_frequency.multitaper",
        "mne.time_frequency.psd",
        "mne.time_frequency.spectrum",
        "mne.time_frequency.tfr",
        "mne.transforms",
        "mne.utils",
        "mne.utils.check",
        "mne.utils.config",
        "mne.utils.dataframe",
        "mne.utils.docs",
        "mne.utils.linalg",
        "mne.utils.misc",
        "mne.utils.mixin",
        "mne.utils.numerics",
        "mne.utils.progressbar",
        "mne.utils.spectrum",
        "mne.viz",
        "mne.viz.epochs",
        "mne.viz.evoked",
        "mne.viz.evoked_field",
        "mne.viz.misc",
        "mne.viz.montage",
        "mne.viz.raw",
        "mne.viz.topo",
        "mne.viz.topomap",
        "mne.viz.ui_events",
        "mne.viz.utils",
        "mne.viz._3d",
        "mne.viz._3d_overlay",
        "mne.viz._dipole",
    ]
    return list(dict.fromkeys(names))


root = Path(SPECPATH)
pathex = [str(root)]

hidden = [
    "specparam",
    "bycycle",
    "neurodsp",
    "pandas",
    "scipy.signal",
    "mne",
    "mne.io",
    "mne.io.edf",
    "mne.io.brainvision",
    "mne.io.cnt",
    "mne.io.eeglab",
    "mne.io.fiff",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_qtagg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
]
hidden += collect_submodules("bifooof")
hidden += _needed_mne_modules()
hidden += collect_submodules("specparam")
hidden += collect_submodules("bycycle")
hidden += collect_submodules("neurodsp")

icon = root / "bifooof" / "assets" / "bifooof.ico"
# MNE loads its public API from these stubs. They have to sit beside the package.
mne_dir = Path(mne.__file__).resolve().parent
mne_stubs = [
    (str(path), str(Path("mne") / path.relative_to(mne_dir).parent))
    for path in mne_dir.rglob("*.pyi")
]

a = Analysis(
    [str(root / "bifooof" / "launch.py")],
    pathex=pathex,
    binaries=[],
    datas=([(str(icon), "bifooof/assets")] if icon.is_file() else []) + collect_data_files("mne") + mne_stubs,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg", "QtAgg"]}},
    runtime_hooks=[],
    excludes=[
        "tkinter", "PyQt5", "PyQt6",
        "torch", "tensorflow", "tensorboard",
        "IPython", "jupyter", "notebook", "pytest", "black", "jedi",
        "nilearn", "skimage", "cv2", "imageio", "sqlalchemy", "sympy",
        "av", "sklearn", "tables", "h5py", "pyarrow", "dask", "distributed",
    ],
    noarchive=False,
)
# Qt 6 on Windows imports the OS ICU interface (unversioned function names).
# A PDF/tooling directory on PATH can supply a different icuuc.dll whose exports
# have version suffixes. Do not shadow Windows' ICU with that unrelated copy.
if os.name == "nt":
    windows_icu = {"icu.dll", "icuuc.dll", "icuin.dll"}
    a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in windows_icu]
pyz = PYZ(a.pure)

# One file. Windows treats BiFooof.exe and bifooof.exe as the same file, so a
# second name cannot carry a different window mode. No arguments opens the
# window; info, run, and compare stay on the command line. The console is
# hidden when the exe is double-clicked and kept when it is started from a prompt.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BiFooof",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    hide_console="hide-early",
    disable_windowed_traceback=False,
    icon=str(icon) if icon.is_file() else None,
)
