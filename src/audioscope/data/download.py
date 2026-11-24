"""Download ESC-50 into ``data/esc50``.

    python -m audioscope.data.download --dataset esc50 --dest data/esc50

Tries a shallow ``git clone`` of the official repository first, then the
GitHub zip archive. The recordings are CC BY-NC (see the dataset's LICENSE);
they are kept under ``data/`` and never committed.

UrbanSound8K requires accepting its terms on the dataset page; unpack it to
``data/urbansound8k`` by hand (``metadata/`` and ``audio/fold1..10``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ESC50_GIT = "https://github.com/karolpiczak/ESC-50.git"
ESC50_ZIP = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def _is_complete(dest: Path) -> bool:
    return (dest / "meta" / "esc50.csv").exists() and len(list((dest / "audio").glob("*.wav"))) == 2000


def download_esc50(dest: str | Path = "data/esc50", method: str = "auto") -> Path:
    dest = Path(dest)
    if _is_complete(dest):
        print(f"ESC-50 already present at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    if method in ("auto", "git") and shutil.which("git"):
        try:
            if dest.exists():
                shutil.rmtree(dest)
            subprocess.run(["git", "clone", "--depth", "1", ESC50_GIT, str(dest)], check=True)
            shutil.rmtree(dest / ".git", ignore_errors=True)
            if _is_complete(dest):
                return dest
        except subprocess.CalledProcessError as exc:
            errors.append(f"git: {exc}")
    if method in ("auto", "zip"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zpath = Path(tmp) / "esc50.zip"
                print(f"downloading {ESC50_ZIP}")
                urllib.request.urlretrieve(ESC50_ZIP, zpath)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(tmp)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(Path(tmp) / "ESC-50-master"), dest)
            if _is_complete(dest):
                return dest
        except OSError as exc:
            errors.append(f"zip: {exc}")
    raise RuntimeError("ESC-50 download failed: " + "; ".join(errors))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esc50", choices=["esc50"])
    ap.add_argument("--dest", default="data/esc50")
    ap.add_argument("--method", default="auto", choices=["auto", "git", "zip"])
    args = ap.parse_args()
    path = download_esc50(args.dest, args.method)
    print(f"ESC-50 ready at {path}")


if __name__ == "__main__":
    main()
