"""Download ESC-50 into ``data/esc50``.

    python -m audioscope.data.download --dataset esc50 --dest data/esc50

Tries a shallow ``git clone`` of the official repository first, then the
GitHub zip archive. The recordings are CC BY-NC (see the dataset's LICENSE);
they are kept under ``data/`` and never committed.

Pretrained weights for Stage 3 go to ``data/pretrained``:

    python -m audioscope.data.download --weights resnet18 cnn14

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


WEIGHTS = {
    # torchvision ResNet18_Weights.IMAGENET1K_V1 (BSD-3-Clause, torchvision)
    "resnet18": ("https://download.pytorch.org/models/resnet18-f37072fd.pth", "resnet18-f37072fd.pth"),
    # PANNs Cnn14, AudioSet mAP 0.431 (Zenodo record 3987831, CC BY 4.0)
    "cnn14": ("https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1", "Cnn14_mAP=0.431.pth"),
}


def download_weights(name: str, dest: str | Path = "data/pretrained") -> Path:
    url, fname = WEIGHTS[name]
    path = Path(dest) / fname
    if path.exists():
        print(f"{name}: already at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: downloading {url}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esc50", choices=["esc50"])
    ap.add_argument("--dest", default="data/esc50")
    ap.add_argument("--method", default="auto", choices=["auto", "git", "zip"])
    ap.add_argument("--weights", nargs="*", choices=sorted(WEIGHTS), help="download pretrained weights instead")
    ap.add_argument("--weights-dest", default="data/pretrained")
    args = ap.parse_args()
    if args.weights:
        for name in args.weights:
            print(f"{name} ready at {download_weights(name, args.weights_dest)}")
        return
    path = download_esc50(args.dest, args.method)
    print(f"ESC-50 ready at {path}")


if __name__ == "__main__":
    main()
