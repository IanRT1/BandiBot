"""Validate installed resources in a fresh environment outside the checkout."""

import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile


def main():
    wheels = list(Path("dist").glob("*.whl"))
    assert len(wheels) == 1, "Build into a clean dist directory before checking the wheel"
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required = {
        "assets/BandiBot.onnx", "assets/wake_activation.wav", "core/paths.py",
        "data/instructions.example.txt", "data/server_info.example.txt",
    }
    assert required <= names, f"Missing wheel files: {sorted(required - names)}"
    assert not any(name.startswith(("tests/", "build/", ".env")) for name in names)
    assert not {"data/instructions.txt", "data/server_info.txt"} & names

    with tempfile.TemporaryDirectory(prefix="bandibot-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            check=True, cwd=root,
        )
        subprocess.run([str(python), "-I", "-c", """
from importlib.metadata import distribution
from pathlib import Path
import sys
from core.paths import assets_root, context_path, packaged_data_root
assert (assets_root() / 'BandiBot.onnx').is_file()
assert (assets_root() / 'wake_activation.wav').is_file()
assert (packaged_data_root() / 'server_info.example.txt').is_file()
assert context_path('instructions.txt').is_file()
assert Path(sys.prefix).resolve() in assets_root().resolve().parents
entry = next(e for e in distribution('bandibot').entry_points if e.name == 'bandibot')
assert entry.value == 'core.client:main'
"""], check=True, cwd=root)
    print(f"Validated {wheel.name}")


if __name__ == "__main__":
    main()
