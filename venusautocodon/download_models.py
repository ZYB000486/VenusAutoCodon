"""Download one species or all 16 model pairs from GitHub Releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/ZYB000486/VenusAutoCodon/releases/download/v0.1.0"


def unpack_model(archive: Path, output: Path, species: str) -> None:
    expected = {f"{species}/{name}" for name in ("generator.pt", "reward_model.json", "metadata.json")}
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if len(members) != 3 or {m.name for m in members} != expected or any(not m.isfile() for m in members):
            raise ValueError("Unexpected model archive contents")
        folder = output / species
        folder.mkdir(parents=True, exist_ok=True)
        for member in members:
            with handle.extractfile(member) as source, (folder / Path(member.name).name).open("wb") as dest:
                shutil.copyfileobj(source, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--species", nargs="+")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--output", default="models")
    parser.add_argument("--manifest", default="", help="Optional local manifest JSON")
    args = parser.parse_args()
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        with urllib.request.urlopen(f"{RELEASE}/manifest.json", timeout=60) as response:
            manifest = json.load(response)
    entries = {m["species"]: m for m in manifest["models"]}
    if args.list:
        print("\n".join(sorted(entries)))
        return
    selected = sorted(entries) if args.all else args.species
    unknown = set(selected) - set(entries)
    if unknown:
        parser.error(f"Unknown species: {', '.join(sorted(unknown))}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for species in selected:
        entry = entries[species]
        with tempfile.TemporaryDirectory(prefix="venusautocodon-") as tmp:
            archive = Path(tmp) / "model.tar.gz"
            print(f"Downloading {species}…", flush=True)
            digest = hashlib.sha256()
            with urllib.request.urlopen(f"{RELEASE}/{entry['asset']}", timeout=120) as response, archive.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for {species}")
            unpack_model(archive, output, species)
        print(f"Ready: {output / species}", flush=True)


if __name__ == "__main__":
    main()
