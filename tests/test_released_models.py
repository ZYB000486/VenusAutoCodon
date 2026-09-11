import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from autoresearch.vocab import translate_cds
from venusautocodon.generate import generate
from venusautocodon.predict import predict

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "models/manifest.json").read_text())
torch.set_num_threads(2)


@pytest.mark.parametrize("entry", MANIFEST["models"], ids=lambda e: e["species"])
def test_released_pair(entry):
    folder = ROOT / "models" / entry["species"]
    if not (folder / "generator.pt").exists():
        pytest.skip("Download the model archive to run this test")
    for name, key in [("generator.pt", "generator"), ("reward_model.json", "reward_model")]:
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == entry[key]["sha256"]
    proteins = [("short", "MAELGKFD"), ("all_residues", "MACDEFGHIKLMNPQRSTVWY*")]
    result = generate(proteins, folder, samples=2, seed=7)
    assert len(result) == 4
    assert all(translate_cds(row["predicted_cds"]) == row["aas"] for row in result)
    scores = predict([row["predicted_cds"] for row in result], folder)
    assert len(scores) == 4
    assert np.isfinite(scores).all()
    repeat = generate(proteins, folder, samples=2, seed=7)
    assert [r["predicted_cds"] for r in repeat] == [r["predicted_cds"] for r in result]
    greedy = generate(proteins[:1], folder, samples=1, greedy=True)
    assert translate_cds(greedy[0]["predicted_cds"]) == "MAELGKFD*"

