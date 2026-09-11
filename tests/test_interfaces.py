import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from autoresearch import run_pipeline
from autoresearch.dataset_agent import (
    BRANCH_1_PAXDB_RM_RL, BRANCH_2_TRANSCRIPTOME_RM_RL, BRANCH_3_CDS_ONLY_EA,
    _validate_branch_payload,
)
from autoresearch.uniref_prompts import download_prompts
from venusautocodon.download_models import unpack_model
from venusautocodon.io import model_directory, read_fasta


def test_fasta(tmp_path):
    path = tmp_path / "proteins.fasta"
    path.write_text(">protein 1\nMAEL\nGKFD\n>protein 2\nMACD\n")
    assert read_fasta(path) == [("protein 1", "MAELGKFD"), ("protein 2", "MACD")]


@pytest.mark.parametrize("content", ["MAEL", ">\nMAEL", ">x\n", ">x\nMAEL\n>x\nMAEL"])
def test_invalid_fasta(tmp_path, content):
    path = tmp_path / "bad.fasta"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_fasta(path)


def test_no_model_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        model_directory(tmp_path, "../other")


def test_archive_safety(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        item = tarfile.TarInfo("../outside")
        item.size = 1
        handle.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        unpack_model(archive, tmp_path / "models", "test_species")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("branch,expected", [(BRANCH_1_PAXDB_RM_RL, "rl"),
    (BRANCH_2_TRANSCRIPTOME_RM_RL, "rl"), (BRANCH_3_CDS_ONLY_EA, "evolution")])
def test_branch_routing(branch, expected):
    assert run_pipeline.determine_effective_mode(SimpleNamespace(mode="auto"),
        branch_name=branch, has_abundance=expected == "rl") == expected


def test_branch_label_mismatch(tmp_path):
    with pytest.raises(ValueError):
        _validate_branch_payload({"branch_name": BRANCH_1_PAXDB_RM_RL},
            dataset_csv=tmp_path / "data.csv", cleaned_columns=["gene_name", "cds"])


def test_prompt_download(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {"X-UniProt-Release": "test"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(
        b">a\nMAEL\n>b\nMACD\n>duplicate\nMAEL\n>c\nMWYY\n>invalid\nMX\n"))
    train, validation = tmp_path / "train.fasta", tmp_path / "validation.fasta"
    download_prompts(train, validation, count=3, validation_count=1)
    assert len(read_fasta(train)) == 2
    assert len(read_fasta(validation)) == 1
    assert not ({s for _, s in read_fasta(train)} & {s for _, s in read_fasta(validation)})
    assert json.loads(train.with_suffix(".manifest.json").read_text())["release"] == "test"
    with pytest.raises(FileExistsError):
        download_prompts(train, validation, count=3, validation_count=1)


def test_existing_prompt_split(tmp_path):
    train, validation = tmp_path / "train.fasta", tmp_path / "validation.fasta"
    train.write_text(">a\nMAEL\n")
    validation.write_text(">b\nMACD\n")
    info = run_pipeline.prepare_fixed_uniref_prompt_split(species_root=tmp_path / "run", max_aa_len=512,
        train_prompt_path=str(train), validation_prompt_path=str(validation))
    assert info["train_prompt_count"] == info["validation_prompt_count"] == 1
    with pytest.raises(ValueError):
        run_pipeline.prepare_fixed_uniref_prompt_split(species_root=tmp_path / "run", max_aa_len=512,
            train_prompt_path=str(train), validation_prompt_path=str(train))
