from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import load_dataset_frame


PACKAGE_ROOT = Path(__file__).resolve().parent
DATASET_AGENT_SOP = PACKAGE_ROOT / "assets" / "dataset_agent_sop.md"

BRANCH_1_PAXDB_RM_RL = "branch_1_paxdb_rm_rl"
BRANCH_2_TRANSCRIPTOME_RM_RL = "branch_2_transcriptome_rm_rl"
BRANCH_3_CDS_ONLY_EA = "branch_3_cds_only_ea"
VALID_BRANCHES = {
    BRANCH_1_PAXDB_RM_RL,
    BRANCH_2_TRANSCRIPTOME_RM_RL,
    BRANCH_3_CDS_ONLY_EA,
}


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    if slug:
        return slug
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]
    return f"species_{digest}"


@dataclass
class DatasetAgentRequest:
    species: str
    stage_dir: str
    codex_model: str = ""
    timeout_seconds: int = 1800
    extra_instructions: str = ""


@dataclass
class DatasetAgentResult:
    request: dict[str, Any]
    dataset_csv: str
    branch_json: str
    requested_species: str
    canonical_species_name: str
    canonical_species_slug: str
    branch_name: str
    label_source: str
    has_abundance: bool
    dataset_schema: list[str]
    codex_returncode: int
    timed_out: bool


def build_dataset_agent_prompt(
    *,
    species: str,
    dataset_csv: Path,
    branch_json: Path,
    extra_instructions: str = "",
) -> str:
    lines = [
        f"Read and follow this SOP exactly: {DATASET_AGENT_SOP}",
        "",
        "Task context:",
        f"- requested species: {species}",
        "",
        "Output contract for this run:",
        f"- create exactly one standardized dataset CSV at: {dataset_csv}",
        f"- create exactly one branch-decision JSON at: {branch_json}",
        f"- the JSON field `dataset_csv` must equal: {dataset_csv}",
        "- the branch must be decided by you, not by fallback code",
        "- do not modify tracked source files",
        "- do not leave any extra files behind in the repository",
    ]
    if extra_instructions.strip():
        lines.extend(
            [
                "",
                "Additional caller instructions:",
                extra_instructions.strip(),
            ]
        )
    lines.extend(
        [
            "",
            "At the end, print the dataset CSV path and the branch JSON path.",
        ]
    )
    return "\n".join(lines)


def _validate_branch_payload(
    payload: dict[str, Any],
    *,
    dataset_csv: Path,
    cleaned_columns: list[str],
) -> tuple[str, bool]:
    branch_name = str(payload.get("branch_name", "")).strip()
    if branch_name not in VALID_BRANCHES:
        raise ValueError(f"Dataset agent returned invalid branch_name={branch_name!r}")
    has_abundance = "abundance" in cleaned_columns
    if branch_name in {BRANCH_1_PAXDB_RM_RL, BRANCH_2_TRANSCRIPTOME_RM_RL} and not has_abundance:
        raise ValueError(f"{branch_name} requires abundance labels, but dataset has columns {cleaned_columns}")
    if branch_name == BRANCH_3_CDS_ONLY_EA and has_abundance:
        raise ValueError("branch_3_cds_only_ea must not emit abundance labels")
    label_source = str(payload.get("label_source", "")).strip()
    valid_sources = {"paxdb", "transcriptome", "cds_only"}
    if label_source not in valid_sources:
        raise ValueError(f"Dataset agent returned invalid label_source={label_source!r}")
    json_dataset_csv = str(payload.get("dataset_csv", "")).strip()
    if json_dataset_csv and Path(json_dataset_csv).expanduser().resolve() != dataset_csv:
        raise ValueError(
            f"Dataset agent JSON points to {json_dataset_csv}, expected {dataset_csv}"
        )
    payload_schema = [str(item) for item in payload.get("dataset_schema", [])]
    if payload_schema and payload_schema != cleaned_columns:
        raise ValueError(f"Dataset schema mismatch: json={payload_schema} cleaned={cleaned_columns}")
    return branch_name, has_abundance


def run_dataset_agent(request: DatasetAgentRequest) -> DatasetAgentResult:
    if not DATASET_AGENT_SOP.exists():
        raise FileNotFoundError(f"Missing dataset-agent SOP: {DATASET_AGENT_SOP}")

    stage_dir = Path(request.stage_dir).expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = stage_dir / "standardized_dataset.csv"
    branch_json = stage_dir / "branch_decision.json"

    prompt = build_dataset_agent_prompt(
        species=request.species,
        dataset_csv=dataset_csv,
        branch_json=branch_json,
        extra_instructions=request.extra_instructions,
    )

    cmd = [
        "codex",
        "--search",
        "exec",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(PACKAGE_ROOT),
        "-",
    ]
    if request.codex_model:
        cmd[3:3] = ["-m", request.codex_model]

    proc = subprocess.Popen(cmd, cwd=PACKAGE_ROOT, text=True, stdin=subprocess.PIPE)
    try:
        proc.communicate(prompt, timeout=request.timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()

    if timed_out:
        raise TimeoutError(f"Dataset agent timed out after {request.timeout_seconds} seconds")
    if proc.returncode != 0:
        raise RuntimeError(f"Dataset agent failed with return code {proc.returncode}")
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset agent did not create dataset CSV: {dataset_csv}")
    if not branch_json.exists():
        raise FileNotFoundError(f"Dataset agent did not create branch JSON: {branch_json}")

    _, cleaned_df = load_dataset_frame(dataset=dataset_csv)
    cleaned_df.to_csv(dataset_csv, index=False)
    cleaned_columns = [str(column) for column in cleaned_df.columns]

    payload = json.loads(branch_json.read_text(encoding="utf-8"))
    branch_name, has_abundance = _validate_branch_payload(
        payload,
        dataset_csv=dataset_csv,
        cleaned_columns=cleaned_columns,
    )
    payload["dataset_csv"] = str(dataset_csv)
    payload["dataset_schema"] = cleaned_columns
    payload["has_abundance"] = bool(has_abundance)
    payload.setdefault("requested_species", request.species)
    payload.setdefault("canonical_species_name", request.species)
    payload.setdefault("canonical_species_slug", slugify(payload.get("canonical_species_name", request.species)))
    payload.setdefault("notes", "")
    branch_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = DatasetAgentResult(
        request=asdict(request),
        dataset_csv=str(dataset_csv),
        branch_json=str(branch_json),
        requested_species=str(payload["requested_species"]),
        canonical_species_name=str(payload["canonical_species_name"]),
        canonical_species_slug=str(payload["canonical_species_slug"]),
        branch_name=branch_name,
        label_source=str(payload["label_source"]),
        has_abundance=bool(has_abundance),
        dataset_schema=cleaned_columns,
        codex_returncode=int(proc.returncode),
        timed_out=bool(timed_out),
    )
    return result
