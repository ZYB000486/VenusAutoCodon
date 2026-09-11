# VenusAutoCodon

AutoResearch-guided, host-specific codon optimization. This repository provides:

1. An end-to-end workflow from a species name to a CDS design method: dataset collection, branch selection, supervised policy initialization, reward-model AutoResearch, GRPO AutoResearch, and inference export. Species without expression labels use the evolutionary-search branch.
2. Selected reward models and GRPO-trained CDS generators for 16 expression-labelled yeast species, with standalone inference interfaces.

## Installation

Python 3.10 or newer is required. Run commands from the cloned repository root.

```bash
git clone https://github.com/ZYB000486/VenusAutoCodon.git
cd VenusAutoCodon
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Inference supports CPU or CUDA. Install a CUDA-compatible PyTorch build for GPU use. The complete AutoResearch workflow additionally needs an authenticated `codex` CLI, Git, and `timeout` (or `gtimeout`). The unlabelled-species evolutionary branch needs `RNAfold` from ViennaRNA. On macOS, `gtimeout` is provided by GNU coreutils. GPU training is recommended; the default search budgets are 8 hours for RM and 4 hours for RL per labelled species, plus dataset collection and policy initialization.

## Download models

Code is stored in Git; model pairs are attached to [release v0.1.0](https://github.com/ZYB000486/VenusAutoCodon/releases/tag/v0.1.0). Each species archive contains:

- `reward_model.json`: learned reward-model weights, feature normalization and reference statistics.
- `generator.pt`: the GRPO-trained generator weights and architecture configuration.
- `metadata.json`: species, label type, selection objectives, model configuration and checksums.

```bash
python -m venusautocodon.download_models --species saccharomyces_cerevisiae
# Or download all 16 pairs:
python -m venusautocodon.download_models --all
```

The downloader verifies SHA-256 checksums. The full [model manifest](models/manifest.json) records each archive's size and checksum. Generator checkpoints contain inference weights only; optimizer states and training datasets are not included. Reward models use their native JSON format.

## Predict a CDS reward score

```bash
python -m venusautocodon.predict \
  --species saccharomyces_cerevisiae \
  --cds ATGGCTGAACTGGGTAAATTCGATTAA \
  --output predictions_reward.csv

# Batch input: --input coding_sequences.fasta
```

The output is a species-specific learned reward score for comparing CDS candidates within that species. It is not a calibrated protein yield, and scores from different species are on different scales. Neither this interface nor generator inference requires training datasets or an agent service.

## Generate optimized CDSs

```bash
python -m venusautocodon.generate \
  --species saccharomyces_cerevisiae \
  --protein MAELGKFD \
  --samples 5 \
  --output predictions_cds.csv

# Batch input and GPU:
python -m venusautocodon.generate \
  --species saccharomyces_cerevisiae \
  --input proteins.fasta --samples 5 --device cuda \
  --output predictions_cds.csv
```

Input proteins use the 20 standard amino-acid letters, optionally followed by a terminal `*`. Output CDSs include a stop codon. The interface checks that each CDS translates back to its input protein under the model's codon table. Released models support at most 1,023 amino acids plus a terminal stop. `--greedy --samples 1` selects deterministic decoding; sampling uses temperature 1.0 and seed 42 by default. Use `--models-dir` to select a different model directory.

The code's codon constraint uses the standard genetic code for all hosts. Hosts with alternative nuclear codon assignments require host-specific interpretation of generated sequences.

## Run the complete AutoResearch workflow

Activate the environment and ensure `codex`, `git` and `timeout` are available on `PATH`. The agent stages execute and revise Python programs with full process permissions; run them in a dedicated research environment. They use your configured Codex account and consume its usage.

```bash
python -m autoresearch.run_pipeline \
  --species saccharomyces_cerevisiae \
  --mode auto --device cuda \
  --run-name scer_autoresearch
```

The dataset agent searches for protein-abundance labels, then transcript-abundance labels, and otherwise collects sequence-only CDSs. Labelled branches initialize a species-specific AA-to-CDS policy, search reward-model training programs by validation Spearman correlation, and search GRPO programs by validation reward. The sequence-only branch uses head-region evolutionary optimization.

For a fresh labelled run, missing RL protein prompts are downloaded from UniRef50 and split into fixed training/validation FASTAs. These are newly retrieved inputs for that run. To use an existing split instead:

```bash
python -m autoresearch.run_pipeline \
  --species saccharomyces_cerevisiae --mode auto --device cuda \
  --rl-train-prompts /absolute/path/train.fasta \
  --rl-validation-prompts /absolute/path/validation.fasta
```

Results are written beneath `autoresearch/runs/_pipelines/<run-name>/<species>/`, including selected RM/RL artifacts and `inference/run_aas2cds_inference.sh`. Run that script with `--aas MAELGKFD` or `--aas-file proteins.fasta`. Keep the associated run directory for inference from a newly trained model. Stage budgets can be set with `--rm-stage-seconds` and `--rl-stage-seconds`. AutoResearch uses Git worktrees created from the selected commit; commit local source edits before launching a new search. A new adaptive search may select a different program from the released models.

Implementation entry points and stage details are documented in [autoresearch/README.md](autoresearch/README.md).

## Supported released species

| Protein-abundance labels | Transcript-abundance labels |
| --- | --- |
| `candida_albicans` | `candida_tropicalis` |
| `kluyveromyces_lactis` | `cyberlindnera_jadinii` |
| `saccharomyces_cerevisiae` | `debaryomyces_hansenii` |
| `schizosaccharomyces_pombe` | `kluyveromyces_marxianus` |
| | `komagataella_pastoris` |
| | `lachancea_thermotolerans` |
| | `lipomyces_starkeyi` |
| | `ogataea_parapolymorpha` |
| | `torulaspora_delbrueckii` |
| | `wickerhamomyces_anomalus` |
| | `yarrowia_lipolytica` |
| | `zygosaccharomyces_rouxii` |

“Selected” means the best validation artifact retained in the corresponding species search. Matching reward models and RL generators are distributed together.

## Validation and license

```bash
python -m pip install -e '.[test]'
python -m pytest -q
# After downloading every model:
python -m pytest -q tests/test_released_models.py
```

Tests with released weights check all 16 loaders, finite reward scores and amino-acid preservation during generation. Full multi-hour autonomous searches are not part of the test suite.

The source code and released model weights are distributed under the [MIT License](LICENSE). Third-party tools and data retrieved at runtime retain their own licenses. The AutoResearch loop follows the iterative program-editing approach popularized by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch); GRPO follows [DeepSeekMath](https://arxiv.org/abs/2402.03300).
