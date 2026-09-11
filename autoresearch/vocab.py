from __future__ import annotations

from typing import Iterable

import torch

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY*")

AA_TO_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"],
}

CODON_TOKENS = sorted({codon for codons in AA_TO_CODONS.values() for codon in codons})
CODON_TO_AA = {codon: aa for aa, codons in AA_TO_CODONS.items() for codon in codons}

SRC_VOCAB = SPECIAL_TOKENS + AA_TOKENS
TGT_VOCAB = SPECIAL_TOKENS + CODON_TOKENS

SRC_TOKEN_TO_ID = {token: idx for idx, token in enumerate(SRC_VOCAB)}
SRC_ID_TO_TOKEN = {idx: token for token, idx in SRC_TOKEN_TO_ID.items()}
TGT_TOKEN_TO_ID = {token: idx for idx, token in enumerate(TGT_VOCAB)}
TGT_ID_TO_TOKEN = {idx: token for token, idx in TGT_TOKEN_TO_ID.items()}

SRC_PAD_ID = SRC_TOKEN_TO_ID["<pad>"]
SRC_BOS_ID = SRC_TOKEN_TO_ID["<bos>"]
SRC_EOS_ID = SRC_TOKEN_TO_ID["<eos>"]
SRC_UNK_ID = SRC_TOKEN_TO_ID["<unk>"]

TGT_PAD_ID = TGT_TOKEN_TO_ID["<pad>"]
TGT_BOS_ID = TGT_TOKEN_TO_ID["<bos>"]
TGT_EOS_ID = TGT_TOKEN_TO_ID["<eos>"]
TGT_UNK_ID = TGT_TOKEN_TO_ID["<unk>"]


def normalize_aas(seq: str) -> str:
    text = (seq or "").strip().upper()
    if not text:
        raise ValueError("Empty amino-acid sequence")
    invalid = sorted({token for token in text if token not in AA_TO_CODONS})
    if invalid:
        raise ValueError(f"Invalid amino-acid tokens: {''.join(invalid)}")
    return text


def normalize_cds(seq: str) -> str:
    text = (seq or "").strip().upper().replace("U", "T")
    if not text:
        raise ValueError("Empty CDS")
    if len(text) % 3 != 0:
        raise ValueError("CDS length must be divisible by 3")
    invalid = sorted({token for token in text if token not in {"A", "C", "G", "T"}})
    if invalid:
        raise ValueError(f"Invalid nucleotide tokens: {''.join(invalid)}")
    return text


def split_codons(cds: str) -> list[str]:
    seq = normalize_cds(cds)
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def translate_cds(cds: str) -> str:
    aas: list[str] = []
    for codon in split_codons(cds):
        aa = CODON_TO_AA.get(codon)
        if aa is None:
            raise ValueError(f"Unsupported codon: {codon}")
        aas.append(aa)
    return "".join(aas)


def encode_source(aas: str) -> list[int]:
    return [SRC_TOKEN_TO_ID[token] for token in normalize_aas(aas)]


def encode_target_from_codons(codons: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for codon in codons:
        token = codon.upper()
        if token not in TGT_TOKEN_TO_ID:
            raise ValueError(f"Unknown codon token: {token}")
        ids.append(TGT_TOKEN_TO_ID[token])
    return ids


def encode_target_from_cds(cds: str) -> list[int]:
    return encode_target_from_codons(split_codons(cds))


def decode_target_ids(token_ids: Iterable[int], skip_special: bool = True) -> list[str]:
    codons: list[str] = []
    for token_id in token_ids:
        token = TGT_ID_TO_TOKEN[int(token_id)]
        if skip_special and token in SPECIAL_TOKENS:
            continue
        codons.append(token)
    return codons


def decode_cds(token_ids: Iterable[int], skip_special: bool = True) -> str:
    return "".join(decode_target_ids(token_ids, skip_special=skip_special))


def build_constraint_bias() -> torch.Tensor:
    bias = torch.zeros((len(SRC_VOCAB), len(TGT_VOCAB)), dtype=torch.float32)
    valid_target_ids = {TGT_TOKEN_TO_ID[codon] for codon in CODON_TOKENS}
    all_invalid = torch.full((len(TGT_VOCAB),), -1e9, dtype=torch.float32)

    for aa in AA_TOKENS:
        row = all_invalid.clone()
        for codon in AA_TO_CODONS[aa]:
            row[TGT_TOKEN_TO_ID[codon]] = 0.0
        bias[SRC_TOKEN_TO_ID[aa]] = row

    for token in SPECIAL_TOKENS:
        token_id = SRC_TOKEN_TO_ID[token]
        bias[token_id, list(valid_target_ids)] = 0.0

    return bias
