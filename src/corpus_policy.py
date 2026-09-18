"""Corpus loading and explicit data-quality policies."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import CONFLICT_PAIRS, NEAR_DUP_CLUSTERS


@dataclass
class Document:
    id: str
    title: str
    text: str

    @property
    def search_text(self) -> str:
        return f"{self.title} {self.text}".strip()


def _fix_encoding(s: str) -> str:
    """Normalize common mojibake / replacement for em dashes; do not invent content."""
    if not s:
        return s
    # UTF-8 / Windows-1252 mojibake for em dash and similar
    replacements = {
        "\ufffd": "-",  # replacement character
        "â€”": "—",
        "â€“": "–",
        "Ã¢â‚¬â€": "—",
        "Â": "",
    }
    out = s
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    # Collapse runs of replacement leftovers around dashes in titles
    out = re.sub(r"\s*[—–\-]\s*", " — ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return unicodedata.normalize("NFKC", out)


def load_docs(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            docs.append(
                Document(
                    id=raw["id"],
                    title=_fix_encoding(raw.get("title", "")),
                    text=raw.get("text", "").strip(),
                )
            )
    return docs


def near_dup_cluster(doc_id: str) -> frozenset[str] | None:
    for cluster in NEAR_DUP_CLUSTERS:
        if doc_id in cluster:
            return cluster
    return None


def conflict_partner_ids(doc_ids: Iterable[str]) -> set[str]:
    """Return all ids that participate in a known conflict with any of doc_ids."""
    present = set(doc_ids)
    partners: set[str] = set()
    for pair in CONFLICT_PAIRS:
        if present & pair:
            partners |= set(pair)
    return partners


def expand_near_dups(doc_ids: Iterable[str]) -> list[str]:
    """When citing near-duplicates, include the whole cluster if any member is present."""
    seen: list[str] = []
    added: set[str] = set()
    for did in doc_ids:
        cluster = near_dup_cluster(did)
        members = sorted(cluster) if cluster else [did]
        for m in members:
            if m not in added:
                seen.append(m)
                added.add(m)
    return seen


def detect_pressure_conflict(hits: list[Document]) -> str | None:
    """
    If retrieved docs include conflicting P-200 max-pressure claims, return a disclosure.
    Policy: disclose conflict > pick arbitrarily.
    """
    ids = {d.id for d in hits}
    if not {"DOC-01", "DOC-02"} <= ids:
        return None
    values: dict[str, str] = {}
    for d in hits:
        if d.id not in {"DOC-01", "DOC-02"}:
            continue
        m = re.search(
            r"maximum operating pressure is (\d+)\s*bar",
            d.text,
            flags=re.I,
        )
        if m:
            values[d.id] = m.group(1)
    if len(values) >= 2 and len(set(values.values())) >= 2:
        parts = [f"{did} states {val} bar" for did, val in sorted(values.items())]
        return (
            "Conflicting information in the documents: "
            + "; ".join(parts)
            + ". Both sources are cited; a single authoritative value is not chosen."
        )
    return None
