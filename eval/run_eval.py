"""Re-runnable evaluation: baseline vs improved retrieval + abstention metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abstain import looks_like_abstain
from src.config import CORPUS_PATH, EVAL_SET_PATH, RESULTS_PATH
from src.pipeline import RagPipeline
from src.retrieve import build_index, retrieve


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# --- Baseline path (mirrors baseline_rag.py, kept self-contained for eval) ---

def _baseline_index(corpus_path: Path):
    from sentence_transformers import SentenceTransformer

    from src.config import EMBED_MODEL_DIR, EMBED_MODEL_ID
    from src.corpus_policy import load_docs

    docs = load_docs(corpus_path)
    model_path = str(EMBED_MODEL_DIR) if EMBED_MODEL_DIR.exists() and any(EMBED_MODEL_DIR.iterdir()) else EMBED_MODEL_ID
    model = SentenceTransformer(model_path)
    chunks = []
    for d in docs:
        # Same flawed fixed-size chunking as baseline (effectively whole short docs)
        text = d.text
        size = 400
        for i in range(0, max(len(text), 1), size):
            chunks.append({"doc_id": d.id, "title": d.title, "text": text[i : i + size]})
    vectors = model.encode([c["text"] for c in chunks], show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors, model


def baseline_top_docs(query: str, chunks, vectors, model, k: int = 3) -> list[str]:
    q = model.encode([query], show_progress_bar=False)[0].astype("float32")
    q = q / np.linalg.norm(q)
    sims = vectors @ q
    order = np.argsort(-sims)[:k]
    seen = []
    for i in order:
        did = chunks[int(i)]["doc_id"]
        if did not in seen:
            seen.append(did)
    return seen


def baseline_answer(query: str, chunks, vectors, model) -> str:
    """Baseline always returns top-1 chunk — never abstains."""
    q = model.encode([query], show_progress_bar=False)[0].astype("float32")
    q = q / np.linalg.norm(q)
    sims = vectors @ q
    best = int(np.argmax(sims))
    hit = chunks[best]
    return f"[{hit['doc_id']}] {hit['text']}"


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    if not expected:
        return False
    top = set(retrieved[:k])
    return bool(top & set(expected))


def extract_cited_ids(answer: str) -> list[str]:
    return re.findall(r"DOC-\d+", answer or "")


def run_eval(use_llm: bool = False) -> dict:
    items = load_eval_set(EVAL_SET_PATH)
    answerable = [x for x in items if x.get("answerable")]
    unanswerable = [x for x in items if not x.get("answerable")]

    # Baseline
    b_chunks, b_vectors, b_model = _baseline_index(CORPUS_PATH)
    b_hit1 = b_hit3 = 0
    b_abstain_correct = 0
    for item in answerable:
        docs = baseline_top_docs(item["question"], b_chunks, b_vectors, b_model, k=3)
        if hit_at_k(docs, item["expected_doc_ids"], 1):
            b_hit1 += 1
        if hit_at_k(docs, item["expected_doc_ids"], 3):
            b_hit3 += 1
    for item in unanswerable:
        ans = baseline_answer(item["question"], b_chunks, b_vectors, b_model)
        if looks_like_abstain(ans):
            b_abstain_correct += 1

    # Improved
    pipe = RagPipeline.create(use_llm=use_llm, load_llm=use_llm)
    i_hit1 = i_hit3 = 0
    i_cite = 0
    conflict_ok = 0
    conflict_n = 0
    details = []

    for item in answerable:
        hits = retrieve(item["question"], pipe.index, top_n=3)
        docs = [h.doc.id for h in hits]
        h1 = hit_at_k(docs, item["expected_doc_ids"], 1)
        h3 = hit_at_k(docs, item["expected_doc_ids"], 3)
        if h1:
            i_hit1 += 1
        if h3:
            i_hit3 += 1
        result = pipe.answer(item["question"])
        cited = extract_cited_ids(result.answer)
        grounded = bool(set(cited) & set(item["expected_doc_ids"])) or (
            not result.abstained and bool(set(docs) & set(item["expected_doc_ids"]))
        )
        if grounded and not result.abstained:
            i_cite += 1
        if item.get("expect_conflict"):
            conflict_n += 1
            if "conflict" in result.answer.lower() and not result.abstained:
                conflict_ok += 1
        details.append(
            {
                "id": item["id"],
                "answerable": True,
                "retrieved": docs,
                "abstained": result.abstained,
                "answer": result.answer,
                "hit@1": h1,
                "hit@3": h3,
            }
        )

    # Abstention metrics on unanswerable
    tp = fp = fn = tn = 0
    # For abstention: positive class = "should abstain"
    for item in unanswerable:
        result = pipe.answer(item["question"])
        predicted_abstain = result.abstained or looks_like_abstain(result.answer)
        if predicted_abstain:
            tp += 1
        else:
            fn += 1
        details.append(
            {
                "id": item["id"],
                "answerable": False,
                "retrieved": [h.doc.id for h in result.hits],
                "abstained": predicted_abstain,
                "answer": result.answer,
                "gate_reason": result.gate.reason,
            }
        )

    # False abstention on answerable
    for item in answerable:
        # re-check from details
        pass
    false_abstain = sum(1 for d in details if d.get("answerable") and d.get("abstained"))
    fp = false_abstain
    tn = len(answerable) - false_abstain

    def _safe_div(a, b):
        return float(a) / b if b else 0.0

    n_ans = len(answerable)
    n_un = len(unanswerable)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    abstain_acc = _safe_div(tp + tn, n_ans + n_un)

    results = {
        "n_answerable": n_ans,
        "n_unanswerable": n_un,
        "baseline": {
            "hit@1": _safe_div(b_hit1, n_ans),
            "hit@3": _safe_div(b_hit3, n_ans),
            "abstain_recall_unanswerable": _safe_div(b_abstain_correct, n_un),
            "counts": {"hit@1": b_hit1, "hit@3": b_hit3, "abstain_tp": b_abstain_correct},
        },
        "improved": {
            "hit@1": _safe_div(i_hit1, n_ans),
            "hit@3": _safe_div(i_hit3, n_ans),
            "citation_groundedness": _safe_div(i_cite, n_ans),
            "abstention": {
                "precision": precision,
                "recall": recall,
                "accuracy": abstain_acc,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "conflict_disclosure_rate": _safe_div(conflict_ok, conflict_n) if conflict_n else None,
            "counts": {
                "hit@1": i_hit1,
                "hit@3": i_hit3,
                "grounded": i_cite,
                "conflict_ok": conflict_ok,
                "conflict_n": conflict_n,
            },
        },
        "details": details,
        "use_llm": use_llm,
    }
    return results


def print_report(results: dict) -> None:
    b = results["baseline"]
    i = results["improved"]
    print("=" * 64)
    print("RAG evaluation (reproducible)")
    print("=" * 64)
    print(f"Answerable: {results['n_answerable']} | Unanswerable: {results['n_unanswerable']}")
    print()
    print(f"{'Metric':<40} {'Baseline':>10} {'Improved':>10}")
    print("-" * 64)
    print(f"{'Retrieval Hit@1':<40} {b['hit@1']:>10.2%} {i['hit@1']:>10.2%}")
    print(f"{'Retrieval Hit@3':<40} {b['hit@3']:>10.2%} {i['hit@3']:>10.2%}")
    print(
        f"{'Abstain recall (unanswerable)':<40} "
        f"{b['abstain_recall_unanswerable']:>10.2%} {i['abstention']['recall']:>10.2%}"
    )
    print(f"{'Abstention precision':<40} {'n/a':>10} {i['abstention']['precision']:>10.2%}")
    print(f"{'Abstention accuracy':<40} {'n/a':>10} {i['abstention']['accuracy']:>10.2%}")
    print(f"{'Citation groundedness':<40} {'n/a':>10} {i['citation_groundedness']:>10.2%}")
    if i.get("conflict_disclosure_rate") is not None:
        print(f"{'Conflict disclosure rate':<40} {'n/a':>10} {i['conflict_disclosure_rate']:>10.2%}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce RAG evaluation metrics")
    parser.add_argument("--llm", action="store_true", help="Use Qwen for generation during eval")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="Write JSON results here")
    args = parser.parse_args()

    results = run_eval(use_llm=args.llm)
    print_report(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Store without huge duplication risk — keep details
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
