"""
DocuChat RAG Pipeline — Accuracy Evaluation Script
====================================================
Measures retrieval quality WITHOUT requiring a Groq API key.

Metrics computed
----------------
  Hit Rate @k   : % of questions where the correct chunk is among the top-k
                  retrieved results  (k = 1, 3, 6)
  MRR           : Mean Reciprocal Rank — rewards correct answers ranked higher
  Precision @6  : Fraction of top-6 retrieved chunks that are truly relevant
  Avg Latency   : Mean retrieval time per query in milliseconds

Run
---
    python tests/evaluate_rag.py           # pretty-print report
    python tests/evaluate_rag.py --json    # also write results/eval_report.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from docuchat.core.document import extract_text_from_file
from docuchat.core.rag import build_vector_store

# ---------------------------------------------------------------------------
# QA Dataset — 30 questions across 3 documents, each tagged with at least one
# keyword/phrase that MUST appear in the correctly retrieved chunk.
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"

QA_DATASET: list[dict] = [
    # ── Company Policy (company_policy.txt) ──────────────────────────────────
    {
        "id": "CP-01",
        "doc": "company_policy.txt",
        "question": "How many remote work days are employees allowed per week?",
        "gold_keywords": ["3 days per week", "remote"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-02",
        "doc": "company_policy.txt",
        "question": "What is the PTO accrual rate for an employee in their first year?",
        "gold_keywords": ["15 days per year", "1.25"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-03",
        "doc": "company_policy.txt",
        "question": "How many sick days do employees receive each year?",
        "gold_keywords": ["10 sick days"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-04",
        "doc": "company_policy.txt",
        "question": "How many weeks of parental leave does a primary caregiver get?",
        "gold_keywords": ["16 weeks"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-05",
        "doc": "company_policy.txt",
        "question": "What percentage of health insurance premiums does the company cover for the employee?",
        "gold_keywords": ["90%"],
        "category": "Policy / Percentages",
    },
    {
        "id": "CP-06",
        "doc": "company_policy.txt",
        "question": "What is the annual professional development budget per employee?",
        "gold_keywords": ["$2,000", "2,000"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-07",
        "doc": "company_policy.txt",
        "question": "Who administers the company's 401k retirement plan?",
        "gold_keywords": ["Fidelity"],
        "category": "Policy / Facts",
    },
    {
        "id": "CP-08",
        "doc": "company_policy.txt",
        "question": "How long does a Performance Improvement Plan run?",
        "gold_keywords": ["60 days"],
        "category": "Policy / Numbers",
    },
    {
        "id": "CP-09",
        "doc": "company_policy.txt",
        "question": "What is the screen lock timeout requirement on company devices?",
        "gold_keywords": ["5 minutes"],
        "category": "Policy / Security",
    },
    {
        "id": "CP-10",
        "doc": "company_policy.txt",
        "question": "How many bereavement days are given for an immediate family member?",
        "gold_keywords": ["5 paid bereavement"],
        "category": "Policy / Numbers",
    },
    # ── Product Spec (product_spec.txt) ──────────────────────────────────────
    {
        "id": "PS-01",
        "doc": "product_spec.txt",
        "question": "What is the peak CEC efficiency of the Helios X1 inverter?",
        "gold_keywords": ["97.8%"],
        "category": "Spec / Efficiency",
    },
    {
        "id": "PS-02",
        "doc": "product_spec.txt",
        "question": "What is the maximum DC input power of the inverter?",
        "gold_keywords": ["6,000 W", "6000"],
        "category": "Spec / Electrical",
    },
    {
        "id": "PS-03",
        "doc": "product_spec.txt",
        "question": "What is the rated AC output power?",
        "gold_keywords": ["5,000 W", "5 kW"],
        "category": "Spec / Electrical",
    },
    {
        "id": "PS-04",
        "doc": "product_spec.txt",
        "question": "How many MPPT inputs does the Helios X1 support?",
        "gold_keywords": ["2", "MPPT"],
        "category": "Spec / Electrical",
    },
    {
        "id": "PS-05",
        "doc": "product_spec.txt",
        "question": "What is the ingress protection rating of the inverter?",
        "gold_keywords": ["IP65"],
        "category": "Spec / Environmental",
    },
    {
        "id": "PS-06",
        "doc": "product_spec.txt",
        "question": "What is the weight of the Helios X1 unit?",
        "gold_keywords": ["18.4 kg"],
        "category": "Spec / Physical",
    },
    {
        "id": "PS-07",
        "doc": "product_spec.txt",
        "question": "What is the standard warranty period for the inverter?",
        "gold_keywords": ["10 years"],
        "category": "Spec / Warranty",
    },
    {
        "id": "PS-08",
        "doc": "product_spec.txt",
        "question": "What communication protocols does the inverter support?",
        "gold_keywords": ["Modbus", "SunSpec"],
        "category": "Spec / Communication",
    },
    {
        "id": "PS-09",
        "doc": "product_spec.txt",
        "question": "What is the operating temperature range of the inverter?",
        "gold_keywords": ["-25", "+60", "60°C"],
        "category": "Spec / Environmental",
    },
    {
        "id": "PS-10",
        "doc": "product_spec.txt",
        "question": "What safety standard certifications does the inverter have?",
        "gold_keywords": ["UL 1741"],
        "category": "Spec / Certifications",
    },
    # ── Research Paper (research_paper.txt) ──────────────────────────────────
    {
        "id": "RP-01",
        "doc": "research_paper.txt",
        "question": "What percentage reduction in executive function was found in severely sleep-deprived workers?",
        "gold_keywords": ["34%"],
        "category": "Research / Statistics",
    },
    {
        "id": "RP-02",
        "doc": "research_paper.txt",
        "question": "How many participants were in the study?",
        "gold_keywords": ["342"],
        "category": "Research / Methodology",
    },
    {
        "id": "RP-03",
        "doc": "research_paper.txt",
        "question": "What device was used to objectively measure sleep duration?",
        "gold_keywords": ["Fitbit"],
        "category": "Research / Methodology",
    },
    {
        "id": "RP-04",
        "doc": "research_paper.txt",
        "question": "What was the increase in decision-making errors in the severe CPSD group?",
        "gold_keywords": ["41%"],
        "category": "Research / Statistics",
    },
    {
        "id": "RP-05",
        "doc": "research_paper.txt",
        "question": "By how much did sleep-deprived workers overestimate their performance?",
        "gold_keywords": ["18 percentage points"],
        "category": "Research / Statistics",
    },
    {
        "id": "RP-06",
        "doc": "research_paper.txt",
        "question": "How long was the study conducted?",
        "gold_keywords": ["12-week", "12 weeks"],
        "category": "Research / Methodology",
    },
    {
        "id": "RP-07",
        "doc": "research_paper.txt",
        "question": "What is the recommended sleep duration for adults according to the National Sleep Foundation?",
        "gold_keywords": ["7 to 9 hours", "7–9 hours"],
        "category": "Research / Facts",
    },
    {
        "id": "RP-08",
        "doc": "research_paper.txt",
        "question": "Did caffeine fully compensate for decision-making deficits in severely sleep-deprived workers?",
        "gold_keywords": ["no significant", "cannot compensate", "no"],
        "category": "Research / Conclusions",
    },
    {
        "id": "RP-09",
        "doc": "research_paper.txt",
        "question": "Which institution conducted the study?",
        "gold_keywords": ["Stanford"],
        "category": "Research / Facts",
    },
    {
        "id": "RP-10",
        "doc": "research_paper.txt",
        "question": "What cognitive test battery was used in the assessments?",
        "gold_keywords": ["CANTAB", "Cambridge Neuropsychological"],
        "category": "Research / Methodology",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    id: str
    question: str
    category: str
    doc: str
    gold_keywords: list[str]
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_6: bool = False
    reciprocal_rank: float = 0.0
    precision_at_6: float = 0.0
    latency_ms: float = 0.0
    top_sources: list[str] = field(default_factory=list)


def _chunk_contains_any(chunk_text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears (case-insensitive) in the chunk."""
    lower = chunk_text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _build_stores(all_docs: dict[str, str]) -> dict[str, object]:
    """Build per-document FAISS stores (dict keyed by filename)."""
    stores = {}
    for filename, text in all_docs.items():
        file_dict = [{"original_name": filename, "text_content": text}]
        store = build_vector_store(file_dict)
        if store:
            stores[filename] = store
    return stores


def _build_combined_store(all_docs: dict[str, str]) -> object:
    """Build a single FAISS store from all documents combined."""
    file_list = [
        {"original_name": name, "text_content": text}
        for name, text in all_docs.items()
    ]
    return build_vector_store(file_list)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------
def evaluate(k_values: tuple[int, ...] = (1, 3, 6)) -> list[QueryResult]:
    """Run retrieval evaluation on all QA pairs. Returns a list of QueryResult."""
    print("\n📂  Loading test fixtures …", end="", flush=True)
    all_docs: dict[str, str] = {}
    for filename in ["company_policy.txt", "product_spec.txt", "research_paper.txt"]:
        path = str(FIXTURES_DIR / filename)
        text = extract_text_from_file(path, filename)
        all_docs[filename] = text
    print(" done ✓")

    print("🔨  Building FAISS vector store …", end="", flush=True)
    combined_store = _build_combined_store(all_docs)
    print(" done ✓\n")

    results: list[QueryResult] = []
    K_MAX = max(k_values)

    for qa in QA_DATASET:
        t0 = time.perf_counter()
        docs_with_scores = combined_store.similarity_search_with_relevance_scores(
            qa["question"], k=K_MAX
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Check each rank position for relevance
        hit_rank: Optional[int] = None
        relevant_count = 0
        top_sources = []
        for rank, (doc, score) in enumerate(docs_with_scores, start=1):
            is_relevant = _chunk_contains_any(doc.page_content, qa["gold_keywords"])
            if is_relevant:
                relevant_count += 1
                if hit_rank is None:
                    hit_rank = rank
            top_sources.append(doc.metadata.get("source", "?"))

        rr = (1.0 / hit_rank) if hit_rank else 0.0
        precision_at_6 = relevant_count / K_MAX

        result = QueryResult(
            id=qa["id"],
            question=qa["question"],
            category=qa["category"],
            doc=qa["doc"],
            gold_keywords=qa["gold_keywords"],
            hit_at_1=(hit_rank == 1) if hit_rank else False,
            hit_at_3=(hit_rank is not None and hit_rank <= 3),
            hit_at_6=(hit_rank is not None and hit_rank <= 6),
            reciprocal_rank=rr,
            precision_at_6=precision_at_6,
            latency_ms=elapsed_ms,
            top_sources=top_sources,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
GREY  = "\033[90m"


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def print_report(results: list[QueryResult]) -> dict:
    total = len(results)
    hit_1 = sum(r.hit_at_1 for r in results)
    hit_3 = sum(r.hit_at_3 for r in results)
    hit_6 = sum(r.hit_at_6 for r in results)
    mrr   = sum(r.reciprocal_rank for r in results) / total
    p6    = sum(r.precision_at_6 for r in results) / total
    avg_lat = sum(r.latency_ms for r in results) / total

    # ── Per-category breakdown ────────────────────────────────────────────
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        cat = r.category.split("/")[0].strip()
        categories.setdefault(cat, []).append(r)

    # ── Per-document breakdown ────────────────────────────────────────────
    docs: dict[str, list[QueryResult]] = {}
    for r in results:
        docs.setdefault(r.doc, []).append(r)

    sep = "─" * 72

    print(f"\n{BOLD}{CYAN}{'='*72}{RESET}")
    print(f"{BOLD}{CYAN}  DocuChat RAG Evaluation Report{RESET}")
    print(f"{CYAN}{'='*72}{RESET}\n")

    # Overall metrics
    print(f"{BOLD}  OVERALL METRICS  ({total} questions across 3 documents){RESET}")
    print(f"  {sep}")
    print(f"  {'Metric':<30}  {'Score':>10}  {'Interpretation':>20}")
    print(f"  {sep}")
    rows = [
        ("Hit Rate @1",    hit_1/total, "Top-1 chunk is correct"),
        ("Hit Rate @3",    hit_3/total, "Correct chunk in top 3"),
        ("Hit Rate @6",    hit_6/total, "Correct chunk in top 6"),
        ("MRR",            mrr,         "Avg reciprocal rank"),
        ("Precision @6",   p6,          "% relevant in top 6"),
    ]
    for name, val, interp in rows:
        color = GREEN if val >= 0.80 else (YELLOW if val >= 0.60 else RED)
        print(f"  {name:<30}  {color}{_pct(val):>10}{RESET}  {GREY}{interp:>20}{RESET}")
    print(f"  {sep}")
    print(f"  {'Avg Retrieval Latency':<30}  {avg_lat:>9.1f}ms")
    print()

    # Per-document breakdown
    print(f"\n{BOLD}  PER-DOCUMENT BREAKDOWN{RESET}")
    print(f"  {sep}")
    print(f"  {'Document':<32}  {'@1':>6}  {'@3':>6}  {'@6':>6}  {'MRR':>6}")
    print(f"  {sep}")
    for docname, rs in docs.items():
        n = len(rs)
        d_h1  = _pct(sum(r.hit_at_1 for r in rs) / n)
        d_h3  = _pct(sum(r.hit_at_3 for r in rs) / n)
        d_h6  = _pct(sum(r.hit_at_6 for r in rs) / n)
        d_mrr = f"{sum(r.reciprocal_rank for r in rs)/n:.3f}"
        label = docname.replace(".txt", "")
        print(f"  {label:<32}  {d_h1:>6}  {d_h3:>6}  {d_h6:>6}  {d_mrr:>6}")
    print()

    # Per-question detail
    print(f"\n{BOLD}  PER-QUESTION RESULTS{RESET}")
    print(f"  {sep}")
    header = f"  {'ID':<7}  {'Category':<24}  {'@1':>4}  {'@3':>4}  {'@6':>4}  {'RR':>5}  {'ms':>6}"
    print(header)
    print(f"  {sep}")
    for r in results:
        h1 = f"{GREEN}✓{RESET}" if r.hit_at_1 else f"{RED}✗{RESET}"
        h3 = f"{GREEN}✓{RESET}" if r.hit_at_3 else f"{RED}✗{RESET}"
        h6 = f"{GREEN}✓{RESET}" if r.hit_at_6 else f"{RED}✗{RESET}"
        rr = f"{r.reciprocal_rank:.3f}"
        ms = f"{r.latency_ms:.1f}"
        cat = r.category[:24]
        print(f"  {r.id:<7}  {cat:<24}  {h1:>4}  {h3:>4}  {h6:>4}  {rr:>5}  {ms:>6}")
    print()

    # Failures
    failures = [r for r in results if not r.hit_at_6]
    if failures:
        print(f"\n{BOLD}{RED}  MISSED QUESTIONS (not in top-6):{RESET}")
        for r in failures:
            print(f"  {RED}✗{RESET} [{r.id}] {r.question[:70]}")
    else:
        print(f"\n  {GREEN}{BOLD}✅  All questions retrieved correctly within top-6!{RESET}")

    print(f"\n{CYAN}{'='*72}{RESET}\n")

    return {
        "total_questions": total,
        "hit_rate_at_1": round(hit_1 / total, 4),
        "hit_rate_at_3": round(hit_3 / total, 4),
        "hit_rate_at_6": round(hit_6 / total, 4),
        "mrr": round(mrr, 4),
        "precision_at_6": round(p6, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "per_document": {
            docname: {
                "hit_rate_at_1": round(sum(r.hit_at_1 for r in rs) / len(rs), 4),
                "hit_rate_at_3": round(sum(r.hit_at_3 for r in rs) / len(rs), 4),
                "hit_rate_at_6": round(sum(r.hit_at_6 for r in rs) / len(rs), 4),
                "mrr": round(sum(r.reciprocal_rank for r in rs) / len(rs), 4),
            }
            for docname, rs in docs.items()
        },
        "per_question": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    write_json = "--json" in sys.argv
    results = evaluate()
    metrics = print_report(results)

    if write_json:
        out_dir = Path(__file__).parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "eval_report.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  📄  JSON report saved to {out_path}\n")
