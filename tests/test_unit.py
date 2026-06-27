"""
DocuChat — Unit Test Suite
==========================
Tests document extraction, text cleaning, chunking, vector store
construction, retrieval correctness, and API key validation.

Run:
    pytest tests/test_unit.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from docuchat.core.document import _clean_text, extract_text_from_file
from docuchat.core.rag import build_vector_store
from docuchat.core.validator import validate_groq_api_key

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# =============================================================================
# 1. Text Cleaning
# =============================================================================


class TestCleanText:
    def test_removes_non_breaking_spaces(self):
        assert _clean_text("hello\xa0world") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert _clean_text("hello   world") == "hello world"

    def test_normalises_curly_quotes(self):
        result = _clean_text("\u2018hello\u2019 \u201cworld\u201d")
        assert result == "'hello' \"world\""

    def test_normalises_dashes(self):
        assert _clean_text("A\u2013B\u2014C") == "A-B-C"

    def test_collapses_excess_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = _clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_preserves_newlines(self):
        result = _clean_text("line1\nline2")
        assert "\n" in result


# =============================================================================
# 2. Document Extraction (TXT)
# =============================================================================


class TestTextExtraction:
    def test_extracts_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello, world!\nSecond line.", encoding="utf-8")
        result = extract_text_from_file(str(f), "sample.txt")
        assert "Hello, world!" in result
        assert "Second line" in result

    def test_unsupported_extension_returns_message(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"data")
        result = extract_text_from_file(str(f), "file.xyz")
        assert "Unsupported" in result

    def test_handles_encoding_errors_gracefully(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_bytes(b"Good text \xff\xfe bad bytes")
        result = extract_text_from_file(str(f), "bad.txt")
        # Should not raise; returns a string
        assert isinstance(result, str)

    def test_extracts_company_policy_fixture(self):
        path = str(FIXTURES_DIR / "company_policy.txt")
        result = extract_text_from_file(path, "company_policy.txt")
        assert "Nova Tech Solutions" in result
        assert len(result) > 500

    def test_extracts_product_spec_fixture(self):
        path = str(FIXTURES_DIR / "product_spec.txt")
        result = extract_text_from_file(path, "product_spec.txt")
        assert "Helios X1" in result
        assert "97.8%" in result

    def test_extracts_research_paper_fixture(self):
        path = str(FIXTURES_DIR / "research_paper.txt")
        result = extract_text_from_file(path, "research_paper.txt")
        assert "Stanford" in result
        assert "342" in result


# =============================================================================
# 3. Vector Store Construction
# =============================================================================


class TestVectorStore:
    @pytest.fixture(scope="class")
    def sample_store(self):
        docs = [
            {
                "original_name": "test.txt",
                "text_content": (
                    "The capital of France is Paris. "
                    "The Eiffel Tower is located in Paris. "
                    "France is a country in Western Europe. " * 10
                ),
            }
        ]
        return build_vector_store(docs)

    def test_build_returns_non_none(self, sample_store):
        assert sample_store is not None

    def test_build_with_empty_content_returns_none(self):
        result = build_vector_store([{"original_name": "empty.txt", "text_content": ""}])
        assert result is None

    def test_build_with_empty_list_returns_none(self):
        result = build_vector_store([])
        assert result is None

    def test_similarity_search_returns_results(self, sample_store):
        results = sample_store.similarity_search("capital of France", k=2)
        assert len(results) >= 1

    def test_similarity_search_returns_relevant_chunk(self, sample_store):
        results = sample_store.similarity_search("capital of France", k=3)
        combined = " ".join(r.page_content for r in results).lower()
        assert "paris" in combined or "france" in combined

    def test_source_metadata_preserved(self, sample_store):
        results = sample_store.similarity_search("Eiffel Tower", k=1)
        assert results[0].metadata.get("source") == "test.txt"

    def test_multiple_documents(self):
        docs = [
            {"original_name": "doc_a.txt", "text_content": "The sky is blue. " * 20},
            {"original_name": "doc_b.txt", "text_content": "The grass is green. " * 20},
        ]
        store = build_vector_store(docs)
        assert store is not None
        results = store.similarity_search("color of the sky", k=3)
        sources = [r.metadata.get("source") for r in results]
        assert "doc_a.txt" in sources


# =============================================================================
# 4. Retrieval Accuracy — keyword-level ground truth
# =============================================================================


class TestRetrievalAccuracy:
    """Verify that the RAG system retrieves the correct chunk for specific
    factual questions from the fixture documents."""

    @pytest.fixture(scope="class")
    def policy_store(self):
        text = extract_text_from_file(
            str(FIXTURES_DIR / "company_policy.txt"), "company_policy.txt"
        )
        return build_vector_store(
            [{"original_name": "company_policy.txt", "text_content": text}]
        )

    @pytest.fixture(scope="class")
    def spec_store(self):
        text = extract_text_from_file(
            str(FIXTURES_DIR / "product_spec.txt"), "product_spec.txt"
        )
        return build_vector_store(
            [{"original_name": "product_spec.txt", "text_content": text}]
        )

    @pytest.fixture(scope="class")
    def research_store(self):
        text = extract_text_from_file(
            str(FIXTURES_DIR / "research_paper.txt"), "research_paper.txt"
        )
        return build_vector_store(
            [{"original_name": "research_paper.txt", "text_content": text}]
        )

    def _top_k_contains(self, store, question: str, keywords: list[str], k: int = 6) -> bool:
        results = store.similarity_search(question, k=k)
        combined = " ".join(r.page_content for r in results).lower()
        return any(kw.lower() in combined for kw in keywords)

    # Policy document
    def test_policy_remote_days(self, policy_store):
        assert self._top_k_contains(policy_store, "How many remote work days per week?", ["3 days per week"])

    def test_policy_pto_year_one(self, policy_store):
        assert self._top_k_contains(policy_store, "PTO accrual rate first year", ["15 days"])

    def test_policy_sick_days(self, policy_store):
        assert self._top_k_contains(policy_store, "How many sick days per year?", ["10 sick days"])

    def test_policy_parental_leave(self, policy_store):
        assert self._top_k_contains(policy_store, "Primary caregiver parental leave weeks", ["16 weeks"])

    def test_policy_401k_admin(self, policy_store):
        assert self._top_k_contains(policy_store, "Who administers the 401k plan?", ["Fidelity"])

    def test_policy_pip_duration(self, policy_store):
        assert self._top_k_contains(policy_store, "How long is a Performance Improvement Plan?", ["60 days"])

    # Product spec
    def test_spec_efficiency(self, spec_store):
        assert self._top_k_contains(spec_store, "Peak CEC efficiency of Helios X1", ["97.8%"])

    def test_spec_weight(self, spec_store):
        assert self._top_k_contains(spec_store, "What is the weight of the inverter?", ["18.4 kg"])

    def test_spec_warranty(self, spec_store):
        assert self._top_k_contains(spec_store, "Standard warranty period", ["10 years"])

    def test_spec_ip_rating(self, spec_store):
        assert self._top_k_contains(spec_store, "Ingress protection rating", ["IP65"])

    # Research paper
    def test_research_participants(self, research_store):
        assert self._top_k_contains(research_store, "How many participants in the study?", ["342"])

    def test_research_exec_function_drop(self, research_store):
        assert self._top_k_contains(research_store, "Reduction in executive function in sleep deprived workers", ["34%"])

    def test_research_decision_errors(self, research_store):
        assert self._top_k_contains(research_store, "Increase in decision-making errors", ["41%"])

    def test_research_perception_gap(self, research_store):
        assert self._top_k_contains(research_store, "How much did workers overestimate their performance?", ["18 percentage points"])

    def test_research_device_used(self, research_store):
        assert self._top_k_contains(research_store, "Device used to measure sleep duration", ["Fitbit"])


# =============================================================================
# 5. API Key Validation
# =============================================================================


class TestApiKeyValidation:
    def test_valid_format_accepted(self):
        valid, _ = validate_groq_api_key("gsk_" + "A" * 40)
        assert valid is True

    def test_empty_key_rejected(self):
        valid, msg = validate_groq_api_key("")
        assert valid is False
        assert "required" in msg.lower()

    def test_missing_gsk_prefix_rejected(self):
        valid, msg = validate_groq_api_key("sk_" + "A" * 40)
        assert valid is False
        assert "gsk_" in msg

    def test_too_short_rejected(self):
        valid, msg = validate_groq_api_key("gsk_short")
        assert valid is False
        assert "short" in msg.lower()

    def test_too_long_rejected(self):
        valid, msg = validate_groq_api_key("gsk_" + "A" * 200)
        assert valid is False

    def test_invalid_characters_rejected(self):
        valid, msg = validate_groq_api_key("gsk_abc$def!xyz" + "B" * 30)
        assert valid is False

    def test_demo_placeholder_rejected(self):
        for fake in ["test", "demo", "123", "no-key-needed"]:
            valid, _ = validate_groq_api_key(fake)
            assert valid is False, f"Expected '{fake}' to be rejected"

    def test_none_rejected(self):
        valid, msg = validate_groq_api_key(None)
        assert valid is False

    def test_strips_whitespace(self):
        valid, _ = validate_groq_api_key("  gsk_" + "A" * 40 + "  ")
        assert valid is True
