"""Deterministic (no-LLM) anti-hallucination checks. A rewritten bullet may NOT introduce
new numbers or new technology/tool names that are not already in the original bullet or its project's tech list."""
import re
from typing import List

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#]*(?:\.[A-Za-z0-9]+)*")

# Generic words/acronyms a rewrite may use freely
GENERIC = {"ai", "ml", "api", "apis", "ui", "ux", "llm", "llms", "json", "rest", "ci", "cd",
           "ml-based", "end", "to"}


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(text)


def allowed_vocab(original: str, tech: List[str]) -> set:
    vocab = {t.lower() for t in tokens(original)}
    for item in tech:
        vocab.update(t.lower() for t in tokens(item))
    return vocab | GENERIC


def _looks_technical(tok: str, is_first: bool) -> bool:
    if any(c.isdigit() for c in tok):
        return True
    if any(c.isupper() for c in tok[1:]):        # LangChain, FastAPI, RAG, PyTorch
        return True
    return tok[0].isupper() and not is_first     # a capitalised word mid-sentence (Docker, Kubernetes)


def check_bullet(original: str, rewritten: str, tech: List[str]) -> List[str]:
    """Return a list of problems (empty list = passed)."""
    issues = []
    if not rewritten.strip():
        return ["rewritten bullet is empty"]
    if len(rewritten) > 1.4 * len(original) + 20:
        issues.append("rewritten bullet is much longer than the original")

    new_numbers = set(_NUM.findall(rewritten)) - set(_NUM.findall(original))
    if new_numbers:
        issues.append(f"introduces numbers not in the original: {sorted(new_numbers)}")

    vocab = allowed_vocab(original, tech)
    toks = tokens(rewritten)
    new_terms = []
    for i, tok in enumerate(toks):
        if tok.lower() in vocab:
            continue
        if _looks_technical(tok, is_first=(i == 0)) and tok not in new_terms:
            new_terms.append(tok)
    if new_terms:
        issues.append(f"introduces terms not in the original/project tech: {new_terms}")
    return issues