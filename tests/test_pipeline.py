"""Run with:  pytest -q
These tests use a fake LLM and (mostly) a fake compiler, so they need no API key and no LaTeX."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from google.genai import errors

from agent import checks, report
from agent.fake_llm import FakeLLM
from agent.graph import generate
from agent.llm import GeminiLLM, LLMError
from agent.schemas import BulletVerdict, JDAnalysis, RewriteResult, RewrittenBullet, VerifyResult

ROOT = Path(__file__).resolve().parent.parent
MASTER = yaml.safe_load((ROOT / "data" / "master.yaml").read_text(encoding="utf-8"))
JD = ("GenAI Engineer. Required: Python, FastAPI, LangChain, RAG. "
      "Nice to have: Kubernetes, Docker. You will build LLM agents.")


class Scripted(FakeLLM):
    is_mock = False

    def __init__(self, rewrite=None, verify=None):
        self.rewrite, self.verify, self.prompts, self.rewrite_calls = rewrite, verify, [], 0

    def structured(self, *, system, prompt, schema, temperature=0.2):
        self.prompts.append(system + prompt)
        if schema is RewriteResult and self.rewrite:
            self.rewrite_calls += 1
            items = json.loads(prompt.split("ITEMS_JSON:\n", 1)[1])
            return RewriteResult(bullets=[RewrittenBullet(id=i["id"], text=self.rewrite(i)) for i in items])
        if schema is VerifyResult and self.verify:
            items = json.loads(prompt.split("ITEMS_JSON:\n", 1)[1])
            return VerifyResult(verdicts=[self.verify(i) for i in items])
        return super().structured(system=system, prompt=prompt, schema=schema, temperature=temperature)


def fake_compile(pages_seq):
    calls = {"n": 0}

    def compile_fn(tex_path):
        pdf = tex_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 fake")
        n = pages_seq[min(calls["n"], len(pages_seq) - 1)]
        calls["n"] += 1
        tex_path.with_suffix(".log").write_text(f"Output written on {pdf.name} ({n} page, 100 bytes).")
        return pdf
    return compile_fn


def run(llm, tmp_path, pages=(1,)):
    return generate(llm, MASTER, JD, "Acme", "GenAI", out_root=tmp_path, compile_fn=fake_compile(pages))


# ---------- deterministic checks ----------
def test_checks_accept_safe_rewrite():
    orig = "Built an AI agent using LangChain tool calling for telecom diagnosis."
    assert checks.check_bullet(orig, "Engineered an AI agent using LangChain tool calling for telecom diagnosis.",
                               ["Python", "LangChain"]) == []


def test_checks_flag_new_tech_and_numbers():
    orig = "Built an AI agent using LangChain."
    assert checks.check_bullet(orig, "Built an AI agent using LangChain and Kubernetes.", ["LangChain"])
    assert checks.check_bullet(orig, "Built an AI agent using LangChain, cutting latency by 40%.", ["LangChain"])


def test_checks_allow_project_tech_but_not_other_projects_tech():
    orig = "Built an API for the agent."
    assert checks.check_bullet(orig, "Built a FastAPI service for the agent.", ["FastAPI"]) == []
    assert checks.check_bullet(orig, "Built a FastAPI service for the agent.", ["Flask"])   # FastAPI not in tech


# ---------- pipeline behaviour ----------
def test_good_rewrite_is_accepted(tmp_path):
    llm = Scripted(rewrite=lambda i: i["original"].replace("Built ", "Engineered ", 1),
                   verify=lambda i: BulletVerdict(id=i["id"], faithful=True, problem=""))
    s = run(llm, tmp_path)
    assert s["provenance"]["tt-1"] == "rewritten"
    assert s["final_text"]["tt-1"].startswith("Engineered")
    assert Path(s["pdf_path"]).exists() and Path(s["tex_path"]).exists()


def test_fabricated_tech_is_rejected_retried_then_original_kept(tmp_path):
    llm = Scripted(rewrite=lambda i: i["original"].rstrip(".") + " using Kubernetes.")
    s = run(llm, tmp_path)
    assert llm.rewrite_calls == 3                       # first try + 2 retries
    for bid, orig in s["originals"].items():
        assert s["final_text"][bid] == orig             # nothing invented reached the resume
        assert "rejected" in s["provenance"][bid]
    assert "Kubernetes" not in Path(s["tex_path"]).read_text()


def test_llm_verifier_veto_keeps_original(tmp_path):
    llm = Scripted(rewrite=lambda i: i["original"].replace("reliable", "robust"),
                   verify=lambda i: BulletVerdict(id=i["id"], faithful=False, problem="meaning changed"))
    s = run(llm, tmp_path)
    assert s["final_text"]["tt-3"] == s["originals"]["tt-3"]
    assert "meaning changed" in s["provenance"]["tt-3"]


def test_contact_details_never_sent_to_llm(tmp_path):
    llm = Scripted(rewrite=lambda i: i["original"])
    run(llm, tmp_path)
    blob = " ".join(llm.prompts)
    for secret in ("9579146611", "rajnandinik05", "7.28", "Modern College", "drive.google.com"):
        assert secret not in blob


def test_one_page_fit_trims_lowest_ranked_first(tmp_path):
    s = run(Scripted(), tmp_path, pages=(2, 2, 1))
    assert len(s["removed"]) == 2 and s["pages"] == 1
    assert sum(len(p["bullets"]) for p in s["final_data"]["projects"]) == 11 - 2


def test_skills_are_reordered_never_added(tmp_path):
    s = run(Scripted(), tmp_path)
    before = {c["category"]: sorted(c["items"]) for c in MASTER["skills"]}
    after = {c["category"]: sorted(c["items"]) for c in s["final_data"]["skills"]}
    assert before == after


def test_coverage_separates_missing_from_hidden():
    jd = {"must_have_skills": ["Python", "Kubernetes"], "nice_to_have_skills": ["Streamlit"], "keywords": []}
    master = {"skills": [{"category": "x", "items": ["Python", "Streamlit"]}], "projects": [], "internships": [],
              "achievements": []}
    final = {"skills": [{"category": "x", "items": ["Python"]}], "projects": [], "internships": [],
             "achievements": []}
    cov = report.coverage(jd, final, master)
    assert cov["must_have"]["covered"] == ["Python"]
    assert cov["must_have"]["not_in_master"] == ["Kubernetes"]
    assert cov["nice_to_have"]["in_master_not_shown"] == ["Streamlit"]


def test_short_jd_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(FakeLLM(), MASTER, "too short", out_root=tmp_path, compile_fn=fake_compile((1,)))


# ---------- Gemini wrapper retry logic (no network) ----------
def make_gemini(responses, models=("m1", "m2")):
    g = GeminiLLM.__new__(GeminiLLM)
    g.models, g.retries, g.sleeps = list(models), 3, []
    g.sleep = g.sleeps.append
    queue = list(responses)

    def generate_content(**kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    g.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return g


OK = JDAnalysis(role_title="r", seniority="s", must_have_skills=[], nice_to_have_skills=[], keywords=[],
                responsibilities=[])
GOOD = SimpleNamespace(parsed=OK, text=None)


def test_gemini_retries_rate_limit_then_succeeds():
    g = make_gemini([errors.ClientError(429, {"error": {}}), errors.ClientError(429, {"error": {}}), GOOD])
    assert g.structured(system="s", prompt="p", schema=JDAnalysis) == OK
    assert g.sleeps == [4, 8]


def test_gemini_falls_back_to_next_model_on_404():
    g = make_gemini([errors.ClientError(404, {"error": {}}), GOOD])
    assert g.structured(system="s", prompt="p", schema=JDAnalysis) == OK


def test_gemini_bad_request_fails_fast_and_exhausted_retries_raise():
    with pytest.raises(LLMError):
        make_gemini([errors.ClientError(400, {"error": {}})]).structured(system="s", prompt="p", schema=JDAnalysis)
    with pytest.raises(LLMError):
        make_gemini([errors.ClientError(429, {"error": {}})] * 6).structured(system="s", prompt="p",
                                                                             schema=JDAnalysis)


def test_gemini_parses_text_when_parsed_is_missing():
    resp = SimpleNamespace(parsed=None, text=OK.model_dump_json())
    assert make_gemini([resp]).structured(system="s", prompt="p", schema=JDAnalysis) == OK


# ---------- real LaTeX build (skipped if LaTeX or fontawesome5 is missing) ----------
def _have_latex():
    if not shutil.which("pdflatex"):
        return False
    return subprocess.run(["kpsewhich", "fontawesome5.sty"], capture_output=True, text=True).stdout.strip() != ""


@pytest.mark.skipif(not _have_latex(), reason="pdflatex + fontawesome5 not available")
def test_real_pdf_is_one_page(tmp_path):
    from render.latex import compile_pdf
    s = generate(FakeLLM(), MASTER, JD, "Acme", "GenAI", out_root=tmp_path, compile_fn=compile_pdf)
    assert s["pages"] == 1 and Path(s["pdf_path"]).read_bytes().startswith(b"%PDF")