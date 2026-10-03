"""Offline stand-in for the LLM, so you can test the whole pipeline (graph, LaTeX, PDF, app) with no API key.
It does NOT tailor anything: bullets are returned unchanged. Used by `--mock` and by the tests."""
import json
import re

from agent.schemas import (BulletVerdict, JDAnalysis, RewriteResult, RewrittenBullet,
                           SelectionPlan, VerifyResult)

_KNOWN = ["Python", "SQL", "Java", "JavaScript", "TypeScript", "FastAPI", "Flask", "Streamlit", "Docker",
          "Kubernetes", "AWS", "GCP", "Azure", "LangChain", "LangGraph", "RAG", "LLM", "LLMs", "FAISS",
          "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy", "Git", "REST", "NLP", "MLOps"]


def _json_after(marker: str, prompt: str):
    return json.loads(prompt.split(marker, 1)[1])


class FakeLLM:
    is_mock = True

    def structured(self, *, system, prompt, schema, temperature=0.2):
        if schema is JDAnalysis:
            jd = prompt.split("<job_description>", 1)[1]
            found = [k for k in _KNOWN if re.search(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", jd, re.I)]
            return JDAnalysis(role_title="Target role (mock)", seniority="unspecified",
                              must_have_skills=found[:3], nice_to_have_skills=found[3:],
                              keywords=found, responsibilities=[])
        if schema is SelectionPlan:
            projects = _json_after("PROJECTS_JSON:\n", prompt)
            return SelectionPlan(project_ids=[p["id"] for p in projects],
                                 bullet_ids=[b["id"] for p in projects for b in p["bullets"]])
        if schema is RewriteResult:
            items = _json_after("ITEMS_JSON:\n", prompt)
            return RewriteResult(bullets=[RewrittenBullet(id=i["id"], text=i["original"]) for i in items])
        if schema is VerifyResult:
            items = _json_after("ITEMS_JSON:\n", prompt)
            return VerifyResult(verdicts=[BulletVerdict(id=i["id"], faithful=True, problem="") for i in items])
        raise ValueError(f"FakeLLM cannot handle {schema}")