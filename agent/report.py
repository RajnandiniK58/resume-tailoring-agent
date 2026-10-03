"""Keyword coverage + a human-readable report. Everything here is deterministic (no LLM)."""
import re
from typing import Dict, List


def has_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term.strip())}(?![A-Za-z0-9])", text, re.I))


def strip_markup(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)\^?", r"\1", s)
    return re.sub(r"[*`]", "", s)


def resume_text(data: dict, include_all_projects: bool = True) -> str:
    parts: List[str] = []
    for s in data.get("skills", []):
        parts += [s["category"], *s["items"]]
    for p in data.get("projects", []):
        parts += [p["title"], *p["tech"], *[b["text"] for b in p["bullets"]]]
    for i in data.get("internships", []):
        parts += [i["title"], i["org"], *[b["text"] for b in i["bullets"]]]
    parts += [a["text"] for a in data.get("achievements", [])]
    return strip_markup("\n".join(parts))


def coverage(jd: dict, final_data: dict, master: dict) -> Dict[str, dict]:
    final_txt = resume_text(final_data)
    master_txt = resume_text(master)
    out = {}
    for label, key in (("must_have", "must_have_skills"), ("nice_to_have", "nice_to_have_skills"),
                       ("keywords", "keywords")):
        covered, hidden, absent = [], [], []
        for term in dict.fromkeys(t.strip() for t in jd.get(key, []) if t.strip()):
            if has_term(final_txt, term):
                covered.append(term)
            elif has_term(master_txt, term):
                hidden.append(term)       # you have it in master.yaml but it is not on this resume
            else:
                absent.append(term)       # not anywhere in master.yaml
        out[label] = {"covered": covered, "in_master_not_shown": hidden, "not_in_master": absent}
    return out


def render_report(state: dict, cov: Dict[str, dict]) -> str:
    jd = state["jd"]
    lines = [f"# Tailoring report -- {jd.get('role_title', 'role')}", ""]
    for label, title in (("must_have", "Must-have skills"), ("nice_to_have", "Nice-to-have skills"),
                         ("keywords", "JD keywords")):
        c = cov[label]
        total = sum(len(v) for v in c.values())
        lines += [f"## {title}: {len(c['covered'])}/{total} on this resume", ""]
        if c["covered"]:
            lines.append("- On resume: " + ", ".join(c["covered"]))
        if c["in_master_not_shown"]:
            lines.append("- In your master.yaml but not on this resume: " + ", ".join(c["in_master_not_shown"]))
        if c["not_in_master"]:
            lines.append("- Not in your master.yaml at all: " + ", ".join(c["not_in_master"]))
        lines.append("")
    lines += ["> Missing items are never added automatically. If you genuinely have a skill, add it to "
              "`data/master.yaml` and re-run.", "", "## Bullet changes", ""]
    for bid, status in state.get("provenance", {}).items():
        orig, final = state["originals"][bid], state["final_text"][bid]
        lines.append(f"- **{bid}** ({status})")
        if orig != final:
            lines += [f"  - before: {orig}", f"  - after: {final}"]
    if state.get("removed"):
        lines += ["", "## Trimmed to fit one page", ""] + [f"- {r}" for r in state["removed"]]
    if state.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in state["warnings"]]
    return "\n".join(lines) + "\n"