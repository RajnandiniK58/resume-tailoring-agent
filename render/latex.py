"""Render resume data into LaTeX with Jinja2, compile to PDF, and fit it to one page."""
import copy
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent
MIN_BULLETS_PER_PROJECT = 2

_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESC_RE = re.compile("|".join(re.escape(k) for k in _ESCAPES))

# [label](url)^?  |  **bold**  |  *italic*  |  `code`
_MARKUP = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)(\^)?"
    r"|\*\*(.+?)\*\*"
    r"|(?<!\*)\*(?!\*)(.+?)\*"
    r"|`([^`]+)`"
)


def _esc(text: str) -> str:
    out = _ESC_RE.sub(lambda m: _ESCAPES[m.group(0)], text)
    return re.sub(r'"([^"]*)"', r"``\1''", out)  # straight quotes -> LaTeX quotes


def tex(text: str) -> str:
    """Turn light markup + plain text into safe LaTeX."""
    text = str(text)
    out, pos = [], 0
    for m in _MARKUP.finditer(text):
        out.append(_esc(text[pos:m.start()]))
        link_label, url, icon, bold, ital, code = m.groups()
        if link_label:
            out.append(rf"\href{{{url}}}{{{tex(link_label)}}}")
            if icon:
                out.append(r" \faExternalLink*")
        elif bold:
            out.append(rf"\textbf{{{tex(bold)}}}")
        elif ital:
            out.append(rf"\textit{{{tex(ital)}}}")
        elif code:
            out.append(rf"\texttt{{{_esc(code)}}}")
        pos = m.end()
    out.append(_esc(text[pos:]))
    return "".join(out)


def make_env() -> Environment:
    # LaTeX-friendly delimiters so Jinja never clashes with { } in LaTeX.
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        block_start_string=r"\BLOCK{", block_end_string="}",
        variable_start_string=r"\VAR{", variable_end_string="}",
        comment_start_string=r"\#{", comment_end_string="}",
        trim_blocks=True, lstrip_blocks=True, autoescape=False,
    )
    env.filters["tex"] = tex
    return env


def render_tex(data: dict, template: str = "resume.tex.j2") -> str:
    return make_env().get_template(template).render(**data)


def compile_pdf(tex_path: Path) -> Path:
    """Compile with pdflatex (twice, for stable layout). Raises with the LaTeX error if it fails."""
    engine = shutil.which("pdflatex") or shutil.which("tectonic")
    if engine is None:
        raise RuntimeError("No LaTeX engine found. Install TeX Live/MiKTeX or Tectonic.")
    pdf = tex_path.with_suffix(".pdf")
    log = tex_path.with_suffix(".log")
    pdf.unlink(missing_ok=True)  # never mistake an old PDF for a fresh build
    cmd = [engine, tex_path.name] if "tectonic" in engine else [engine, "-interaction=nonstopmode", tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
    if not pdf.exists():
        errors = [l for l in log.read_text(errors="ignore").splitlines() if l.startswith("!")] if log.exists() else []
        raise RuntimeError("LaTeX compile failed: " + (errors[0] if errors else f"see {log}"))
    return pdf


def page_count(pdf_path: Path) -> int:
    """Read the page count LaTeX itself reports in the .log file (tectonic: falls back to scanning the PDF)."""
    log = pdf_path.with_suffix(".log")
    if log.exists():
        m = re.search(r"Output written on .*? \((\d+) pages?", log.read_text(errors="ignore"))
        if m:
            return int(m.group(1))
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))


def _trim_one(data: dict):
    """Remove the least important content (the end of the list = lowest ranked). Returns a description or None."""
    projects = data["projects"]
    for p in reversed(projects):
        if len(p["bullets"]) > MIN_BULLETS_PER_PROJECT:
            b = p["bullets"].pop()
            return f"Dropped a bullet from '{p['title']}' to fit one page: {b['text'][:70]}..."
    if len(projects) > 1:
        p = projects.pop()
        return f"Dropped project '{p['title']}' to fit one page"
    return None


def build_fit(data: dict, out_dir: Path, name: str, compile_fn=compile_pdf, max_pages: int = 1):
    """Render + compile; if too long, trim lowest-ranked content and retry.
    Returns (pdf_path, final_data, removed_notes, pages)."""
    data = copy.deepcopy(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{name}.tex"
    removed = []
    while True:
        tex_path.write_text(render_tex(data), encoding="utf-8")
        pdf = compile_fn(tex_path)
        pages = page_count(pdf)
        if pages <= max_pages:
            return pdf, data, removed, pages
        note = _trim_one(data)
        if note is None:
            return pdf, data, removed, pages  # cannot shrink further
        removed.append(note)


def build(master_path: Path, out_dir: Path, name: str = "resume") -> Path:
    """Phase 0 behaviour: render the master data as-is."""
    data = yaml.safe_load(Path(master_path).read_text(encoding="utf-8"))
    pdf, _, _, pages = build_fit(data, out_dir, name, max_pages=99)
    print(f"Built {pdf} ({pages} page{'s' if pages != 1 else ''})")
    if pages > 1:
        print("WARNING: resume is longer than one page.")
    return pdf


if __name__ == "__main__":
    build(ROOT / "data" / "master.yaml", ROOT / "outputs" / "baseline", "resume")