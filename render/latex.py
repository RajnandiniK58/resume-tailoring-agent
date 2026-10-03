"""Render master data into a LaTeX resume with Jinja2 and compile it to PDF."""
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent

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
    # straight double quotes -> LaTeX quotes
    return re.sub(r'"([^"]*)"', r"``\1''", out)


def tex(text: str) -> str:
    """Turn light markup + plain text into safe LaTeX."""
    text = str(text)
    out, pos = [], 0
    for m in _MARKUP.finditer(text):
        out.append(_esc(text[pos:m.start()]))
        link_label, url, icon, bold, ital, code = m.groups()
        if link_label:
            label = tex(link_label)
            out.append(rf"\href{{{url}}}{{{label}}}")
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
    """Compile with pdflatex (run twice for stable layout). Returns the PDF path."""
    engine = shutil.which("pdflatex") or shutil.which("tectonic")
    if engine is None:
        raise RuntimeError("No LaTeX engine found. Install TeX Live/MiKTeX or Tectonic.")
    cmd = [engine, "-interaction=nonstopmode", tex_path.name]
    if "tectonic" in engine:
        cmd = [engine, tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise RuntimeError(f"Compile failed; see {tex_path.with_suffix('.log')}")
    return pdf


def page_count(pdf_path: Path) -> int:
    """Read the page count LaTeX itself reports in the .log file."""
    log = pdf_path.with_suffix(".log").read_text(errors="ignore")
    m = re.search(r"Output written on .*? \((\d+) pages?", log)
    return int(m.group(1)) if m else 0


def build(master_path: Path, out_dir: Path, name: str = "resume") -> Path:
    data = yaml.safe_load(Path(master_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(render_tex(data), encoding="utf-8")
    pdf = compile_pdf(tex_path)
    pages = page_count(pdf)
    print(f"Built {pdf} ({pages} page{'s' if pages != 1 else ''})")
    if pages > 1:
        print("WARNING: resume is longer than one page.")
    return pdf


if __name__ == "__main__":
    build(ROOT / "data" / "master.yaml", ROOT / "outputs" / "baseline", "resume")