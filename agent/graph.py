"""Wires the nodes into a LangGraph workflow and exposes one function: generate()."""
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.nodes import make_nodes, slug
from render.latex import ROOT, compile_pdf


class State(TypedDict, total=False):
    jd_text: str
    out_dir: str
    jd: dict
    selection: List[dict]
    originals: Dict[str, str]
    rewrites: Dict[str, str]
    pending: List[str]
    verified: Dict[str, str]
    failed: Dict[str, str]
    attempts: int
    verifier_unavailable: bool
    final_text: Dict[str, str]
    provenance: Dict[str, str]
    final_data: dict
    removed: List[str]
    pages: int
    pdf_path: str
    tex_path: str
    coverage: dict
    report_md: str
    warnings: List[str]


def build_graph(llm, master: dict, compile_fn: Callable = compile_pdf):
    n = make_nodes(llm, master, compile_fn)
    g = StateGraph(State)
    for name in ("analyze_jd", "select_content", "rewrite_bullets", "verify_bullets",
                 "assemble_render", "make_report"):
        g.add_node(name, n[name])
    g.add_edge(START, "analyze_jd")
    g.add_edge("analyze_jd", "select_content")
    g.add_edge("select_content", "rewrite_bullets")
    g.add_edge("rewrite_bullets", "verify_bullets")
    g.add_conditional_edges("verify_bullets", n["route"],
                            {"rewrite_bullets": "rewrite_bullets", "assemble_render": "assemble_render"})
    g.add_edge("assemble_render", "make_report")
    g.add_edge("make_report", END)
    return g.compile()


def generate(llm, master: dict, jd_text: str, company: str = "", role: str = "",
             out_root: Optional[Path] = None, compile_fn: Callable = compile_pdf) -> dict:
    """Run the whole pipeline. Returns the final state (pdf_path, tex_path, report_md, coverage, warnings, ...)."""
    out_root = Path(out_root) if out_root else ROOT / "outputs"
    label = slug(f"{company} {role}") or "resume"
    out_dir = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    graph = build_graph(llm, master, compile_fn)
    return graph.invoke({"jd_text": jd_text, "out_dir": str(out_dir)})