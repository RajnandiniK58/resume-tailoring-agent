"""Command line:  python run.py --jd jd.txt --company Acme --role "GenAI Engineer"
Add --mock to test everything without an API key (bullets are returned unchanged)."""
import argparse
import sys
from pathlib import Path

import yaml

from agent.fake_llm import FakeLLM
from agent.graph import generate
from agent.llm import GeminiLLM, LLMError

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Tailor your LaTeX resume to a job description.")
    ap.add_argument("--jd", required=True, help="path to a text file containing the job description ('-' = stdin)")
    ap.add_argument("--company", default="")
    ap.add_argument("--role", default="")
    ap.add_argument("--master", default=str(ROOT / "data" / "master.yaml"))
    ap.add_argument("--mock", action="store_true", help="no AI, no API key; for testing the pipeline only")
    args = ap.parse_args()

    jd_text = sys.stdin.read() if args.jd == "-" else Path(args.jd).read_text(encoding="utf-8")
    master = yaml.safe_load(Path(args.master).read_text(encoding="utf-8"))
    try:
        llm = FakeLLM() if args.mock else GeminiLLM()
        state = generate(llm, master, jd_text, args.company, args.role)
    except (LLMError, RuntimeError, ValueError) as e:
        sys.exit(f"Error: {e}")

    print(f"PDF:    {state['pdf_path']}  ({state['pages']} page{'s' if state['pages'] != 1 else ''})")
    print(f"LaTeX:  {state['tex_path']}")
    print(f"Report: {Path(state['out_dir']) / 'report.md'}")
    for w in state.get("warnings", []):
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()