"""Streamlit app:  streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st
import yaml

from agent.fake_llm import FakeLLM
from agent.graph import generate
from agent.llm import GeminiLLM

ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "data" / "master.yaml"

st.set_page_config(page_title="Resume Tailoring Agent", layout="wide")
st.title("Resume Tailoring Agent")
st.caption("Paste a job description. The agent reorders and rewords your resume to fit it, "
           "without adding anything that is not already in data/master.yaml.")

mock = st.sidebar.checkbox("Mock mode (no AI, no API key)", value=os.getenv("RESUME_AGENT_MOCK") == "1")
if mock:
    st.sidebar.info("Mock mode only tests the pipeline. Bullets are NOT tailored.")

c1, c2 = st.columns(2)
company = c1.text_input("Company (used in the output folder name)")
role = c2.text_input("Role (used in the output folder name)")
jd = st.text_area("Job description", height=300)

if st.button("Generate tailored resume", type="primary"):
    if not jd.strip():
        st.error("Paste a job description first.")
    else:
        try:
            master = yaml.safe_load(MASTER_PATH.read_text(encoding="utf-8"))
            llm = FakeLLM() if mock else GeminiLLM()
            with st.spinner("Analysing the JD, selecting, rewriting, fact-checking, building the PDF..."):
                st.session_state["result"] = generate(llm, master, jd, company, role)
        except Exception as e:  # show any failure in the page instead of a stack trace
            st.session_state.pop("result", None)
            st.error(str(e))

res = st.session_state.get("result")
if res:
    pdf_path, tex_path = Path(res["pdf_path"]), Path(res["tex_path"])
    st.success(f"Built {pdf_path.name} ({res['pages']} page{'s' if res['pages'] != 1 else ''}). "
               f"Saved in {pdf_path.parent}")
    for w in res.get("warnings", []):
        st.warning(w)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download PDF", pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
    d2.download_button("Download .tex", tex_path.read_text(encoding="utf-8"), file_name=tex_path.name)
    d3.download_button("Download report", res["report_md"], file_name="report.md")

    st.subheader("Keyword coverage")
    cov = res["coverage"]
    for label, title in (("must_have", "Must-have"), ("nice_to_have", "Nice-to-have"), ("keywords", "Keywords")):
        c = cov[label]
        total = sum(len(v) for v in c.values())
        st.markdown(f"**{title}: {len(c['covered'])}/{total} on the resume**")
        if c["in_master_not_shown"]:
            st.write("In your master.yaml but not on this resume:", ", ".join(c["in_master_not_shown"]))
        if c["not_in_master"]:
            st.write("Not in your master.yaml at all (add only if you truly have it):",
                     ", ".join(c["not_in_master"]))

    st.subheader("What changed")
    present = {b["id"] for p in res["final_data"]["projects"] for b in p["bullets"]}
    for bid, status in res["provenance"].items():
        if bid in present:
            with st.expander(f"{bid}: {status}"):
                st.write("Before:", res["originals"][bid])
                st.write("After:", res["final_text"][bid])
    for note in res.get("removed", []):
        st.info(note)