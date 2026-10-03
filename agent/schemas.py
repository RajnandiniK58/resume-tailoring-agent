"""Pydantic schemas = the exact JSON shapes the LLM must return.
(No defaults/Optionals on purpose: Gemini's structured output handles plain required fields most reliably.)"""
from typing import List

from pydantic import BaseModel


class JDAnalysis(BaseModel):
    role_title: str
    seniority: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    keywords: List[str]
    responsibilities: List[str]


class SelectionPlan(BaseModel):
    project_ids: List[str]   # best-fit project first
    bullet_ids: List[str]    # most relevant bullets first


class RewrittenBullet(BaseModel):
    id: str
    text: str


class RewriteResult(BaseModel):
    bullets: List[RewrittenBullet]


class BulletVerdict(BaseModel):
    id: str
    faithful: bool
    problem: str             # empty string when faithful


class VerifyResult(BaseModel):
    verdicts: List[BulletVerdict]