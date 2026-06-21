"""
main.py
=======
SkillGaps.AI – Ethical Workforce Transition & ML Platform
----------------------------------------------------------
Orchestration entry point.  Runs the complete pipeline end-to-end:

  Phase 0  │  Print the conceptual MySQL schema
  Phase 1  │  Generate synthetic employee + course data
  Phase 2  │  Run the Recommendation Pipeline
             │    → Embeddings (sentence-transformers)
             │    → Cosine Similarity
             │    → Ethical Fairness Layer (demographic parity)
             │    → XAI Explanations
  Phase 3  │  Print a structured results report
  Phase 4  │  Export artefacts to CSV (mimics DB insert)
  Phase 5  │  Print a full Ethics & Transparency Audit Summary

Usage
-----
  python main.py

Dependencies
------------
See requirements.txt.  Install with:
  pip install -r requirements.txt
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from data_generation import (
    MYSQL_SCHEMA,
    generate_courses,
    generate_employees,
)
from models import RecommendationPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_N               = 3      # Recommendations per employee
MODEL_NAME          = "all-MiniLM-L6-v2"
PROTECTED_ATTRIBUTE = "gender"
OUTPUT_DIR          = "outputs"


# ===========================================================================
# Utility helpers
# ===========================================================================

def _divider(char: str = "─", width: int = 70) -> str:
    return char * width


def _section(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _wrap(text: str, indent: int = 4) -> str:
    """Wrap long text at 80 chars for terminal readability."""
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix,
                         subsequent_indent=prefix)


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ===========================================================================
# Phase 3 – Formatted Results Report
# ===========================================================================

def print_results_report(logs: pd.DataFrame, top_k: int = 10) -> None:
    """
    Print a readable sample of recommendations (top_k employees by risk).
    """
    _section("PHASE 3 │ RECOMMENDATION RESULTS REPORT")

    # Show the top_k highest-risk employees and their recommendations
    high_risk_ids = (
        logs.sort_values("risk_score", ascending=False)["employee_id"]
        .drop_duplicates()
        .head(top_k)
        .tolist()
    )

    for emp_id in high_risk_ids:
        emp_logs = logs[logs["employee_id"] == emp_id].sort_values("rank")
        first    = emp_logs.iloc[0]

        print(f"\n{'─' * 70}")
        print(
            f"  Employee #{first['employee_id']:02d} │ {first['employee_name']}"
            f"  │ {first['gender']}  │ Risk: {first['risk_score']:.3f}"
        )
        print(_divider())

        for _, rec in emp_logs.iterrows():
            print(f"\n  Rank #{int(rec['rank'])}  ──  {rec['course_title']}")
            print(f"    Raw similarity   : {rec['raw_similarity']:.5f}")
            print(f"    Fairness score   : {rec['fairness_score']:.5f}  "
                  f"(Δ {rec['fairness_delta']:+.5f})")
            print(f"    Demographic group: {rec['demographic_group']}")
            # Print the XAI reason, indented for readability
            print()
            for line in rec["xai_reason"].split("\n"):
                print(f"    {line}")


# ===========================================================================
# Phase 4 – CSV Export  (simulates MySQL INSERT)
# ===========================================================================

def export_artefacts(
    employees: pd.DataFrame,
    courses:   pd.DataFrame,
    logs:      pd.DataFrame,
    out_dir:   str,
) -> dict:
    """
    Write three CSV files mirroring the three MySQL tables.

    Returns a dict of {table_name: filepath}.
    """
    _section("PHASE 4 │ EXPORTING ARTEFACTS  (CSV → mirrors MySQL INSERT)")
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    for name, df in [
        ("employees",     employees),
        ("nptel_courses", courses),
        ("upskilling_logs", logs),
    ]:
        fpath = os.path.join(out_dir, f"{name}_{ts}.csv")
        df.to_csv(fpath, index=False)
        print(f"  ✓  {name:20s}  →  {fpath}  ({len(df)} rows)")
        paths[name] = fpath

    return paths


# ===========================================================================
# Phase 5 – Ethics & Transparency Audit Summary
# ===========================================================================

def print_ethics_audit(
    logs:      pd.DataFrame,
    employees: pd.DataFrame,
    pipeline:  RecommendationPipeline,
) -> None:
    """
    Print a structured ethics audit that an IIT research panel or external
    auditor can use to verify fairness, transparency, and accountability.
    """
    _section("PHASE 5 │ ETHICS & TRANSPARENCY AUDIT SUMMARY")

    fl      = pipeline.fairness_layer
    attr    = fl.protected_attribute

    # ── 5a. Demographic composition ──────────────────────────────────────
    print("\n  ── 5a. Workforce Demographic Composition ──")
    composition = employees[attr].value_counts()
    total        = len(employees)
    for group, count in composition.items():
        print(f"    {group:25s} : {count:3d} employees  ({count/total:.1%})")

    # ── 5b. Fairness layer adjustments ───────────────────────────────────
    print(f"\n  ── 5b. Ethical Fairness Layer  (protected attribute: '{attr}') ──")
    print(f"    Demographic parity score BEFORE adjustment : "
          f"{fl.parity_score_before:.4f}")

    post_parity = logs["fair_dist_score"].iloc[0]
    print(f"    Demographic parity score AFTER  adjustment : {post_parity:.4f}")
    print(f"    Parity scale: 0.0 = severe disparity │ 1.0 = perfect parity")

    print("\n    Group-level adjustments applied:")
    for group, adj in sorted(fl.group_adjustments.items()):
        direction = "BOOST  ▲" if adj > 0 else ("REDUCE ▼" if adj < 0 else "NONE   –")
        print(f"      {group:25s} : {adj:+.5f}  ({direction})")

    # ── 5c. Recommendation distribution ─────────────────────────────────
    print(f"\n  ── 5c. Top-1 Recommendation Distribution by '{attr}' ──")
    top1 = logs[logs["rank"] == 1]
    dist  = top1[attr].value_counts()
    for group, count in dist.items():
        group_total = (employees[attr] == group).sum()
        rate         = count / group_total if group_total else 0
        print(f"    {group:25s} : {count:3d} / {group_total:3d}  ({rate:.1%} recommendation rate)")

    # ── 5d. Score statistics ──────────────────────────────────────────────
    print("\n  ── 5d. Score Statistics ──")
    print(f"    Raw similarity  │ mean={logs['raw_similarity'].mean():.4f}  "
          f"std={logs['raw_similarity'].std():.4f}  "
          f"min={logs['raw_similarity'].min():.4f}  "
          f"max={logs['raw_similarity'].max():.4f}")
    print(f"    Fairness score  │ mean={logs['fairness_score'].mean():.4f}  "
          f"std={logs['fairness_score'].std():.4f}  "
          f"min={logs['fairness_score'].min():.4f}  "
          f"max={logs['fairness_score'].max():.4f}")
    print(f"    Fairness delta  │ mean={logs['fairness_delta'].mean():+.5f}  "
          f"std={logs['fairness_delta'].std():.5f}")

    # ── 5e. Most-recommended courses ─────────────────────────────────────
    print("\n  ── 5e. Most-Recommended Courses (all ranks) ──")
    course_counts = logs["course_title"].value_counts().head(5)
    for title, count in course_counts.items():
        print(f"    {count:3d}×  {title}")

    # ── 5f. XAI coverage check ───────────────────────────────────────────
    print("\n  ── 5f. XAI Transparency Coverage ──")
    has_explanation  = logs["xai_reason"].str.len() > 0
    print(f"    Recommendations with explanation : {has_explanation.sum()} / {len(logs)}")
    has_fairness_note = logs["xai_reason"].str.contains("FAIRNESS ADJUSTMENT")
    print(f"    Explanations with fairness note  : {has_fairness_note.sum()} / {len(logs)}")

    # ── 5g. Ethical principles compliance ────────────────────────────────
    print("\n  ── 5g. Ethical Principles Compliance Checklist ──")
    checks = {
        "Transparency (XAI reasons logged)":
            has_explanation.all(),
        "Fairness (parity score > 0.85 after adjustment)":
            post_parity >= 0.85,
        "Non-replacement (no termination data in logs)":
            "termination" not in " ".join(logs.columns).lower(),
        "Accountability (raw & adjusted scores both recorded)":
            "raw_similarity" in logs.columns and "fairness_score" in logs.columns,
        "Auditability (demographic group recorded per log)":
            "demographic_group" in logs.columns,
    }
    for principle, passed in checks.items():
        status = "✓  PASS" if passed else "✗  FAIL"
        print(f"    [{status}]  {principle}")

    print(f"\n{'═' * 70}")
    print("  AUDIT COMPLETE  –  SkillGaps.AI Ethical Workforce Transition Platform")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """
    Full pipeline entry point.
    """

    # ── Phase 0 : MySQL Schema ────────────────────────────────────────────
    _section("PHASE 0 │ CONCEPTUAL MYSQL DATABASE SCHEMA")
    print(MYSQL_SCHEMA)

    # ── Phase 1 : Data Generation ─────────────────────────────────────────
    _section("PHASE 1 │ SYNTHETIC DATA GENERATION")
    print("  Generating 50 employee records …")
    employees = generate_employees(50)
    print(f"  ✓  {len(employees)} employees created.")

    print("  Generating 10 NPTEL course records …")
    courses = generate_courses()
    print(f"  ✓  {len(courses)} courses created.")

    # Quick demographic summary
    print("\n  Demographic snapshot (intentional historical imbalance):")
    for gender, count in employees["gender"].value_counts().items():
        print(f"    {gender:12s} : {count} employees")

    print("\n  Risk score distribution:")
    risk_bins = pd.cut(employees["risk_score"], bins=[0, 0.3, 0.55, 1.0],
                       labels=["Low (<0.30)", "Medium (0.30–0.55)", "High (>0.55)"])
    for label, count in risk_bins.value_counts().sort_index().items():
        print(f"    {label:25s} : {count} employees")

    # ── Phase 2 : ML Pipeline ─────────────────────────────────────────────
    _section("PHASE 2 │ ML & ETHICAL RECOMMENDATION PIPELINE")

    pipeline = RecommendationPipeline(
        top_n=TOP_N,
        model_name=MODEL_NAME,
        protected_attribute=PROTECTED_ATTRIBUTE,
    )
    logs = pipeline.run(employees, courses)

    # ── Phase 3 : Results Report ──────────────────────────────────────────
    print_results_report(logs, top_k=6)

    # ── Phase 4 : Export ──────────────────────────────────────────────────
    out_dir = _ensure_output_dir()
    paths   = export_artefacts(employees, courses, logs, out_dir)

    # ── Phase 5 : Ethics Audit ────────────────────────────────────────────
    print_ethics_audit(logs, employees, pipeline)

    # ── Done ──────────────────────────────────────────────────────────────
    print("  Run complete.  Output files written to:", out_dir)
    for name, path in paths.items():
        print(f"    {name:20s} → {path}")


if __name__ == "__main__":
    main()
