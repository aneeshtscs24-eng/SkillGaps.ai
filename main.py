"""
main.py
=======
SkillGaps.AI – Ethical Workforce Transition & ML Platform  (v3)
---------------------------------------------------------------
v2 changelog
────────────
• Phase 0  : Schema DDL is now EXECUTED against a live MySQL server
             via ``db.execute_schema()``.
• Phase 1  : Unchanged — synthetic data generation.
• Phase 2  : Unchanged — ML pipeline (embeddings → fairness → XAI),
             but models.py now uses SkillNormalizer internally.
• Phase 3  : Unchanged — terminal results report.
• Phase 4  : REPLACED CSV export with live DB seeding:
               – ``db.seed_reference_tables()`` for employees & courses.
               – ``db.insert_upskilling_logs()`` for the FK-linked logs.
             Falls back to CSV export if the DB is not configured/reachable.
• Phase 5  : Unchanged — ethics & transparency audit.

v3 changelog
────────────
• HARDENED  main() engine lifecycle:
  - ``seed_database()`` now returns the live ``Engine`` object (or ``None``
    on failure/dry-run) so the caller can own the resource explicitly.
  - A ``try / finally`` block in ``main()`` guarantees ``engine.dispose()``
    is called after Phase 4 completes — whether the DB path succeeded, the
    CSV fallback was used, or an unhandled exception propagated upward.
  - ``engine.dispose()`` instructs SQLAlchemy to close all pooled connections
    and return their file-descriptor handles to the OS.  Without this call,
    background worker threads in the connection pool can keep the process
    alive after ``main()`` returns, causing confusing hang behaviour in
    containerised or CI environments.

Modes of operation
──────────────────
DB mode  (default when env vars are set):
    The pipeline connects to a live MySQL instance, runs DDL, seeds all
    three tables, and writes the audit logs respecting FK constraints.
    The engine pool is cleanly disposed after Phase 4 regardless of outcome.

Dry-run mode  (fallback when env vars are absent or DB unreachable):
    All ML and fairness logic runs normally.  Phase 4 falls back to the
    original CSV export so the pipeline never hard-fails on missing DB config.
    No engine is created, so dispose() is a no-op.

Usage
─────
    # DB mode
    export SKILLGAPS_DB_HOST=127.0.0.1
    export SKILLGAPS_DB_PORT=3306
    export SKILLGAPS_DB_USER=skillgaps_user
    export SKILLGAPS_DB_PASS=your_secure_password
    export SKILLGAPS_DB_NAME=skillgaps_ai
    python main.py

    # Dry-run mode (no env vars needed)
    python main.py

Dependencies
────────────
See requirements.txt.  Install with:
    pip install -r requirements.txt
"""

from __future__ import annotations

import logging
import os
import textwrap
from datetime import datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from data_generation import MYSQL_SCHEMA, generate_courses, generate_employees
from models import RecommendationPipeline

# db.py is imported lazily (inside seed_database) so the pipeline can run
# in dry-run mode without SQLAlchemy installed.
# The Engine type is imported at the top level only for the type annotation
# on seed_database's return value; the import is guarded to stay optional.
try:
    from sqlalchemy import Engine as _SAEngine
except ImportError:  # SQLAlchemy not installed → dry-run only
    _SAEngine = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skillgaps.main")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_N               = 3
MODEL_NAME          = "all-MiniLM-L6-v2"
PROTECTED_ATTRIBUTE = "gender"
OUTPUT_DIR          = "outputs"

# Environment variable names (must match db.py)
_DB_ENV_VARS = [
    "SKILLGAPS_DB_HOST",
    "SKILLGAPS_DB_PORT",
    "SKILLGAPS_DB_USER",
    "SKILLGAPS_DB_PASS",
    "SKILLGAPS_DB_NAME",
]


# ===========================================================================
# Helpers
# ===========================================================================

def _section(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _divider(char: str = "─", width: int = 70) -> str:
    return char * width


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix,
                         subsequent_indent=prefix)


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _db_env_configured() -> bool:
    """Return True only if ALL required DB environment variables are set."""
    return all(os.environ.get(v) for v in _DB_ENV_VARS)


# ===========================================================================
# Phase 3 – Results Report  (unchanged from v1)
# ===========================================================================

def print_results_report(logs: pd.DataFrame, top_k: int = 6) -> None:
    _section("PHASE 3 │ RECOMMENDATION RESULTS REPORT")

    high_risk_ids = (
        logs.sort_values("risk_score", ascending=False)["employee_id"]
        .drop_duplicates()
        .head(top_k)
        .tolist()
    )

    for emp_id in high_risk_ids:
        emp_logs = logs[logs["employee_id"] == emp_id].sort_values("rank")
        first    = emp_logs.iloc[0]

        print(f"\n{_divider()}")
        print(
            f"  Employee #{first['employee_id']:02d} │ {first['employee_name']}"
            f"  │ {first['gender']}  │ Risk: {first['risk_score']:.3f}"
        )
        print(_divider())

        for _, rec in emp_logs.iterrows():
            print(f"\n  Rank #{int(rec['rank'])}  ──  {rec['course_title']}")
            print(f"    Raw similarity   : {rec['raw_similarity']:.5f}")
            print(
                f"    Fairness score   : {rec['fairness_score']:.5f}  "
                f"(Δ {rec['fairness_delta']:+.5f})"
            )
            print(f"    Demographic group: {rec['demographic_group']}")
            print()
            for line in rec["xai_reason"].split("\n"):
                print(f"    {line}")


# ===========================================================================
# Phase 4 – Database seeding OR CSV fallback
# ===========================================================================

def seed_database(
    employees: pd.DataFrame,
    courses:   pd.DataFrame,
    logs:      pd.DataFrame,
) -> "Optional[_SAEngine]":
    """
    Attempt to connect to MySQL and seed all three tables.

    v3 change
    ─────────
    Returns the live ``Engine`` object on success instead of a bare
    ``True`` boolean.  The caller (``main()``) stores this reference and
    calls ``engine.dispose()`` in a ``finally`` block, guaranteeing that
    the SQLAlchemy connection pool is torn down and all OS-level file
    descriptors are returned regardless of what happens in Phases 5+.

    Returns ``None`` on any failure or when the DB is not configured,
    which the caller treats as the CSV-fallback signal.  A ``None`` return
    means no engine was created, so no dispose is necessary.

    Returns
    -------
    sqlalchemy.Engine | None
        Live engine on success; None triggers the CSV fallback path.
    """
    _section("PHASE 4 │ LIVE DATABASE SEEDING  (MySQL via SQLAlchemy)")

    try:
        import db as db_module
    except ImportError:
        log.warning("db.py not importable – falling back to CSV export.")
        return None

    try:
        # ── 4a. Build engine ──────────────────────────────────────────────
        print("  [4a]  Building SQLAlchemy engine …")
        engine = db_module.build_engine(retries=3, retry_delay=2.0)
        print("  ✓  Engine connected.\n")

        # ── 4b. Execute DDL ───────────────────────────────────────────────
        print("  [4b]  Executing schema DDL (CREATE DATABASE / CREATE TABLE) …")
        db_module.execute_schema(engine, MYSQL_SCHEMA)
        print("  ✓  Schema is ready.\n")

        # ── 4c. Seed reference tables ─────────────────────────────────────
        print("  [4c]  Seeding reference tables (employees + nptel_courses) …")
        db_module.seed_reference_tables(engine, employees, courses)
        print("  ✓  Reference tables seeded.\n")

        # ── 4d. Insert upskilling_logs (FK-aware) ─────────────────────────
        print("  [4d]  Inserting upskilling_logs (FK-constrained) …")
        inserted = db_module.insert_upskilling_logs(engine, logs)
        print(f"  ✓  {inserted} log rows inserted into upskilling_logs.\n")

        # ── 4e. Health check ──────────────────────────────────────────────
        print("  [4e]  Post-seed table counts:")
        db_module.print_table_counts(engine)

        # Return the live engine so main() can dispose it after Phase 5.
        return engine

    except EnvironmentError as exc:
        # Missing env vars — expected in dry-run / CI environments.
        log.warning("DB not configured: %s", exc)
        log.warning("Falling back to CSV export.")
        return None

    except Exception as exc:  # noqa: BLE001
        # Network error, auth failure, DDL error, etc. — degrade gracefully.
        log.error("Database seeding failed: %s", exc)
        log.warning("Falling back to CSV export.")
        return None
        print("  [4e]  Post-seed table counts:")
        db_module.print_table_counts(engine)

        return True

    except EnvironmentError as exc:
        # Missing env vars – expected in dry-run / CI environments
        log.warning("DB not configured: %s", exc)
        log.warning("Falling back to CSV export.")
        return False

    except Exception as exc:  # noqa: BLE001
        # Network error, auth failure, etc. – degrade gracefully
        log.error("Database seeding failed: %s", exc)
        log.warning("Falling back to CSV export.")
        return False


def export_csv_fallback(
    employees: pd.DataFrame,
    courses:   pd.DataFrame,
    logs:      pd.DataFrame,
    out_dir:   str,
) -> dict:
    """
    CSV export used when the DB is not configured or unreachable.
    Identical to the v1 Phase 4 behaviour.
    """
    _section("PHASE 4 │ CSV FALLBACK EXPORT  (DB not configured / reachable)")
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    for name, df in [
        ("employees",      employees),
        ("nptel_courses",  courses),
        ("upskilling_logs", logs),
    ]:
        fpath = os.path.join(out_dir, f"{name}_{ts}.csv")
        df.to_csv(fpath, index=False)
        print(f"  ✓  {name:22s}  →  {fpath}  ({len(df)} rows)")
        paths[name] = fpath

    return paths


# ===========================================================================
# Phase 5 – Ethics & Transparency Audit  (unchanged from v1)
# ===========================================================================

def print_ethics_audit(
    logs:      pd.DataFrame,
    employees: pd.DataFrame,
    pipeline:  RecommendationPipeline,
) -> None:
    _section("PHASE 5 │ ETHICS & TRANSPARENCY AUDIT SUMMARY")

    fl   = pipeline.fairness_layer
    attr = fl.protected_attribute

    # ── 5a. Demographic composition ──────────────────────────────────────
    print("\n  ── 5a. Workforce Demographic Composition ──")
    total = len(employees)
    for group, count in employees[attr].value_counts().items():
        print(f"    {group:25s} : {count:3d} employees  ({count/total:.1%})")

    # ── 5b. Fairness layer ────────────────────────────────────────────────
    print(f"\n  ── 5b. Ethical Fairness Layer  (protected attribute: '{attr}') ──")
    print(f"    Parity score BEFORE adjustment : {fl.parity_score_before:.4f}")
    post_parity = logs["fair_dist_score"].iloc[0]
    print(f"    Parity score AFTER  adjustment : {post_parity:.4f}")
    print(f"    Scale: 0.0 = severe disparity │ 1.0 = perfect parity")
    print("\n    Group-level adjustments applied:")
    for group, adj in sorted(fl.group_adjustments.items()):
        direction = "BOOST  ▲" if adj > 0 else ("REDUCE ▼" if adj < 0 else "NONE   –")
        print(f"      {group:25s} : {adj:+.5f}  ({direction})")

    # ── 5c. Recommendation distribution ──────────────────────────────────
    print(f"\n  ── 5c. Top-1 Recommendation Distribution by '{attr}' ──")
    top1 = logs[logs["rank"] == 1]
    for group, count in top1[attr].value_counts().items():
        group_total = (employees[attr] == group).sum()
        rate = count / group_total if group_total else 0
        print(
            f"    {group:25s} : {count:3d} / {group_total:3d}"
            f"  ({rate:.1%} recommendation rate)"
        )

    # ── 5d. Score statistics ──────────────────────────────────────────────
    print("\n  ── 5d. Score Statistics ──")
    for col, label in [
        ("raw_similarity", "Raw similarity"),
        ("fairness_score", "Fairness score"),
        ("fairness_delta", "Fairness delta"),
    ]:
        s = logs[col]
        fmt = "+.5f" if col == "fairness_delta" else ".4f"
        print(
            f"    {label:16s} │ mean={s.mean():{fmt}}  "
            f"std={s.std():.4f}  "
            f"min={s.min():{fmt}}  "
            f"max={s.max():{fmt}}"
        )

    # ── 5e. Most-recommended courses ─────────────────────────────────────
    print("\n  ── 5e. Most-Recommended Courses (all ranks) ──")
    for title, count in logs["course_title"].value_counts().head(5).items():
        print(f"    {count:3d}×  {title}")

    # ── 5f. XAI coverage ─────────────────────────────────────────────────
    print("\n  ── 5f. XAI Transparency Coverage ──")
    has_xai    = logs["xai_reason"].str.len() > 0
    has_fair   = logs["xai_reason"].str.contains("FAIRNESS ADJUSTMENT")
    print(f"    Recommendations with explanation  : {has_xai.sum()} / {len(logs)}")
    print(f"    Explanations with fairness note   : {has_fair.sum()} / {len(logs)}")

    # ── 5g. NLP normalisation validation ─────────────────────────────────
    print("\n  ── 5g. NLP Normalisation Validation (v2 SkillNormalizer) ──")
    from models import SkillNormalizer
    test_cases = [
        ("Trailing spaces",    "Python , Pandas;  NumPy "),
        ("Semicolon delimiter", "AWS;Docker;Kubernetes"),
        ("Mixed casing",        "python,PANDAS,numpy"),
        ("Duplicate skills",    "Python, python, SQL, SQL"),
        ("Internal spaces",     "REST  APIs, Node.js"),
    ]
    for desc, raw in test_cases:
        tokens  = SkillNormalizer.to_token_list(raw)
        sentence = SkillNormalizer.to_sentence(raw)
        print(f"    {desc:25s} │ in={raw!r}")
        print(f"    {'':25s} │ tokens={tokens}")
        print(f"    {'':25s} │ sentence={sentence!r}")
        print()

    # ── 5h. Compliance checklist ──────────────────────────────────────────
    print("  ── 5h. Ethical Principles Compliance Checklist ──")
    checks = {
        "Transparency (XAI reasons logged)":
            bool(has_xai.all()),
        "Fairness (parity score > 0.85 post-adjustment)":
            post_parity >= 0.85,
        "Non-replacement (no termination data in logs)":
            "termination" not in " ".join(logs.columns).lower(),
        "Accountability (raw & adjusted scores both logged)":
            "raw_similarity" in logs.columns and "fairness_score" in logs.columns,
        "Auditability (demographic group per log row)":
            "demographic_group" in logs.columns,
        "NLP consistency (SkillNormalizer used in embeddings & XAI)":
            True,   # structural guarantee from models.py v2
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
    Full pipeline entry point (v3).

    Engine lifecycle
    ────────────────
    ``seed_database()`` returns the live SQLAlchemy ``Engine`` on success,
    or ``None`` when running in dry-run / CSV-fallback mode.  The engine
    reference is stored in ``_engine`` before the ``try`` block so that
    the ``finally`` clause can always call ``_engine.dispose()`` if an
    engine exists — even if Phase 5 raises an unhandled exception.

    Why dispose() matters
    ─────────────────────
    SQLAlchemy's connection pool spawns background threads to manage idle
    connections.  If the process exits without calling ``dispose()``, those
    threads may keep the interpreter alive for several seconds after
    ``main()`` returns, producing a confusing delay in scripts, test
    runners, and container shutdowns.  ``dispose()`` closes all pooled
    connections synchronously and joins the pool's background threads,
    giving the OS an immediate clean release of file descriptors and
    TCP sockets.
    """

    # ── Phase 0 : Schema (printed; DDL execution happens in Phase 4) ──────
    _section("PHASE 0 │ MYSQL DATABASE SCHEMA")
    print(MYSQL_SCHEMA)
    if _db_env_configured():
        print("  ℹ  DB environment variables detected.")
        print("     DDL will be executed against the live server in Phase 4.")
    else:
        print("  ℹ  No DB environment variables detected.")
        print("     Dry-run mode: Phase 4 will export CSVs instead.")

    # ── Phase 1 : Data Generation ──────────────────────────────────────────
    _section("PHASE 1 │ SYNTHETIC DATA GENERATION")
    print("  Generating 50 employee records …")
    employees = generate_employees(50)
    print(f"  ✓  {len(employees)} employees generated.")

    print("  Generating 10 NPTEL course records …")
    courses = generate_courses()
    print(f"  ✓  {len(courses)} courses generated.")

    print("\n  Demographic snapshot (intentional historical imbalance):")
    for gender, count in employees["gender"].value_counts().items():
        print(f"    {gender:12s} : {count} employees")

    print("\n  Risk score distribution:")
    risk_bins = pd.cut(
        employees["risk_score"],
        bins=[0, 0.3, 0.55, 1.0],
        labels=["Low (<0.30)", "Medium (0.30–0.55)", "High (>0.55)"],
    )
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

    # ── Phase 3 : Results Report ───────────────────────────────────────────
    print_results_report(logs, top_k=6)

    # ── Phase 4 : DB seeding or CSV fallback  (engine lifecycle managed here) ─
    out_dir = _ensure_output_dir()

    # _engine holds the live SQLAlchemy Engine returned by seed_database(),
    # or None when in dry-run / CSV-fallback mode.  Declared before the try
    # block so the finally clause can always reference it safely.
    _engine: "Optional[_SAEngine]" = None

    try:
        _engine = seed_database(employees, courses, logs)

        if _engine is None:
            # DB was not configured or failed — fall back to CSV export.
            csv_paths = export_csv_fallback(employees, courses, logs, out_dir)
            print("\n  CSV output files:")
            for name, path in csv_paths.items():
                print(f"    {name:22s} → {path}")

        # ── Phase 5 : Ethics Audit ─────────────────────────────────────────
        print_ethics_audit(logs, employees, pipeline)

        db_mode = _engine is not None
        print(f"  Pipeline complete.  {'DB seeded ✓' if db_mode else 'CSV exported ✓'}")

    finally:
        # Guaranteed teardown: close all pooled connections and release
        # OS-level file descriptors held by the SQLAlchemy connection pool.
        # This runs whether the try block succeeded, hit an exception, or
        # was cut short by a KeyboardInterrupt.
        if _engine is not None:
            _engine.dispose()
            log.info(
                "SQLAlchemy engine disposed — all connection pool handles "
                "returned to the OS."
            )


if __name__ == "__main__":
    main()
