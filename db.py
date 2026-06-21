"""
db.py
=====
SkillGaps.AI – Ethical Workforce Transition & ML Platform
----------------------------------------------------------
Database layer: connection management, DDL execution, and data seeding.

Responsibilities
────────────────
1. BUILD a SQLAlchemy engine from environment variables (never hard-coded
   credentials).
2. EXECUTE the multi-statement DDL (CREATE DATABASE … / USE … / CREATE TABLE …)
   against the live MySQL server, statement-by-statement, idempotently.
3. SEED the ``employees`` and ``nptel_courses`` tables with pandas DataFrames
   via ``DataFrame.to_sql()``.
4. INSERT ``upskilling_logs`` rows respecting the FK constraints that link
   back to ``employee_id`` and ``course_id``.
5. PROVIDE retry logic and clean error handling for connection failures.

Environment variables required
───────────────────────────────
Set these before running main.py.  Never commit real values to source control.

    # Linux / macOS
    export SKILLGAPS_DB_HOST=127.0.0.1
    export SKILLGAPS_DB_PORT=3306
    export SKILLGAPS_DB_USER=skillgaps_user
    export SKILLGAPS_DB_PASS=your_secure_password
    export SKILLGAPS_DB_NAME=skillgaps_ai      # must match CREATE DATABASE name

    # Windows (PowerShell)
    $env:SKILLGAPS_DB_HOST = "127.0.0.1"
    $env:SKILLGAPS_DB_PORT = "3306"
    $env:SKILLGAPS_DB_USER = "skillgaps_user"
    $env:SKILLGAPS_DB_PASS = "your_secure_password"
    $env:SKILLGAPS_DB_NAME = "skillgaps_ai"

    # Or create a .env file and load with python-dotenv (see note in code).

MySQL user creation (run once as root)
───────────────────────────────────────
    CREATE USER 'skillgaps_user'@'localhost' IDENTIFIED BY 'your_secure_password';
    GRANT ALL PRIVILEGES ON skillgaps_ai.* TO 'skillgaps_user'@'localhost';
    FLUSH PRIVILEGES;

Security notes
──────────────
* Credentials are read ONLY from environment variables — no defaults in code.
* Connection strings are never printed; only the sanitised host/port/db are logged.
* The engine uses ``pool_pre_ping=True`` to detect stale connections.
* All DDL and seed operations run inside explicit transactions.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator, List, Optional

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Engine, TextClause, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("skillgaps.db")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ===========================================================================
# 1.  ENGINE FACTORY
# ===========================================================================

def build_engine(
    retries:      int   = 3,
    retry_delay:  float = 2.0,
) -> Engine:
    """
    Build and validate a SQLAlchemy engine from environment variables.

    The function reads six environment variables and constructs a
    ``mysql+pymysql://`` connection URL.  It then attempts a lightweight
    ``SELECT 1`` probe to confirm the server is reachable, retrying up to
    ``retries`` times with ``retry_delay`` seconds between attempts.

    Parameters
    ----------
    retries : int
        Maximum number of connection attempts before raising.
    retry_delay : float
        Seconds to wait between retry attempts.

    Returns
    -------
    sqlalchemy.Engine
        A connected, pre-pinged SQLAlchemy engine.

    Raises
    ------
    EnvironmentError
        If any required environment variable is missing.
    sqlalchemy.exc.OperationalError
        If the database server is unreachable after all retries.
    """
    # ── Read credentials from environment (NEVER hard-code) ───────────────
    required_vars = {
        "host": "SKILLGAPS_DB_HOST",
        "port": "SKILLGAPS_DB_PORT",
        "user": "SKILLGAPS_DB_USER",
        "password": "SKILLGAPS_DB_PASS",
        "database": "SKILLGAPS_DB_NAME",
    }

    config: dict[str, str] = {}
    missing: List[str] = []
    for key, env_var in required_vars.items():
        value = os.environ.get(env_var)
        if not value:
            missing.append(env_var)
        else:
            config[key] = value

    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            "See the docstring in db.py for setup instructions."
        )

    # ── Build the URL (password is URL-encoded to handle special characters) ─
    url = sa.engine.URL.create(
        drivername="mysql+pymysql",
        username=config["user"],
        password=config["password"],   # SQLAlchemy handles URL-encoding
        host=config["host"],
        port=int(config["port"]),
        database=config["database"],
        query={
            "charset": "utf8mb4",
            # Treat DECIMAL columns as Python float (not string)
            "use_unicode": "1",
        },
    )

    # Log a sanitised summary (NO password, NO full URL)
    log.info(
        "Building engine  →  %s@%s:%s/%s",
        config["user"], config["host"], config["port"], config["database"],
    )

    # ── Engine configuration ──────────────────────────────────────────────
    engine: Engine = sa.create_engine(
        url,
        # Detects and recycles broken connections before handing them to
        # client code — essential for long-running pipelines.
        pool_pre_ping=True,
        # Keep a modest pool; the pipeline is single-threaded.
        pool_size=3,
        max_overflow=2,
        # SQLAlchemy 2.x: use 'future=True' API everywhere.
        future=True,
    )

    # ── Connection probe with retry ───────────────────────────────────────
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database connection verified (attempt %d/%d).", attempt, retries)
            return engine
        except OperationalError as exc:
            last_exc = exc
            log.warning(
                "Connection attempt %d/%d failed: %s  Retrying in %.1fs …",
                attempt, retries, exc.orig, retry_delay,
            )
            if attempt < retries:
                time.sleep(retry_delay)

    raise OperationalError(
        f"Could not connect to MySQL after {retries} attempt(s).",
        params=None,
        orig=last_exc,
    )


# ===========================================================================
# 2.  CONTEXT MANAGER  –  transactional connection
# ===========================================================================

@contextmanager
def get_connection(engine: Engine) -> Generator[sa.Connection, None, None]:
    """
    Context manager that yields a SQLAlchemy ``Connection`` inside an
    explicit transaction and rolls back cleanly on any exception.

    Usage
    -----
    ::
        with get_connection(engine) as conn:
            conn.execute(text("INSERT INTO ..."))
        # auto-commit on clean exit, auto-rollback on exception

    Parameters
    ----------
    engine : sqlalchemy.Engine

    Yields
    ------
    sqlalchemy.Connection
    """
    with engine.begin() as conn:   # begin() auto-commits on exit, auto-rolls-back on exception
        try:
            yield conn
        except SQLAlchemyError as exc:
            log.error("Transaction rolled back due to: %s", exc)
            raise


# ===========================================================================
# 3.  DDL EXECUTION  –  CREATE DATABASE + CREATE TABLE
# ===========================================================================

def execute_schema(engine: Engine, schema_sql: str) -> None:
    """
    Execute the multi-statement DDL string against the live MySQL server.

    The DDL contains:
    * ``CREATE DATABASE IF NOT EXISTS skillgaps_ai …``
    * ``USE skillgaps_ai``
    * Three ``CREATE TABLE IF NOT EXISTS …`` blocks

    SQLAlchemy does NOT support multi-statement execution natively, so we
    split the DDL on ``;`` and execute each statement individually.
    Empty tokens (from trailing semicolons / comment lines) are skipped.

    Parameters
    ----------
    engine     : sqlalchemy.Engine – connected engine.
    schema_sql : str               – the MYSQL_SCHEMA constant from data_generation.py.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError  on any DDL failure.
    """
    log.info("Executing schema DDL …")

    # Split on semicolons; filter out blank/comment-only segments.
    statements: List[str] = [
        stmt.strip()
        for stmt in schema_sql.split(";")
        if stmt.strip() and not stmt.strip().startswith("--")
    ]

    # DDL statements (CREATE DATABASE, CREATE TABLE) must run outside a
    # transaction in MySQL because they cause an implicit commit.
    # We use autocommit mode via raw connection.
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        for stmt in statements:
            # Skip pure-comment blocks
            non_comment_lines = [
                ln for ln in stmt.splitlines()
                if ln.strip() and not ln.strip().startswith("--")
            ]
            if not non_comment_lines:
                continue
            clean_stmt = "\n".join(non_comment_lines)
            log.debug("DDL: %s …", clean_stmt[:80].replace("\n", " "))
            cursor.execute(clean_stmt)
        cursor.close()
        raw_conn.commit()
        log.info("Schema DDL executed successfully  (%d statements).", len(statements))
    except Exception as exc:
        raw_conn.rollback()
        log.error("DDL execution failed: %s", exc)
        raise
    finally:
        raw_conn.close()


# ===========================================================================
# 4.  DATA SEEDING  –  employees & nptel_courses
# ===========================================================================

def seed_reference_tables(
    engine:    Engine,
    employees: pd.DataFrame,
    courses:   pd.DataFrame,
) -> None:
    """
    Insert ``employees`` and ``nptel_courses`` DataFrames into MySQL.

    Strategy
    ────────
    * Uses ``pandas.DataFrame.to_sql()`` with ``if_exists='append'`` so
      the function is safe to call after the tables already exist.
    * Before inserting, we check whether rows are already present to keep
      the operation idempotent (running the pipeline twice won't duplicate
      reference data).
    * ``employee_id`` and ``course_id`` are explicit PK columns that match
      the MySQL AUTO_INCREMENT definition — we pass them so ``upskilling_logs``
      FK references resolve correctly.

    Parameters
    ----------
    engine    : sqlalchemy.Engine
    employees : pd.DataFrame  (from data_generation.generate_employees)
    courses   : pd.DataFrame  (from data_generation.generate_courses)
    """
    with get_connection(engine) as conn:
        # ── employees ────────────────────────────────────────────────────
        existing_count: int = conn.execute(
            text("SELECT COUNT(*) FROM employees")
        ).scalar() or 0

        if existing_count == 0:
            # ── Isolate the insert slice with .copy() ─────────────────────
            # Using .copy() here breaks the view/copy ambiguity that pandas
            # raises as a SettingWithCopyWarning when you modify a column on
            # a slice.  Without it, the subsequent .astype(float) below would
            # mutate the caller's DataFrame in-place in some pandas versions,
            # which is a latent data-corruption bug in long-running pipelines.
            emp_insert: pd.DataFrame = employees[[
                "employee_id", "full_name", "department", "role",
                "gender", "age", "years_exp", "skills_current", "risk_score",
            ]].copy()

            # ── Cast risk_score to base Python float ───────────────────────
            # data_generation.py stores risk_score as a Python float rounded
            # to 3 d.p., but after passing through pandas and numpy the column
            # dtype is numpy.float64.  SQLAlchemy's DECIMAL(4,3) type adapter
            # in some versions of PyMySQL does not accept numpy scalar types
            # and raises a DataError / OperationalError during the bind phase.
            # An explicit .astype(float) coerces every cell to a base Python
            # float via numpy's __float__ protocol, which SQLAlchemy's Numeric
            # type adapter reliably serialises to a SQL DECIMAL literal.
            emp_insert["risk_score"] = emp_insert["risk_score"].astype(float)

            emp_insert.to_sql(
                name="employees",
                con=conn,
                if_exists="append",
                index=False,
                # Use VARCHAR for text columns; DECIMAL for risk_score.
                dtype={
                    "skills_current": sa.Text(),
                    "risk_score":     sa.Numeric(4, 3),
                },
            )
            log.info("Inserted %d rows into 'employees'.", len(emp_insert))
        else:
            log.info(
                "Skipping 'employees' seed – table already has %d rows.",
                existing_count,
            )

        # ── nptel_courses ─────────────────────────────────────────────────
        existing_courses: int = conn.execute(
            text("SELECT COUNT(*) FROM nptel_courses")
        ).scalar() or 0

        if existing_courses == 0:
            courses_insert = courses[[
                "course_id", "title", "domain", "duration_weeks",
                "difficulty", "skill_tags", "description",
            ]]
            courses_insert.to_sql(
                name="nptel_courses",
                con=conn,
                if_exists="append",
                index=False,
                dtype={
                    "skill_tags":  sa.Text(),
                    "description": sa.Text(),
                },
            )
            log.info("Inserted %d rows into 'nptel_courses'.", len(courses_insert))
        else:
            log.info(
                "Skipping 'nptel_courses' seed – table already has %d rows.",
                existing_courses,
            )


# ===========================================================================
# 5.  LOG INSERTION  –  upskilling_logs (with FK enforcement)
# ===========================================================================

def insert_upskilling_logs(
    engine:  Engine,
    logs_df: pd.DataFrame,
) -> int:
    """
    Insert the recommendation log records into ``upskilling_logs``.

    FK-safety approach
    ──────────────────
    MySQL enforces ``FOREIGN KEY (employee_id) REFERENCES employees``
    and ``FOREIGN KEY (course_id) REFERENCES nptel_courses``.  Before
    bulk-inserting, we validate that every ``employee_id`` and
    ``course_id`` in ``logs_df`` exists in the parent tables.  Any
    orphaned rows are dropped with a warning rather than crashing the
    insert.

    Chunked insert
    ──────────────
    Logs are inserted in chunks of 500 rows to avoid hitting
    MySQL's ``max_allowed_packet`` limit and to give the connection
    pool time to breathe.

    Parameters
    ----------
    engine  : sqlalchemy.Engine
    logs_df : pd.DataFrame  – output of RecommendationPipeline.run()

    Returns
    -------
    int  – number of rows successfully inserted.
    """
    CHUNK_SIZE = 500

    # Columns that match the upskilling_logs MySQL schema
    LOG_COLUMNS = [
        "employee_id", "course_id",
        "raw_similarity", "fairness_score", "fairness_delta",
        "demographic_group", "xai_reason", "fair_dist_score",
    ]

    with get_connection(engine) as conn:
        # ── FK validation ─────────────────────────────────────────────────
        valid_emp_ids = {
            row[0] for row in
            conn.execute(text("SELECT employee_id FROM employees")).fetchall()
        }
        valid_course_ids = {
            row[0] for row in
            conn.execute(text("SELECT course_id FROM nptel_courses")).fetchall()
        }

        original_len = len(logs_df)
        logs_df = logs_df[
            logs_df["employee_id"].isin(valid_emp_ids) &
            logs_df["course_id"].isin(valid_course_ids)
        ].copy()

        dropped = original_len - len(logs_df)
        if dropped > 0:
            log.warning(
                "Dropped %d log rows with unresolvable FK references.", dropped
            )

        if logs_df.empty:
            log.warning("No valid log rows to insert.")
            return 0

        # ── Prepare insert subset ─────────────────────────────────────────
        logs_insert = logs_df[LOG_COLUMNS].copy()

        # Cast similarity scores to Python float (from numpy float32) so
        # SQLAlchemy's DECIMAL binding doesn't raise type errors.
        for col in ["raw_similarity", "fairness_score", "fairness_delta",
                    "fair_dist_score"]:
            logs_insert[col] = logs_insert[col].astype(float)

        # ── Chunked insert ────────────────────────────────────────────────
        total_inserted = 0
        for start in range(0, len(logs_insert), CHUNK_SIZE):
            chunk = logs_insert.iloc[start: start + CHUNK_SIZE]
            chunk.to_sql(
                name="upskilling_logs",
                con=conn,
                if_exists="append",
                index=False,
                dtype={
                    "raw_similarity":  sa.Numeric(6, 5),
                    "fairness_score":  sa.Numeric(6, 5),
                    "fairness_delta":  sa.Numeric(6, 5),
                    "fair_dist_score": sa.Numeric(5, 4),
                    "xai_reason":      sa.Text(),
                    "demographic_group": sa.String(60),
                },
            )
            total_inserted += len(chunk)
            log.info(
                "Inserted log chunk %d–%d (%d rows).",
                start + 1, start + len(chunk), len(chunk),
            )

        log.info(
            "upskilling_logs: %d / %d rows inserted successfully.",
            total_inserted, original_len,
        )
        return total_inserted


# ===========================================================================
# 6.  HEALTH CHECK  (handy for debugging)
# ===========================================================================

def print_table_counts(engine: Engine) -> None:
    """
    Print row counts for all three core tables — useful as a post-seed
    sanity check.
    """
    tables = ["employees", "nptel_courses", "upskilling_logs"]
    log.info("─── Table row counts ───")
    with get_connection(engine) as conn:
        for table in tables:
            count: int = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            ).scalar() or 0
            log.info("  %-22s : %d rows", table, count)
