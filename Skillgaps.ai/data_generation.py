"""
data_generation.py
==================
SkillGaps.AI – Ethical Workforce Transition & ML Platform
----------------------------------------------------------
Responsible for:
  1. Defining the conceptual MySQL schema (DDL printed as SQL strings).
  2. Synthesising a realistic mock dataset of 50 employees and 10 NPTEL
     courses as pandas DataFrames.

Design notes
------------
* Demographic distribution is intentionally imbalanced (reflecting a
  plausible legacy enterprise workforce) so the Ethical Fairness Layer
  in models.py has something real to detect and correct.
* Skills are stored as comma-separated strings to mirror a VARCHAR/TEXT
  column in MySQL while remaining easy to manipulate in Python.
* All randomness is seeded for full reproducibility.
"""

from __future__ import annotations

import random
import textwrap
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ===========================================================================
# 1.  MYSQL SCHEMA  (DDL – printed/logged, not executed against a live DB)
# ===========================================================================

MYSQL_SCHEMA: str = textwrap.dedent(
    """
    -- =========================================================
    -- SkillGaps.AI  –  Conceptual MySQL Schema
    -- =========================================================

    CREATE DATABASE IF NOT EXISTS skillgaps_ai
        CHARACTER SET utf8mb4
        COLLATE utf8mb4_unicode_ci;

    USE skillgaps_ai;

    -- ---------------------------------------------------------
    -- Table 1 : employees
    --   Stores current workforce snapshot.
    --   'skills_current' is a comma-separated list of skill
    --   tokens (normalised in a production system to a
    --   separate employee_skills junction table).
    -- ---------------------------------------------------------
    CREATE TABLE IF NOT EXISTS employees (
        employee_id     INT             NOT NULL AUTO_INCREMENT,
        full_name       VARCHAR(120)    NOT NULL,
        department      VARCHAR(80)     NOT NULL,
        role            VARCHAR(100)    NOT NULL,
        gender          ENUM('Male','Female','Non-binary','Prefer not to say')
                                        NOT NULL DEFAULT 'Prefer not to say',
        age             TINYINT UNSIGNED NOT NULL,
        years_exp       TINYINT UNSIGNED NOT NULL,
        skills_current  TEXT            NOT NULL COMMENT
            'Comma-separated skill tokens, e.g. "Python,SQL,Excel"',
        risk_score      DECIMAL(4,3)    NOT NULL DEFAULT 0.000
            COMMENT 'Displacement risk score in [0,1] from an upstream model',
        PRIMARY KEY (employee_id)
    ) ENGINE=InnoDB;

    -- ---------------------------------------------------------
    -- Table 2 : nptel_courses
    --   Catalogue of NPTEL upskilling opportunities.
    --   'skill_tags' mirrors the skills vocabulary used in
    --   employees.skills_current so cosine similarity is
    --   semantically meaningful.
    -- ---------------------------------------------------------
    CREATE TABLE IF NOT EXISTS nptel_courses (
        course_id       INT             NOT NULL AUTO_INCREMENT,
        title           VARCHAR(200)    NOT NULL,
        domain          VARCHAR(80)     NOT NULL,
        duration_weeks  TINYINT UNSIGNED NOT NULL,
        difficulty      ENUM('Beginner','Intermediate','Advanced')
                                        NOT NULL,
        skill_tags      TEXT            NOT NULL COMMENT
            'Comma-separated skill tokens this course imparts',
        description     TEXT            NOT NULL,
        PRIMARY KEY (course_id)
    ) ENGINE=InnoDB;

    -- ---------------------------------------------------------
    -- Table 3 : upskilling_logs
    --   Audit trail of every algorithmic recommendation.
    --   Captures the transparency reason, raw similarity score,
    --   fairness-adjusted score, and the fairness delta so
    --   auditors can quantify bias correction post-hoc.
    -- ---------------------------------------------------------
    CREATE TABLE IF NOT EXISTS upskilling_logs (
        log_id              BIGINT          NOT NULL AUTO_INCREMENT,
        employee_id         INT             NOT NULL,
        course_id           INT             NOT NULL,
        recommended_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
        raw_similarity      DECIMAL(6,5)    NOT NULL,
        fairness_score      DECIMAL(6,5)    NOT NULL
            COMMENT 'Score after Ethical Fairness Layer adjustment',
        fairness_delta      DECIMAL(6,5)    NOT NULL
            COMMENT 'fairness_score – raw_similarity; negative = penalised',
        demographic_group   VARCHAR(60)     NOT NULL
            COMMENT 'The protected-attribute value for this employee',
        xai_reason          TEXT            NOT NULL
            COMMENT 'Human-readable explanation generated by XAI module',
        fair_dist_score     DECIMAL(5,4)    NOT NULL
            COMMENT 'Cross-group parity score at time of recommendation [0,1]',
        PRIMARY KEY (log_id),
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
        FOREIGN KEY (course_id)   REFERENCES nptel_courses(course_id),
        INDEX idx_employee   (employee_id),
        INDEX idx_course     (course_id),
        INDEX idx_demo_group (demographic_group)
    ) ENGINE=InnoDB;
    """
)


# ===========================================================================
# 2.  MOCK DATA  –  50 employees  &  10 NPTEL courses
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a.  Employee skill vocabulary (domain-realistic tokens)
# ---------------------------------------------------------------------------

SKILL_POOLS = {
    "data_science": [
        "Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
        "Data Visualisation", "Statistics", "Feature Engineering",
        "Jupyter Notebooks",
    ],
    "data_engineering": [
        "SQL", "Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt",
        "ETL Pipelines", "PostgreSQL", "MySQL", "Data Warehousing",
    ],
    "cloud_devops": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "CI/CD", "Linux", "Bash Scripting", "Monitoring",
    ],
    "software_engineering": [
        "Java", "C++", "JavaScript", "REST APIs", "Microservices",
        "Git", "Agile", "Unit Testing", "System Design", "Node.js",
    ],
    "business_analysis": [
        "Excel", "Power BI", "Tableau", "Requirements Gathering",
        "Stakeholder Management", "Process Mapping", "JIRA",
        "Business Intelligence", "Reporting", "MS Office",
    ],
    "nlp_ai": [
        "NLP", "BERT", "Transformers", "spaCy", "NLTK",
        "Text Classification", "Named Entity Recognition",
        "Sentiment Analysis", "LLMs", "Prompt Engineering",
    ],
}

DEPARTMENTS = [
    "Engineering", "Data & Analytics", "IT Operations",
    "Business Analysis", "Research & Development",
]

ROLES = {
    "Engineering":         ["Software Engineer", "Senior Developer", "Tech Lead"],
    "Data & Analytics":    ["Data Analyst", "Data Scientist", "ML Engineer"],
    "IT Operations":       ["Systems Administrator", "DevOps Engineer", "Cloud Architect"],
    "Business Analysis":   ["Business Analyst", "Product Manager", "Scrum Master"],
    "Research & Development": ["Research Scientist", "NLP Engineer", "AI Researcher"],
}

# Intentionally imbalanced: ~70 % Male, ~28 % Female, ~2 % Non-binary
# This mirrors a plausible legacy tech-sector workforce.
GENDER_POOL = (["Male"] * 35) + (["Female"] * 14) + (["Non-binary"] * 1)

FIRST_NAMES_M = [
    "Arjun", "Rohan", "Vikram", "Suresh", "Rahul", "Aditya", "Kiran",
    "Manoj", "Sanjay", "Deepak", "Amit", "Nikhil", "Pradeep", "Gaurav",
    "Rajesh", "Naveen", "Harish", "Vivek", "Pranav", "Anand",
    "Yusuf", "Samuel", "Daniel", "James", "Carlos",
    "Chen", "Wei", "Hiroshi", "Ali", "Omar",
    "Ethan", "Liam", "Noah", "Oliver", "Lucas",
]
FIRST_NAMES_F = [
    "Priya", "Anjali", "Divya", "Sneha", "Pooja", "Kavya", "Meera",
    "Lakshmi", "Sunita", "Rekha", "Sara", "Emily", "Fatima",
    "Aisha", "Grace",
]
FIRST_NAMES_NB = ["Alex", "Jordan", "Taylor", "Sam", "Robin"]
LAST_NAMES = [
    "Kumar", "Sharma", "Patel", "Singh", "Reddy", "Nair", "Rao",
    "Verma", "Joshi", "Mehta", "Smith", "Johnson", "Brown", "Williams",
    "Jones", "Garcia", "Martinez", "Lee", "Kim", "Chen",
]


def _pick_skills(n_pools: int = 2, skills_per_pool: int = 3) -> str:
    """
    Randomly selects `n_pools` skill domains and draws `skills_per_pool`
    skills from each, returning a deduplicated comma-separated string.
    """
    chosen_pools = random.sample(list(SKILL_POOLS.keys()), k=n_pools)
    skills: List[str] = []
    for pool_name in chosen_pools:
        skills += random.sample(SKILL_POOLS[pool_name], k=skills_per_pool)
    return ", ".join(sorted(set(skills)))


def generate_employees(n: int = 50) -> pd.DataFrame:
    """
    Synthesise `n` realistic employee records.

    Returns
    -------
    pd.DataFrame with columns matching the `employees` MySQL table.
    """
    random.seed(RANDOM_SEED)

    genders = random.sample(GENDER_POOL, k=min(n, len(GENDER_POOL)))
    # If n > pool size, cycle
    while len(genders) < n:
        genders.append(random.choice(GENDER_POOL))

    records = []
    for i in range(n):
        gender = genders[i]
        if gender == "Male":
            first = random.choice(FIRST_NAMES_M)
        elif gender == "Female":
            first = random.choice(FIRST_NAMES_F)
        else:
            first = random.choice(FIRST_NAMES_NB)

        last       = random.choice(LAST_NAMES)
        dept       = random.choice(DEPARTMENTS)
        role       = random.choice(ROLES[dept])
        age        = random.randint(24, 58)
        years_exp  = random.randint(1, min(age - 22, 30))
        # Employees with fewer years experience or non-majority demographics
        # get slightly higher displacement risk (reflecting historical bias).
        risk_base  = round(random.uniform(0.15, 0.65), 3)
        risk_bonus = 0.05 if gender != "Male" else 0.0
        risk_score = min(round(risk_base + risk_bonus, 3), 0.99)

        records.append(
            {
                "employee_id":    i + 1,
                "full_name":      f"{first} {last}",
                "department":     dept,
                "role":           role,
                "gender":         gender,
                "age":            age,
                "years_exp":      years_exp,
                "skills_current": _pick_skills(
                    n_pools=random.randint(2, 3),
                    skills_per_pool=random.randint(2, 4),
                ),
                "risk_score":     risk_score,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2b.  NPTEL Course catalogue  (10 courses)
# ---------------------------------------------------------------------------

NPTEL_COURSES_RAW = [
    {
        "course_id":       1,
        "title":           "Introduction to Machine Learning",
        "domain":          "Data Science",
        "duration_weeks":  8,
        "difficulty":      "Beginner",
        "skill_tags":      "Python, Scikit-learn, Statistics, Feature Engineering, Jupyter Notebooks",
        "description": (
            "A foundational course covering supervised and unsupervised learning algorithms "
            "using Python and Scikit-learn. Topics include regression, classification, "
            "clustering, feature engineering, cross-validation, and model evaluation "
            "with statistical rigour."
        ),
    },
    {
        "course_id":       2,
        "title":           "Deep Learning Specialisation",
        "domain":          "AI & Deep Learning",
        "duration_weeks":  12,
        "difficulty":      "Advanced",
        "skill_tags":      "TensorFlow, PyTorch, NumPy, Python, Data Visualisation",
        "description": (
            "An advanced programme exploring neural network architectures including CNNs, "
            "RNNs, and Transformers using TensorFlow and PyTorch. Covers backpropagation, "
            "optimisation, regularisation, and deployment of deep learning models."
        ),
    },
    {
        "course_id":       3,
        "title":           "Big Data Engineering with Apache Spark",
        "domain":          "Data Engineering",
        "duration_weeks":  10,
        "difficulty":      "Intermediate",
        "skill_tags":      "Apache Spark, Hadoop, Kafka, ETL Pipelines, Python, SQL",
        "description": (
            "Hands-on training in distributed data processing using Apache Spark and "
            "Hadoop. Covers RDDs, DataFrames, Spark SQL, streaming with Kafka, and "
            "building robust ETL pipelines for large-scale datasets."
        ),
    },
    {
        "course_id":       4,
        "title":           "Cloud Computing & DevOps on AWS",
        "domain":          "Cloud & DevOps",
        "duration_weeks":  8,
        "difficulty":      "Intermediate",
        "skill_tags":      "AWS, Docker, Kubernetes, CI/CD, Terraform, Linux, Bash Scripting",
        "description": (
            "A practical course on cloud-native architecture using AWS services. Topics "
            "include containerisation with Docker and Kubernetes, infrastructure-as-code "
            "with Terraform, automated CI/CD pipelines, and cloud security best practices."
        ),
    },
    {
        "course_id":       5,
        "title":           "Natural Language Processing with Transformers",
        "domain":          "NLP & AI",
        "duration_weeks":  10,
        "difficulty":      "Advanced",
        "skill_tags":      "NLP, BERT, Transformers, spaCy, Text Classification, LLMs, Python",
        "description": (
            "An in-depth course on modern NLP using Hugging Face Transformers. Covers "
            "tokenisation, fine-tuning BERT and GPT models, named entity recognition, "
            "text classification, sentiment analysis, and prompt engineering for LLMs."
        ),
    },
    {
        "course_id":       6,
        "title":           "SQL & Database Management Systems",
        "domain":          "Databases",
        "duration_weeks":  6,
        "difficulty":      "Beginner",
        "skill_tags":      "SQL, MySQL, PostgreSQL, Data Warehousing, ETL Pipelines",
        "description": (
            "A comprehensive introduction to relational databases and SQL. Topics span "
            "schema design, normalisation, complex queries, stored procedures, indexing, "
            "query optimisation, and data warehousing concepts with MySQL and PostgreSQL."
        ),
    },
    {
        "course_id":       7,
        "title":           "Data Visualisation & Business Intelligence",
        "domain":          "Business Analytics",
        "duration_weeks":  6,
        "difficulty":      "Beginner",
        "skill_tags":      "Tableau, Power BI, Excel, Business Intelligence, Data Visualisation, Reporting",
        "description": (
            "Practical skills in storytelling with data using Tableau and Power BI. "
            "Covers dashboard design principles, KPI frameworks, Excel analytics, "
            "and communicating insights to non-technical stakeholders through "
            "effective visual reports."
        ),
    },
    {
        "course_id":       8,
        "title":           "Agile Project Management & Scrum",
        "domain":          "Business Analysis",
        "duration_weeks":  5,
        "difficulty":      "Beginner",
        "skill_tags":      (
            "Agile, JIRA, Scrum Master, Stakeholder Management, "
            "Requirements Gathering, Process Mapping"
        ),
        "description": (
            "A structured course on Agile methodologies with focus on Scrum frameworks. "
            "Participants learn sprint planning, backlog grooming, stakeholder communication, "
            "requirements gathering, and using JIRA for project tracking in software teams."
        ),
    },
    {
        "course_id":       9,
        "title":           "MLOps: Deploying Machine Learning at Scale",
        "domain":          "MLOps",
        "duration_weeks":  10,
        "difficulty":      "Advanced",
        "skill_tags":      (
            "Python, Docker, Kubernetes, CI/CD, AWS, Monitoring, "
            "Scikit-learn, REST APIs, Git"
        ),
        "description": (
            "End-to-end machine learning operations: model versioning, experiment tracking, "
            "containerised model serving with Docker and Kubernetes, CI/CD for ML, "
            "A/B testing, drift detection, and monitoring in production cloud environments."
        ),
    },
    {
        "course_id":       10,
        "title":           "Software Architecture & Microservices",
        "domain":          "Software Engineering",
        "duration_weeks":  8,
        "difficulty":      "Intermediate",
        "skill_tags":      (
            "Microservices, REST APIs, System Design, Node.js, "
            "Java, Docker, Git, Unit Testing"
        ),
        "description": (
            "Advanced software design patterns including microservices architecture, "
            "API gateway patterns, service discovery, fault tolerance, and distributed "
            "system design. Uses Java and Node.js with hands-on system design exercises."
        ),
    },
]


def generate_courses() -> pd.DataFrame:
    """
    Return the NPTEL course catalogue as a DataFrame.
    """
    return pd.DataFrame(NPTEL_COURSES_RAW)


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MYSQL SCHEMA")
    print("=" * 60)
    print(MYSQL_SCHEMA)

    employees = generate_employees(50)
    courses   = generate_courses()

    print("\n" + "=" * 60)
    print(f"EMPLOYEES  ({len(employees)} rows)")
    print("=" * 60)
    print(employees[["employee_id", "full_name", "gender", "department",
                      "skills_current", "risk_score"]].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"NPTEL COURSES  ({len(courses)} rows)")
    print("=" * 60)
    print(courses[["course_id", "title", "difficulty", "skill_tags"]].to_string(index=False))

    print("\n[data_generation.py]  ✓  All data generated successfully.")
