"""
models.py
=========
SkillGaps.AI – Ethical Workforce Transition & ML Platform
----------------------------------------------------------
Contains three co-operating components:

  ┌──────────────────────────────────────────────────────────┐
  │  1. EmbeddingEngine                                      │
  │     Converts employee skill profiles and NPTEL course    │
  │     descriptions to dense sentence embeddings using      │
  │     Hugging Face sentence-transformers, then computes    │
  │     cosine similarity to produce raw match scores.       │
  ├──────────────────────────────────────────────────────────┤
  │  2. EthicalFairnessLayer                                 │
  │     Detects demographic imbalances in raw recommendation │
  │     rates (demographic parity check) and applies a       │
  │     Fairlearn-inspired score adjustment that boosts      │
  │     under-represented groups and penalises over-         │
  │     represented ones – narrowing the parity gap without  │
  │     overriding merit entirely.                           │
  ├──────────────────────────────────────────────────────────┤
  │  3. XAIExplainer                                         │
  │     Generates a plain-English, auditable reason for each │
  │     recommendation by surfacing matched skill keywords,  │
  │     the employee's gap profile, and the fairness delta.  │
  └──────────────────────────────────────────────────────────┘

Ethical design principles
--------------------------
* Transparency   – every recommendation ships with a human-readable
                   explanation (XAI) stored in the audit log.
* Fairness       – parity scores and deltas are computed and persisted
                   so any auditor can quantify historical bias correction.
* Non-replacement – the platform's output is upskilling recommendations,
                    NOT termination risk rankings. Risk score is used only
                    to prioritise who receives recommendations first.
* Accountability – the EthicalFairnessLayer records both the raw score
                   and the adjusted score so adjustments are never hidden.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# 1.  EMBEDDING ENGINE
# ===========================================================================

class EmbeddingEngine:
    """
    Wraps a Hugging Face sentence-transformer model to produce embeddings
    for employee skill profiles and NPTEL course descriptions.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model identifier.  The default
        'all-MiniLM-L6-v2' is compact (~80 MB) yet accurate for
        semantic similarity tasks and suitable for CPU inference.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        print(f"[EmbeddingEngine]  Loading model '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"[EmbeddingEngine]  Model ready.")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_skills(self, skill_strings: List[str]) -> np.ndarray:
        """
        Encode a list of comma-separated skill strings into dense vectors.

        Each string is treated as a *sentence* so the transformer can
        capture semantic relationships between technology terms
        (e.g. 'PyTorch' is semantically close to 'TensorFlow').

        Returns
        -------
        np.ndarray  shape (n_employees, embedding_dim)
        """
        # Normalise: replace commas with spaces so the tokeniser sees
        # individual tokens rather than treating the whole list as one word.
        normalised = [s.replace(",", " ").strip() for s in skill_strings]
        embeddings = self.model.encode(
            normalised,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-norm → cosine = dot product
        )
        return embeddings

    def encode_courses(self, descriptions: List[str]) -> np.ndarray:
        """
        Encode course descriptions into the same embedding space.

        Using the full natural-language description (rather than just
        skill tags) lets the model capture pedagogical context, not
        only vocabulary overlap.

        Returns
        -------
        np.ndarray  shape (n_courses, embedding_dim)
        """
        embeddings = self.model.encode(
            descriptions,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def compute_similarity_matrix(
        self,
        employee_embeddings: np.ndarray,
        course_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full cosine similarity matrix.

        Because embeddings are unit-normalised, cosine similarity equals
        the dot product and lies in [-1, 1]; in practice scores are
        positive (skill descriptions share positive semantic content).

        Returns
        -------
        np.ndarray  shape (n_employees, n_courses)
            sim[i, j] = cosine similarity between employee i and course j.
        """
        sim_matrix = cosine_similarity(employee_embeddings, course_embeddings)
        return sim_matrix.astype(np.float32)


# ===========================================================================
# 2.  ETHICAL FAIRNESS LAYER
# ===========================================================================

class EthicalFairnessLayer:
    """
    Implements a Fairlearn-inspired demographic-parity correction.

    Methodology
    -----------
    1. *Compute recommendation rate per group* – for each protected
       attribute value (e.g. gender = 'Female'), count how many employees
       in that group would receive a recommendation above the base threshold
       under raw similarity scores.

    2. *Compute parity gap* – the difference between a group's rate and the
       mean recommendation rate across all groups.

    3. *Apply multiplicative adjustment* – add a small positive delta to
       under-represented groups and a negative delta to over-represented
       groups.  The adjustment magnitude is bounded so the final score
       always reflects a mix of merit (≥ 70 %) and parity correction
       (≤ 30 %).

    4. *Record fair_dist_score* – a scalar in [0, 1] measuring how close
       the adjusted distribution is to perfect demographic parity (1.0).

    Parameters
    ----------
    protected_attribute : str
        Column name in the employee DataFrame that defines demographic
        groups, e.g. 'gender'.
    base_threshold : float
        Raw similarity score above which an employee is counted as
        "receiving a recommendation" when computing group rates.
    max_adjustment : float
        Upper bound on score adjustment applied to any individual.
        Capped at 0.15 so merit dominates.
    """

    def __init__(
        self,
        protected_attribute: str = "gender",
        base_threshold: float = 0.30,
        max_adjustment: float = 0.15,
    ) -> None:
        self.protected_attribute = protected_attribute
        self.base_threshold      = base_threshold
        self.max_adjustment      = max_adjustment
        self._group_adjustments: Dict[str, float] = {}
        self._group_rates:       Dict[str, float] = {}
        self._parity_score:      float            = 0.0

    # ------------------------------------------------------------------
    # Core fairness computation
    # ------------------------------------------------------------------

    def fit(
        self,
        employees: pd.DataFrame,
        raw_sim_matrix: np.ndarray,
    ) -> "EthicalFairnessLayer":
        """
        Analyse raw recommendation rates per demographic group and
        compute the adjustment coefficients.

        Parameters
        ----------
        employees      : DataFrame with at minimum `protected_attribute` column.
        raw_sim_matrix : shape (n_employees, n_courses); scores before fairness.

        Returns
        -------
        self  (for method chaining)
        """
        groups = employees[self.protected_attribute].values
        # For each employee, take the MAX similarity score across all courses
        # as a proxy for "would this person get recommended anything?"
        max_scores = raw_sim_matrix.max(axis=1)

        group_labels  = np.unique(groups)
        group_rates: Dict[str, float] = {}

        for g in group_labels:
            mask    = groups == g
            g_scores = max_scores[mask]
            # Recommendation rate = fraction above threshold
            rate     = float((g_scores >= self.base_threshold).mean())
            group_rates[g] = rate

        mean_rate = float(np.mean(list(group_rates.values())))

        # Compute adjustment: under-represented groups get a positive boost,
        # over-represented get a negative correction (never below 0).
        adjustments: Dict[str, float] = {}
        for g, rate in group_rates.items():
            gap  = mean_rate - rate          # positive → under-represented
            adj  = np.clip(gap, -self.max_adjustment, self.max_adjustment)
            adjustments[g] = float(adj)

        self._group_rates       = group_rates
        self._group_adjustments = adjustments
        self._mean_rate         = mean_rate

        # Parity score: 1 – (range of group rates / mean_rate)
        # A perfectly paritious system scores 1.0.
        if mean_rate > 0:
            rate_range   = max(group_rates.values()) - min(group_rates.values())
            parity_score = max(0.0, 1.0 - (rate_range / mean_rate))
        else:
            parity_score = 1.0

        self._parity_score = round(parity_score, 4)

        # ---- logging ----
        print("\n[EthicalFairnessLayer]  Demographic parity analysis:")
        print(f"  Protected attribute : '{self.protected_attribute}'")
        print(f"  Mean recommendation rate (all groups) : {mean_rate:.1%}")
        for g in sorted(group_rates.keys()):
            adj_str = f"+{adjustments[g]:.4f}" if adjustments[g] >= 0 \
                      else f"{adjustments[g]:.4f}"
            print(
                f"  Group '{g:20s}' | rate={group_rates[g]:.1%}"
                f"  | adjustment={adj_str}"
            )
        print(f"  Demographic parity score (pre-adjust) : {self._parity_score:.4f}")

        return self

    def adjust_scores(
        self,
        employees: pd.DataFrame,
        raw_sim_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the pre-computed group adjustments to produce a fairness-
        corrected similarity matrix.

        Parameters
        ----------
        employees      : DataFrame (must contain `protected_attribute`).
        raw_sim_matrix : shape (n_employees, n_courses).

        Returns
        -------
        adjusted_matrix : np.ndarray  shape (n_employees, n_courses)
        delta_matrix    : np.ndarray  shape (n_employees, n_courses)
            delta[i,j] = adjusted[i,j] – raw[i,j]
        """
        if not self._group_adjustments:
            raise RuntimeError("Call .fit() before .adjust_scores().")

        groups          = employees[self.protected_attribute].values
        adjusted_matrix = raw_sim_matrix.copy()

        for idx, g in enumerate(groups):
            adj = self._group_adjustments.get(g, 0.0)
            adjusted_matrix[idx, :] = np.clip(
                raw_sim_matrix[idx, :] + adj, 0.0, 1.0
            )

        delta_matrix = adjusted_matrix - raw_sim_matrix
        return adjusted_matrix.astype(np.float32), delta_matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Post-adjustment parity score
    # ------------------------------------------------------------------

    def compute_adjusted_parity(
        self,
        employees: pd.DataFrame,
        adjusted_matrix: np.ndarray,
    ) -> float:
        """
        Recompute the demographic parity score on the *adjusted* matrix.
        """
        groups     = employees[self.protected_attribute].values
        max_scores = adjusted_matrix.max(axis=1)
        group_rates: Dict[str, float] = {}

        for g in np.unique(groups):
            mask = groups == g
            rate = float((max_scores[mask] >= self.base_threshold).mean())
            group_rates[g] = rate

        mean_rate = float(np.mean(list(group_rates.values()))) or 1e-9
        rate_range = max(group_rates.values()) - min(group_rates.values())
        score      = max(0.0, 1.0 - (rate_range / mean_rate))
        return round(score, 4)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def group_adjustments(self) -> Dict[str, float]:
        return dict(self._group_adjustments)

    @property
    def parity_score_before(self) -> float:
        return self._parity_score


# ===========================================================================
# 3.  XAI EXPLAINER
# ===========================================================================

class XAIExplainer:
    """
    Generates transparent, human-readable explanations for each
    course recommendation using keyword-overlap XAI.

    Approach
    --------
    Rather than treating the similarity score as a black box, we surface:
      (a) which of the employee's skills directly overlap with the course
          skill tags (direct evidence);
      (b) which skills the course would add that the employee lacks
          (learning opportunity evidence);
      (c) the cosine similarity score (quantitative evidence);
      (d) the fairness delta (ethical transparency).

    This mirrors the SHAP philosophy of attributing a prediction to
    interpretable input features – here, individual skill tokens.
    """

    @staticmethod
    def _tokenise(skill_string: str) -> set:
        """Lower-case, strip, and split a comma-separated skill string."""
        return {s.strip().lower() for s in skill_string.split(",") if s.strip()}

    def explain(
        self,
        employee_row:  pd.Series,
        course_row:    pd.Series,
        raw_score:     float,
        fair_score:    float,
        fair_delta:    float,
        rank:          int = 1,
    ) -> str:
        """
        Build a plain-English explanation for a single (employee, course) pair.

        Parameters
        ----------
        employee_row : Series from the employees DataFrame.
        course_row   : Series from the nptel_courses DataFrame.
        raw_score    : Cosine similarity before fairness adjustment.
        fair_score   : Cosine similarity after fairness adjustment.
        fair_delta   : fair_score – raw_score.
        rank         : 1-based rank of this course in employee's top-N list.

        Returns
        -------
        str : Multi-sentence human-readable explanation.
        """
        emp_skills    = self._tokenise(employee_row["skills_current"])
        course_skills = self._tokenise(course_row["skill_tags"])

        matched_skills = sorted(emp_skills & course_skills)
        gap_skills     = sorted(course_skills - emp_skills)

        # ------------------------------------------------------------------
        # Build the explanation sentence by sentence
        # ------------------------------------------------------------------
        lines: List[str] = []

        # Header
        lines.append(
            f"RECOMMENDATION #{rank} for {employee_row['full_name']} "
            f"({employee_row['role']}, {employee_row['department']})"
        )
        lines.append(f"  Course  : {course_row['title']} [{course_row['difficulty']}]")
        lines.append(f"  Domain  : {course_row['domain']}")

        # Semantic similarity
        lines.append(
            f"\n  [SIMILARITY]  Cosine similarity score = {raw_score:.4f}  "
            f"(range 0–1; higher = stronger profile match)."
        )

        # Skill match evidence
        if matched_skills:
            lines.append(
                f"  [SKILL MATCH]  Your existing skills directly matched "
                f"{len(matched_skills)} course topic(s): "
                f"{', '.join(matched_skills)}. "
                "These overlaps indicate strong prior knowledge relevant "
                "to this course's content."
            )
        else:
            lines.append(
                "  [SKILL MATCH]  No direct keyword overlap was found; "
                "the recommendation is driven by semantic similarity between "
                "your skill profile and the course's subject matter."
            )

        # Skill gap / upskilling opportunity
        if gap_skills:
            lines.append(
                f"  [UPSKILLING OPPORTUNITY]  Completing this course would add "
                f"{len(gap_skills)} new skill(s) to your profile: "
                f"{', '.join(gap_skills)}."
            )

        # Displacement risk context
        risk = employee_row.get("risk_score", 0.0)
        if risk >= 0.55:
            lines.append(
                f"  [PRIORITY FLAG]  Your displacement risk score ({risk:.2f}) "
                "is elevated. This recommendation is prioritised to support "
                "your workforce transition."
            )
        elif risk >= 0.35:
            lines.append(
                f"  [CONTEXT]  Your displacement risk score is moderate ({risk:.2f}). "
                "This course strengthens your adaptability for emerging roles."
            )
        else:
            lines.append(
                f"  [CONTEXT]  Your displacement risk score is low ({risk:.2f}). "
                "This course enriches your existing strengths."
            )

        # Fairness transparency
        if abs(fair_delta) > 0.001:
            direction = "upward" if fair_delta > 0 else "downward"
            lines.append(
                f"  [FAIRNESS ADJUSTMENT]  The Ethical Fairness Layer applied a "
                f"{direction} score correction of {fair_delta:+.4f} based on "
                f"demographic parity analysis across '{employee_row.get('gender', 'N/A')}' "
                "group recommendation rates. Adjusted score = "
                f"{fair_score:.4f}. This correction ensures equitable access "
                "to upskilling opportunities across all demographic groups."
            )
        else:
            lines.append(
                "  [FAIRNESS ADJUSTMENT]  No parity correction was required "
                "for this recommendation; demographic rates were already balanced."
            )

        return "\n".join(lines)


# ===========================================================================
# 4.  RECOMMENDATION PIPELINE  (orchestrates the three components above)
# ===========================================================================

class RecommendationPipeline:
    """
    End-to-end pipeline that ties EmbeddingEngine, EthicalFairnessLayer,
    and XAIExplainer together and produces a structured output DataFrame
    suitable for insertion into the `upskilling_logs` MySQL table.

    Parameters
    ----------
    top_n : int
        Number of course recommendations to generate per employee.
    model_name : str
        Sentence-transformer model identifier.
    protected_attribute : str
        Demographic column for fairness analysis.
    """

    def __init__(
        self,
        top_n: int = 3,
        model_name: str = "all-MiniLM-L6-v2",
        protected_attribute: str = "gender",
    ) -> None:
        self.top_n               = top_n
        self.protected_attribute = protected_attribute
        self.embedding_engine    = EmbeddingEngine(model_name)
        self.fairness_layer      = EthicalFairnessLayer(
            protected_attribute=protected_attribute
        )
        self.xai_explainer       = XAIExplainer()

    # ------------------------------------------------------------------

    def run(
        self,
        employees: pd.DataFrame,
        courses:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the full pipeline and return an upskilling_logs DataFrame.

        Steps
        -----
        1. Embed employee skill profiles.
        2. Embed course descriptions.
        3. Compute raw cosine similarity matrix.
        4. Fit the fairness layer on raw scores.
        5. Compute fairness-adjusted scores + deltas.
        6. For each employee, select top-N courses by adjusted score.
        7. Generate XAI explanation for each recommendation.
        8. Assemble and return the log DataFrame.

        Returns
        -------
        pd.DataFrame  matching the schema of `upskilling_logs`.
        """
        print("\n" + "=" * 60)
        print("RECOMMENDATION PIPELINE  –  START")
        print("=" * 60)

        # ── Step 1 & 2 : Embeddings ──────────────────────────────────
        print("\n[Step 1]  Encoding employee skill profiles …")
        emp_embeddings = self.embedding_engine.encode_skills(
            employees["skills_current"].tolist()
        )
        print(f"          → {emp_embeddings.shape} embeddings generated.")

        print("[Step 2]  Encoding NPTEL course descriptions …")
        course_embeddings = self.embedding_engine.encode_courses(
            courses["description"].tolist()
        )
        print(f"          → {course_embeddings.shape} embeddings generated.")

        # ── Step 3 : Raw similarity ───────────────────────────────────
        print("[Step 3]  Computing cosine similarity matrix …")
        raw_sim = self.embedding_engine.compute_similarity_matrix(
            emp_embeddings, course_embeddings
        )
        print(f"          → Matrix shape: {raw_sim.shape}  "
              f"| Range: [{raw_sim.min():.4f}, {raw_sim.max():.4f}]")

        # ── Step 4 : Fairness fit ─────────────────────────────────────
        print("\n[Step 4]  Fitting Ethical Fairness Layer …")
        self.fairness_layer.fit(employees, raw_sim)

        # ── Step 5 : Fairness adjustment ──────────────────────────────
        print("\n[Step 5]  Applying fairness score adjustments …")
        adj_sim, delta_sim = self.fairness_layer.adjust_scores(employees, raw_sim)
        post_parity = self.fairness_layer.compute_adjusted_parity(employees, adj_sim)
        print(f"          Demographic parity score AFTER adjustment : {post_parity:.4f}")
        print(
            f"          Improvement  : "
            f"{self.fairness_layer.parity_score_before:.4f}  →  {post_parity:.4f}"
        )

        # ── Steps 6, 7, 8 : Recommendations + XAI + Log ──────────────
        print(f"\n[Step 6-8]  Generating top-{self.top_n} recommendations per employee …")
        log_records: List[dict] = []

        # Sort employees by risk_score descending (high-risk first)
        emp_sorted = employees.sort_values("risk_score", ascending=False)

        for _, emp in emp_sorted.iterrows():
            i = int(emp["employee_id"]) - 1   # 0-indexed row in matrices

            # Get top-N course indices by adjusted score
            course_adj_scores = adj_sim[i]
            top_indices       = np.argsort(course_adj_scores)[::-1][: self.top_n]

            for rank, c_idx in enumerate(top_indices, start=1):
                course     = courses.iloc[c_idx]
                raw_score  = float(raw_sim[i, c_idx])
                fair_score = float(adj_sim[i, c_idx])
                fair_delta = float(delta_sim[i, c_idx])

                explanation = self.xai_explainer.explain(
                    employee_row=emp,
                    course_row=course,
                    raw_score=raw_score,
                    fair_score=fair_score,
                    fair_delta=fair_delta,
                    rank=rank,
                )

                log_records.append(
                    {
                        "employee_id":       int(emp["employee_id"]),
                        "employee_name":     emp["full_name"],
                        "gender":            emp["gender"],
                        "risk_score":        emp["risk_score"],
                        "course_id":         int(course["course_id"]),
                        "course_title":      course["title"],
                        "raw_similarity":    round(raw_score,  5),
                        "fairness_score":    round(fair_score, 5),
                        "fairness_delta":    round(fair_delta, 5),
                        "demographic_group": emp[self.protected_attribute],
                        "xai_reason":        explanation,
                        "fair_dist_score":   post_parity,
                        "rank":              rank,
                    }
                )

        logs_df = pd.DataFrame(log_records)
        print(f"          → {len(logs_df)} log records created.")
        print("\n[PIPELINE]  ✓  Complete.\n")
        return logs_df
