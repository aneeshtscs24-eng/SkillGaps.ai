"""
models.py
=========
SkillGaps.AI – Ethical Workforce Transition & ML Platform
----------------------------------------------------------
v2 changelog
────────────
• NEW  SkillNormalizer   — shared text-cleansing utility (Task 2).
  - Handles commas AND semicolons as delimiters.
  - Strips leading/trailing whitespace from every token.
  - Collapses internal multiple-whitespace runs.
  - Title-cases every token for a canonical representation.
  - Exposes both a set-form (for XAI keyword overlap) and a
    sentence-string form (for embedding input).

• CHANGED  EmbeddingEngine.encode_skills()
  - Delegates ALL normalisation to SkillNormalizer instead of the
    inline `s.replace(",", " ").strip()` one-liner.  This guarantees
    that the text fed into the transformer is consistent with the
    tokens used for XAI keyword matching.

• CHANGED  XAIExplainer._tokenise()
  - Replaced the old ad-hoc lower-case split with a call to
    SkillNormalizer.to_token_set(), making keyword overlap
    semantically consistent with the embedding representation.

v3 changelog
────────────
• HARDENED  SkillNormalizer._split_and_clean()
  - Added a defensive type-guard at the top of the method.
  - Guards against None, NaN (pandas NA / float NaN), and any
    non-string type that might arrive from a real HR data feed,
    a DataFrame with missing values, or a malformed CSV import.
  - Short-circuits immediately with [] instead of crashing with
    an AttributeError on regex.split(None).

All other public APIs (EthicalFairnessLayer, RecommendationPipeline)
are unchanged.

Architecture diagram
─────────────────────
  raw skill string  (may be None / NaN from real data sources)
       │
       ▼
  SkillNormalizer._split_and_clean()
  ┌──────────────────────────────────────────────────────────┐
  │  TYPE GUARD: if None / NaN / non-str → return []         │
  │  split on [,;]  →  strip  →  collapse spaces  →  title   │
  └──────────────────────────────────────────────────────────┘
       │                    │
       ▼                    ▼
  sentence string      token set
  (EmbeddingEngine)    (XAIExplainer)
       │
       ▼
  SentenceTransformer  →  cosine similarity matrix
       │
       ▼
  EthicalFairnessLayer  →  parity-adjusted scores
       │
       ▼
  XAIExplainer  →  human-readable explanation
       │
       ▼
  RecommendationPipeline  →  upskilling_logs DataFrame

Ethical design principles (unchanged from v1)
──────────────────────────────────────────────
* Transparency   – every recommendation ships with a human-readable
                   explanation (XAI) stored in the audit log.
* Fairness       – parity scores and deltas are computed and persisted
                   so any auditor can quantify historical bias correction.
* Non-replacement – output is upskilling recommendations, NOT rankings.
* Accountability  – raw and adjusted scores are both logged.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# 0.  SKILL NORMALIZER  (new in v2)
# ===========================================================================

class SkillNormalizer:
    """
    Shared, stateless text-cleansing utility for skill strings.

    Problem solved
    ──────────────
    In v1 the embedding pipeline normalised skill strings with a simple
    ``s.replace(",", " ").strip()`` while XAIExplainer._tokenise() used
    a different ``s.split(",")`` + ``strip().lower()`` approach.  This
    divergence meant a skill stored as ``"  python ; pandas "`` would:
      • be embedded as ``"  python ; pandas "``  (semicolon kept, spaces kept)
      • be tokenised as  ``{"python ; pandas"}`` (treated as ONE token)
    producing zero keyword overlap for what should be two matching skills.

    This class provides ONE canonical normalisation pipeline used by
    both EmbeddingEngine and XAIExplainer, eliminating the mismatch.

    Normalisation steps
    ───────────────────
    1. Split on any comma OR semicolon (``[,;]``).
    2. Strip leading/trailing whitespace from every resulting token.
    3. Collapse any internal multi-space run to a single space
       (handles ``"REST  APIs"`` → ``"REST APIs"``).
    4. Title-case the token (``"python"`` → ``"Python"``,
       ``"REST apis"`` → ``"Rest Apis"`` — good enough for matching;
       exact-case display uses the canonical vocabulary).
    5. Discard empty tokens (artefacts of trailing delimiters).

    Examples
    ────────
    >>> sn = SkillNormalizer()
    >>> sn.to_sentence("Python , pandas;  NumPy ")
    'Python Pandas Numpy'
    >>> sn.to_token_set("  AWS ; docker, Kubernetes  ")
    {'Aws', 'Docker', 'Kubernetes'}

    Note: title-casing ``AWS`` → ``Aws`` is intentional and consistent;
    both the embedding input and the XAI overlap use the same form, so
    the comparison remains correct.  The display layer in XAIExplainer
    uses the original canonical skill vocabulary for human-readable output.
    """

    # Compiled once at class level for efficiency across many calls.
    _DELIMITER_RE:    re.Pattern[str] = re.compile(r"[,;]")
    _MULTI_SPACE_RE:  re.Pattern[str] = re.compile(r"\s{2,}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _split_and_clean(cls, raw: str) -> List[str]:
        """
        Core normalisation: split → strip → collapse spaces → title-case.

        Parameters
        ----------
        raw : str
            Raw skill string, e.g. ``"Python , pandas;  NumPy "``.
            Accepts any type; non-string / null values are handled safely
            by the type-guard and return an empty list instead of raising.

        Returns
        -------
        List[str]
            Ordered list of canonical skill tokens with empty strings removed.
            Returns ``[]`` immediately for None, NaN, or any non-string input.

        Type-guard rationale
        ────────────────────
        Real HR data feeds, CSVs with missing values, and DataFrames loaded
        from SQL can all produce ``None`` or ``float('nan')`` in a
        ``skills_current`` cell.  Without this guard the first call to
        ``cls._DELIMITER_RE.split(raw)`` would raise::

            AttributeError: expected string or bytes-like object

        The guard uses three checks in deliberate order:
        1. ``raw is None``            – catches the Python None singleton.
        2. ``pd.isna(raw)``           – catches float NaN, pd.NA, pd.NaT,
                                        and numpy.nan without importing numpy
                                        directly (pandas is already a dependency).
        3. ``not isinstance(raw, str)`` – catches integers, lists, dicts, or
                                          any other unexpected type that slipped
                                          through upstream validation.
        The ``pd.isna`` call is wrapped in a bare ``except`` because some
        unhashable types (e.g. lists) raise a ``TypeError`` inside ``pd.isna``;
        in that case the third isinstance check handles the fallthrough.
        """
        # ── Type-guard: short-circuit on None, NaN, or non-string input ──────
        if raw is None:
            return []
        try:
            if pd.isna(raw):
                return []
        except (TypeError, ValueError):
            # pd.isna raises TypeError for unhashable types (e.g. list).
            # Fall through to the isinstance check below.
            pass
        if not isinstance(raw, str):
            return []
        # ── Normalisation pipeline ─────────────────────────────────────────
        tokens: List[str] = []
        for part in cls._DELIMITER_RE.split(raw):
            # Step 2 – strip outer whitespace
            stripped: str = part.strip()
            # Step 3 – collapse internal multi-spaces
            collapsed: str = cls._MULTI_SPACE_RE.sub(" ", stripped)
            # Step 4 – title-case
            titled: str = collapsed.title()
            # Step 5 – discard empties
            if titled:
                tokens.append(titled)
        return tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def to_sentence(cls, raw: str) -> str:
        """
        Convert a raw skill string into a space-joined sentence suitable
        for feeding to a sentence-transformer tokeniser.

        The transformer sees individual skill names as separate words,
        preventing the tokeniser from treating ``"Python,Pandas"`` as
        an unknown compound token.

        Parameters
        ----------
        raw : str
            Raw comma/semicolon-separated skill string.

        Returns
        -------
        str
            Space-separated, normalised skill tokens as a single string.
            E.g. ``"Python , pandas;  NumPy "`` → ``"Python Pandas Numpy"``.
        """
        return " ".join(cls._split_and_clean(raw))

    @classmethod
    def to_token_set(cls, raw: str) -> Set[str]:
        """
        Convert a raw skill string into a set of canonical tokens for
        keyword-overlap comparisons in the XAI layer.

        Using a ``set`` automatically deduplicates repeated skills
        (e.g. ``"Python, Python, SQL"`` → ``{"Python", "Sql"}``).

        Parameters
        ----------
        raw : str
            Raw comma/semicolon-separated skill string.

        Returns
        -------
        Set[str]
            Deduplicated set of canonical skill tokens.
        """
        return set(cls._split_and_clean(raw))

    @classmethod
    def to_token_list(cls, raw: str) -> List[str]:
        """
        Ordered version of to_token_set() – preserves insertion order
        (useful when the order of skills matters for display).

        Parameters
        ----------
        raw : str
            Raw comma/semicolon-separated skill string.

        Returns
        -------
        List[str]
            Ordered list of canonical skill tokens (no duplicates via
            seen-set tracking).
        """
        seen: Set[str] = set()
        result: List[str] = []
        for token in cls._split_and_clean(raw):
            if token not in seen:
                seen.add(token)
                result.append(token)
        return result

    @classmethod
    def normalise_batch(cls, raw_strings: List[str]) -> List[str]:
        """
        Apply ``to_sentence()`` to an entire list of raw skill strings.

        Used by EmbeddingEngine.encode_skills() to pre-process the full
        employee skill column in one vectorised-friendly call.

        Parameters
        ----------
        raw_strings : List[str]
            One entry per employee.

        Returns
        -------
        List[str]
            Sentence-form skill strings, one per employee.
        """
        return [cls.to_sentence(s) for s in raw_strings]


# ===========================================================================
# 1.  EMBEDDING ENGINE  (updated encode_skills in v2)
# ===========================================================================

class EmbeddingEngine:
    """
    Wraps a Hugging Face sentence-transformer model to produce embeddings
    for employee skill profiles and NPTEL course descriptions.

    v2 change
    ─────────
    ``encode_skills()`` now delegates ALL text pre-processing to
    ``SkillNormalizer.normalise_batch()``.  The previous inline
    ``s.replace(",", " ").strip()`` is removed.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model identifier.  The default
        ``'all-MiniLM-L6-v2'`` is compact (~80 MB) yet accurate for
        semantic similarity tasks and suitable for CPU inference.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        print(f"[EmbeddingEngine]  Loading model '{model_name}' …")
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self._normalizer = SkillNormalizer()   # shared utility
        print(f"[EmbeddingEngine]  Model ready.")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_skills(self, skill_strings: List[str]) -> np.ndarray:
        """
        Encode a list of raw skill strings into dense unit-norm vectors.

        Pre-processing (v2)
        ───────────────────
        Each raw string is passed through ``SkillNormalizer.normalise_batch()``
        before encoding.  This handles:

        * Comma  delimiters   → ``"Python,SQL"``
        * Semicolon delimiters → ``"Python;SQL"``
        * Mixed delimiters    → ``"Python, SQL; Pandas"``
        * Trailing whitespace → ``"  Python  "``
        * Internal spaces     → ``"REST  APIs"``
        * Irregular casing    → ``"python"`` / ``"PYTHON"``

        All become the same sentence form, guaranteeing that the
        transformer receives clean, consistent input regardless of
        how an employee or HR system entered the skills.

        Parameters
        ----------
        skill_strings : List[str]
            Raw comma/semicolon-separated skill strings, one per employee.

        Returns
        -------
        np.ndarray
            Shape ``(n_employees, embedding_dim)``, unit-normalised.
        """
        # ── v2: use SkillNormalizer instead of inline replace() ──────────
        normalised: List[str] = SkillNormalizer.normalise_batch(skill_strings)

        # Optional: log a sample normalisation for transparency
        if skill_strings and skill_strings[0] != normalised[0]:
            print(
                f"[EmbeddingEngine]  Skill normalisation sample:\n"
                f"    raw       : {skill_strings[0]!r}\n"
                f"    normalised: {normalised[0]!r}"
            )

        embeddings: np.ndarray = self.model.encode(
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
        only vocabulary overlap.  No normalisation needed here since
        descriptions are already well-formed prose.

        Parameters
        ----------
        descriptions : List[str]
            Full-text course descriptions, one per course.

        Returns
        -------
        np.ndarray
            Shape ``(n_courses, embedding_dim)``, unit-normalised.
        """
        embeddings: np.ndarray = self.model.encode(
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
        course_embeddings:   np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full cosine similarity matrix.

        Because embeddings are unit-normalised, cosine similarity equals
        the dot product and lies in [-1, 1]; in practice scores are
        positive (skill descriptions share positive semantic content).

        Parameters
        ----------
        employee_embeddings : np.ndarray  shape (n_employees, D)
        course_embeddings   : np.ndarray  shape (n_courses, D)

        Returns
        -------
        np.ndarray
            Shape ``(n_employees, n_courses)``; ``sim[i, j]`` is the
            cosine similarity between employee ``i`` and course ``j``.
        """
        sim_matrix: np.ndarray = cosine_similarity(
            employee_embeddings, course_embeddings
        )
        return sim_matrix.astype(np.float32)


# ===========================================================================
# 2.  ETHICAL FAIRNESS LAYER  (unchanged from v1)
# ===========================================================================

class EthicalFairnessLayer:
    """
    Implements a Fairlearn-inspired demographic-parity correction.

    Methodology
    ───────────
    1. Compute recommendation rate per group – for each protected
       attribute value (e.g. gender = 'Female'), count how many employees
       in that group would receive a recommendation above the base threshold
       under raw similarity scores.

    2. Compute parity gap – the difference between a group's rate and the
       mean recommendation rate across all groups.

    3. Apply bounded additive adjustment – add a positive delta to
       under-represented groups and a negative delta to over-represented
       ones.  The delta is capped at ``max_adjustment`` so merit always
       contributes ≥ 70 % of the final score.

    4. Record fair_dist_score – a scalar in [0, 1] measuring how close
       the adjusted distribution is to perfect demographic parity (1.0).

    Parameters
    ----------
    protected_attribute : str
        Column name in the employee DataFrame that defines demographic
        groups, e.g. ``'gender'``.
    base_threshold : float
        Raw similarity score above which an employee is counted as
        "receiving a recommendation" when computing group rates.
    max_adjustment : float
        Upper bound on score adjustment applied to any individual.
        Capped at 0.15 so merit dominates.
    """

    def __init__(
        self,
        protected_attribute: str   = "gender",
        base_threshold:      float = 0.30,
        max_adjustment:      float = 0.15,
    ) -> None:
        self.protected_attribute = protected_attribute
        self.base_threshold      = base_threshold
        self.max_adjustment      = max_adjustment
        self._group_adjustments:  Dict[str, float] = {}
        self._group_rates:        Dict[str, float] = {}
        self._parity_score:       float            = 0.0
        self._mean_rate:          float            = 0.0

    # ------------------------------------------------------------------
    # Core fairness computation
    # ------------------------------------------------------------------

    def fit(
        self,
        employees:      pd.DataFrame,
        raw_sim_matrix: np.ndarray,
    ) -> "EthicalFairnessLayer":
        """
        Analyse raw recommendation rates per demographic group and
        compute the adjustment coefficients.

        Parameters
        ----------
        employees      : DataFrame with at minimum ``protected_attribute`` column.
        raw_sim_matrix : shape (n_employees, n_courses); scores before fairness.

        Returns
        -------
        self  (for method chaining)
        """
        groups:     np.ndarray       = employees[self.protected_attribute].values
        max_scores: np.ndarray       = raw_sim_matrix.max(axis=1)
        group_labels: np.ndarray     = np.unique(groups)
        group_rates:  Dict[str, float] = {}

        for g in group_labels:
            mask:      np.ndarray = groups == g
            g_scores:  np.ndarray = max_scores[mask]
            rate:      float      = float((g_scores >= self.base_threshold).mean())
            group_rates[g]        = rate

        mean_rate: float = float(np.mean(list(group_rates.values())))

        adjustments: Dict[str, float] = {}
        for g, rate in group_rates.items():
            gap: float = mean_rate - rate
            adj: float = float(np.clip(gap, -self.max_adjustment, self.max_adjustment))
            adjustments[g] = adj

        self._group_rates       = group_rates
        self._group_adjustments = adjustments
        self._mean_rate         = mean_rate

        if mean_rate > 0:
            rate_range:   float = max(group_rates.values()) - min(group_rates.values())
            parity_score: float = max(0.0, 1.0 - (rate_range / mean_rate))
        else:
            parity_score = 1.0

        self._parity_score = round(parity_score, 4)

        print("\n[EthicalFairnessLayer]  Demographic parity analysis:")
        print(f"  Protected attribute : '{self.protected_attribute}'")
        print(f"  Mean recommendation rate (all groups) : {mean_rate:.1%}")
        for g in sorted(group_rates.keys()):
            adj_str: str = (
                f"+{adjustments[g]:.4f}" if adjustments[g] >= 0
                else f"{adjustments[g]:.4f}"
            )
            print(
                f"  Group '{g:20s}' | rate={group_rates[g]:.1%}"
                f"  | adjustment={adj_str}"
            )
        print(f"  Demographic parity score (pre-adjust) : {self._parity_score:.4f}")
        return self

    def adjust_scores(
        self,
        employees:      pd.DataFrame,
        raw_sim_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply pre-computed group adjustments to produce a fairness-
        corrected similarity matrix.

        Parameters
        ----------
        employees      : DataFrame (must contain ``protected_attribute``).
        raw_sim_matrix : shape (n_employees, n_courses).

        Returns
        -------
        adjusted_matrix : np.ndarray  shape (n_employees, n_courses)
        delta_matrix    : np.ndarray  shape (n_employees, n_courses)
            ``delta[i, j] = adjusted[i, j] – raw[i, j]``
        """
        if not self._group_adjustments:
            raise RuntimeError("Call .fit() before .adjust_scores().")

        groups:          np.ndarray = employees[self.protected_attribute].values
        adjusted_matrix: np.ndarray = raw_sim_matrix.copy()

        for idx, g in enumerate(groups):
            adj: float = self._group_adjustments.get(g, 0.0)
            adjusted_matrix[idx, :] = np.clip(
                raw_sim_matrix[idx, :] + adj, 0.0, 1.0
            )

        delta_matrix: np.ndarray = adjusted_matrix - raw_sim_matrix
        return adjusted_matrix.astype(np.float32), delta_matrix.astype(np.float32)

    def compute_adjusted_parity(
        self,
        employees:       pd.DataFrame,
        adjusted_matrix: np.ndarray,
    ) -> float:
        """Recompute the demographic parity score on the *adjusted* matrix."""
        groups:     np.ndarray       = employees[self.protected_attribute].values
        max_scores: np.ndarray       = adjusted_matrix.max(axis=1)
        group_rates: Dict[str, float] = {}

        for g in np.unique(groups):
            mask: np.ndarray = groups == g
            rate: float      = float((max_scores[mask] >= self.base_threshold).mean())
            group_rates[g]   = rate

        mean_rate:   float = float(np.mean(list(group_rates.values()))) or 1e-9
        rate_range:  float = max(group_rates.values()) - min(group_rates.values())
        score:       float = max(0.0, 1.0 - (rate_range / mean_rate))
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
# 3.  XAI EXPLAINER  (updated _tokenise in v2)
# ===========================================================================

class XAIExplainer:
    """
    Generates transparent, human-readable explanations for each course
    recommendation using keyword-overlap XAI.

    v2 change
    ─────────
    ``_tokenise()`` now delegates to ``SkillNormalizer.to_token_set()``
    instead of its own ``lower().split()`` logic.  This guarantees that
    the keyword overlap reported in the explanation matches the semantic
    signal used by the sentence-transformer, because both now use
    identical canonical token forms.

    Approach
    ────────
    Rather than treating the similarity score as a black box, we surface:
      (a) which of the employee's skills directly overlap with the course
          skill tags  (direct evidence);
      (b) which skills the course would add that the employee lacks
          (learning opportunity evidence);
      (c) the cosine similarity score  (quantitative evidence);
      (d) the fairness delta  (ethical transparency).
    """

    # ------------------------------------------------------------------
    # v2: unified tokeniser
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(skill_string: str) -> Set[str]:
        """
        Convert a raw skill string to a canonical token set.

        v2 change
        ─────────
        Replaced the previous implementation::

            return {s.strip().lower() for s in skill_string.split(",") if s.strip()}

        with a call to ``SkillNormalizer.to_token_set()``.

        This matters because:
        * Old code split ONLY on commas → semicolons produced wrong tokens.
        * Old code used ``.lower()`` while embeddings used ``.title()`` →
          the comparison was between different canonical forms and would fail
          for skills entered in all-caps or all-lowercase.
        * Now both paths produce the same title-cased, delimiter-agnostic,
          whitespace-stripped token set.

        Parameters
        ----------
        skill_string : str
            Raw comma/semicolon-separated skill string from either an
            employee record or a course skill_tags field.

        Returns
        -------
        Set[str]
            Deduplicated set of canonical (title-cased) skill tokens.
        """
        return SkillNormalizer.to_token_set(skill_string)

    # ------------------------------------------------------------------
    # Main explanation generator
    # ------------------------------------------------------------------

    def explain(
        self,
        employee_row: pd.Series,
        course_row:   pd.Series,
        raw_score:    float,
        fair_score:   float,
        fair_delta:   float,
        rank:         int = 1,
    ) -> str:
        """
        Build a plain-English explanation for a single (employee, course) pair.

        Parameters
        ----------
        employee_row : pd.Series   – row from the employees DataFrame.
        course_row   : pd.Series   – row from the nptel_courses DataFrame.
        raw_score    : float       – cosine similarity before fairness adjustment.
        fair_score   : float       – cosine similarity after fairness adjustment.
        fair_delta   : float       – ``fair_score – raw_score``.
        rank         : int         – 1-based rank of this course for this employee.

        Returns
        -------
        str
            Multi-sentence, multi-section human-readable explanation.
        """
        # Both employee skills and course tags are now normalised through
        # the same SkillNormalizer path, guaranteeing matched token forms.
        emp_skills:    Set[str] = self._tokenise(employee_row["skills_current"])
        course_skills: Set[str] = self._tokenise(course_row["skill_tags"])

        matched_skills: List[str] = sorted(emp_skills & course_skills)
        gap_skills:     List[str] = sorted(course_skills - emp_skills)

        lines: List[str] = []

        # ── Header ───────────────────────────────────────────────────────
        lines.append(
            f"RECOMMENDATION #{rank} for {employee_row['full_name']} "
            f"({employee_row['role']}, {employee_row['department']})"
        )
        lines.append(f"  Course  : {course_row['title']} [{course_row['difficulty']}]")
        lines.append(f"  Domain  : {course_row['domain']}")

        # ── Semantic similarity ───────────────────────────────────────────
        lines.append(
            f"\n  [SIMILARITY]  Cosine similarity score = {raw_score:.4f}  "
            "(range 0–1; higher = stronger profile match)."
        )

        # ── Skill match evidence ──────────────────────────────────────────
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

        # ── Skill gap / upskilling opportunity ────────────────────────────
        if gap_skills:
            lines.append(
                f"  [UPSKILLING OPPORTUNITY]  Completing this course would add "
                f"{len(gap_skills)} new skill(s) to your profile: "
                f"{', '.join(gap_skills)}."
            )

        # ── Displacement risk context ─────────────────────────────────────
        risk: float = float(employee_row.get("risk_score", 0.0))
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

        # ── Fairness transparency ─────────────────────────────────────────
        if abs(fair_delta) > 0.001:
            direction: str = "upward" if fair_delta > 0 else "downward"
            lines.append(
                f"  [FAIRNESS ADJUSTMENT]  The Ethical Fairness Layer applied a "
                f"{direction} score correction of {fair_delta:+.4f} based on "
                f"demographic parity analysis across "
                f"'{employee_row.get('gender', 'N/A')}' "
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
# 4.  RECOMMENDATION PIPELINE  (unchanged from v1)
# ===========================================================================

class RecommendationPipeline:
    """
    End-to-end pipeline that ties EmbeddingEngine, EthicalFairnessLayer,
    and XAIExplainer together and produces a structured output DataFrame
    suitable for insertion into the ``upskilling_logs`` MySQL table.

    Parameters
    ----------
    top_n : int
        Number of course recommendations to generate per employee.
    model_name : str
        Sentence-transformer model identifier.
    protected_attribute : str
        Demographic column name for fairness analysis.
    """

    def __init__(
        self,
        top_n:               int = 3,
        model_name:          str = "all-MiniLM-L6-v2",
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
        ─────
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
        pd.DataFrame  matching the schema of ``upskilling_logs``.
        """
        print("\n" + "=" * 60)
        print("RECOMMENDATION PIPELINE  –  START")
        print("=" * 60)

        # ── Step 1 & 2 : Embeddings ───────────────────────────────────────
        print("\n[Step 1]  Encoding employee skill profiles …")
        emp_embeddings: np.ndarray = self.embedding_engine.encode_skills(
            employees["skills_current"].tolist()
        )
        print(f"          → {emp_embeddings.shape} embeddings generated.")

        print("[Step 2]  Encoding NPTEL course descriptions …")
        course_embeddings: np.ndarray = self.embedding_engine.encode_courses(
            courses["description"].tolist()
        )
        print(f"          → {course_embeddings.shape} embeddings generated.")

        # ── Step 3 : Raw similarity ───────────────────────────────────────
        print("[Step 3]  Computing cosine similarity matrix …")
        raw_sim: np.ndarray = self.embedding_engine.compute_similarity_matrix(
            emp_embeddings, course_embeddings
        )
        print(
            f"          → Matrix shape: {raw_sim.shape}  "
            f"| Range: [{raw_sim.min():.4f}, {raw_sim.max():.4f}]"
        )

        # ── Step 4 : Fairness fit ─────────────────────────────────────────
        print("\n[Step 4]  Fitting Ethical Fairness Layer …")
        self.fairness_layer.fit(employees, raw_sim)

        # ── Step 5 : Fairness adjustment ──────────────────────────────────
        print("\n[Step 5]  Applying fairness score adjustments …")
        adj_sim, delta_sim = self.fairness_layer.adjust_scores(employees, raw_sim)
        post_parity: float = self.fairness_layer.compute_adjusted_parity(
            employees, adj_sim
        )
        print(f"          Demographic parity score AFTER adjustment : {post_parity:.4f}")
        print(
            f"          Improvement  : "
            f"{self.fairness_layer.parity_score_before:.4f}  →  {post_parity:.4f}"
        )

        # ── Steps 6, 7, 8 : Recommendations + XAI + Log ──────────────────
        print(f"\n[Step 6-8]  Generating top-{self.top_n} recommendations per employee …")
        log_records: List[dict] = []

        emp_sorted: pd.DataFrame = employees.sort_values(
            "risk_score", ascending=False
        )

        for _, emp in emp_sorted.iterrows():
            i: int = int(emp["employee_id"]) - 1

            course_adj_scores: np.ndarray = adj_sim[i]
            top_indices: np.ndarray = np.argsort(course_adj_scores)[::-1][: self.top_n]

            for rank, c_idx in enumerate(top_indices, start=1):
                course:     pd.Series = courses.iloc[c_idx]
                raw_score:  float     = float(raw_sim[i, c_idx])
                fair_score: float     = float(adj_sim[i, c_idx])
                fair_delta: float     = float(delta_sim[i, c_idx])

                explanation: str = self.xai_explainer.explain(
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

        logs_df: pd.DataFrame = pd.DataFrame(log_records)
        print(f"          → {len(logs_df)} log records created.")
        print("\n[PIPELINE]  ✓  Complete.\n")
        return logs_df
