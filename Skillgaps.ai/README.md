# SkillGaps.AI 🚀 
### *Ethical Workforce Transition & Explainable Recommendation Engine*

SkillGaps.AI is a production-grade machine learning data pipeline that maps workforce skill profiles to educational target tracks (NPTEL courses)[cite: 1, 2]. Built with strict object-oriented patterns, it integrates dense semantic embeddings with an enterprise demographic-parity fairness layer and exact keyword-overlap Explainable AI (XAI)[cite: 1, 3].

---

## 🏗️ Architectural Core

The ecosystem decouples data operations into distinct modules:
*   **`main.py`**: Pipelines orchestration engine implementing a hardened connection pool lifecycle and automated CSV dry-run fallbacks.
*   **`models.py`**: Core algorithmic processing center housing the `SkillNormalizer` string sanitation engine, the Hugging Face `SentenceTransformer` vectors layer, the `EthicalFairnessLayer`, and the plain-English `XAIExplainer` module[cite: 1, 3].
*   **`db.py`**: Robust database adapter abstracting multi-statement DDL handling, relational auto-increment insertions, chunked bulk writes, and strict data type constraints via SQLAlchemy[cite: 2].
*   **`data_generation.py`**: Generates synthetic organizational worker metrics and contains the core target schema mappings[cite: 1, 2].

---

## ⚡ Quick Start & Execution

### 1. System Setup
Initialize an isolated virtual environment context and load dependencies using `uv`:
```bash
# Create local virtual environment
uv venv

# Install requirement manifests
uv pip install -r requirements.txt