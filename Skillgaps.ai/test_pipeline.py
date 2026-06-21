import pytest
import pandas as pd
import numpy as np
from models import SkillNormalizer, EthicalFairnessLayer

def test_skill_normalizer_cleansing():
    """Verify that SkillNormalizer eliminates trailing spaces, casing anomalies, and mixed delimiters."""
    raw_input = "  python ; pandas, NumPy "
    
    # Test text sentence formatting for Embedding Engine
    sentence = SkillNormalizer.to_sentence(raw_input)
    assert sentence == "Python Pandas Numpy"
    
    # Test deduplicated token set formatting for XAI Engine
    token_set = SkillNormalizer.to_token_set(raw_input)
    assert token_set == {"Python", "Pandas", "Numpy"}

def test_skill_normalizer_type_guards():
    """Verify that SkillNormalizer type guards prevent crashes on missing or null values."""
    assert SkillNormalizer._split_and_clean(None) == []
    assert SkillNormalizer._split_and_clean(float('nan')) == []
    assert SkillNormalizer._split_and_clean(12345) == []

def test_ethical_fairness_layer_bounds():
    """Verify that EthicalFairnessLayer modifications strictly enforce the max_adjustment ceiling."""
    fl = EthicalFairnessLayer(protected_attribute="gender", max_adjustment=0.15)
    
    # Create sample evaluation data shape
    mock_employees = pd.DataFrame({"gender": ["Male", "Male", "Female", "Female"]})
    mock_raw_sim = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.1, 0.1],
        [0.2, 0.1]
    ], dtype=np.float32)
    
    fl.fit(mock_employees, mock_raw_sim)
    adjusted_matrix, delta_matrix = fl.adjust_scores(mock_employees, mock_raw_sim)
    
    # Use an absolute tolerance buffer to handle machine precision variations securely
    assert np.all(np.abs(delta_matrix) <= 0.1501)
    
    # Ensure scores stay legally bounded within standard probability scales [0.0, 1.0]
    assert adjusted_matrix.min() >= 0.0
    assert adjusted_matrix.max() <= 1.0