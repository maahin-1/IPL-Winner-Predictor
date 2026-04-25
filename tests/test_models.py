"""
Model output range and schema validation.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_a_prematch import ModelA_PreMatchXGB, INPUT_FEATURES as A_FEATURES
from models.model_b_inmatch import ModelB_InMatchLGB, INPUT_FEATURES as B_FEATURES
from models.model_c_player_impact import ModelC_PlayerImpactXGB, INPUT_FEATURES as C_FEATURES
from models.model_d_season import ModelD_SeasonTrajLGB, INPUT_FEATURES as D_FEATURES
from models.meta_learner import MetaLearner, META_FEATURES


def _make_df(features: list[str], n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((n, len(features))), columns=features)


@pytest.fixture
def trained_model_a():
    df = _make_df(A_FEATURES, 100)
    labels = pd.Series(np.random.randint(0, 2, 100))
    m = ModelA_PreMatchXGB()
    m.fit(df, labels)
    return m


@pytest.fixture
def trained_model_b():
    df = _make_df(B_FEATURES, 100)
    labels = pd.Series(np.random.randint(0, 2, 100))
    m = ModelB_InMatchLGB()
    m.fit(df, labels)
    return m


@pytest.fixture
def trained_model_c():
    df = _make_df(C_FEATURES, 100)
    scores = pd.Series(np.random.uniform(0, 100, 100))
    m = ModelC_PlayerImpactXGB()
    m.fit(df, scores)
    return m


@pytest.fixture
def trained_model_d():
    df = _make_df(D_FEATURES, 100)
    y_w = pd.Series(np.random.randint(0, 2, 100))
    y_p = pd.Series(np.random.randint(0, 2, 100))
    m = ModelD_SeasonTrajLGB()
    m.fit(df, y_w, y_p)
    return m


def test_model_a_probability_in_range(trained_model_a):
    sample = _make_df(A_FEATURES, 1)
    prob = trained_model_a.predict_win_probability(sample)
    assert 0.0 <= prob <= 1.0


def test_model_b_probability_in_range(trained_model_b):
    sample = _make_df(B_FEATURES, 1)
    prob = trained_model_b.predict_win_probability(sample)
    assert 0.0 <= prob <= 1.0


def test_model_c_score_in_range(trained_model_c):
    sample = _make_df(C_FEATURES, 5)
    scores = trained_model_c.predict(sample)
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_model_d_outputs_dict(trained_model_d):
    sample = _make_df(D_FEATURES, 1)
    result = trained_model_d.predict(sample)
    assert "season_winner_probability" in result
    assert "playoff_qualification_probability" in result
    assert 0.0 <= result["season_winner_probability"] <= 1.0
    assert 0.0 <= result["playoff_qualification_probability"] <= 1.0


def test_model_a_raises_on_missing_features():
    m = ModelA_PreMatchXGB()
    m._model = None
    with pytest.raises(RuntimeError):
        m.predict_win_probability(pd.DataFrame([{}]))


def test_meta_learner_output_in_range():
    df = _make_df(META_FEATURES, 100)
    labels = pd.Series(np.random.randint(0, 2, 100))
    meta = MetaLearner(n_folds=3)
    meta.fit(df, labels)
    sample = _make_df(META_FEATURES, 1)
    prob = meta.predict_calibrated_win_probability(sample)
    assert 0.0 <= prob <= 1.0
