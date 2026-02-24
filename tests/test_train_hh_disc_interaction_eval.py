from pathlib import Path

import pandas as pd
import pytest

import caveat.train_hh_disc as train_hh_disc


class _DummyThemeEvaluator:
    def __init__(self, config=None):
        self.config = config or {}

    def evaluate_all(self, df_ref, df_model, label_model):
        return pd.DataFrame(
            [{"Theme": "Timing", "Metric": "dummy", "Label": label_model, "Score": 0.0}]
        )


class _DummyVisualizer:
    def __init__(self, config=None):
        self.config = config or {}

    def plot_marginal_distributions(self, **kwargs):
        return None

    def plot_joint_distribution_2d(self, **kwargs):
        return None

    def plot_stacked_tempograms(self, **kwargs):
        return None


def _prepare_minimal_eval_environment(tmp_path: Path):
    version_dir = tmp_path / "logs" / "exp" / "version_0"
    version_dir.mkdir(parents=True, exist_ok=True)

    observed_path = tmp_path / "observed.csv"
    observed = pd.DataFrame(
        [
            {
                "schedule_id": 0,
                "pid": "p0",
                "hid": 0,
                "member_idx": 0,
                "act": "trip",
                "start": 0.0,
                "end": 60.0,
                "duration": 60.0,
            }
        ]
    )
    observed.to_csv(observed_path, index=False)

    synthetic = pd.DataFrame(
        [
            {
                "schedule_id": 0,
                "pid": "Gen_HH0_M0",
                "hid": 0,
                "member_idx": 0,
                "act": "trip",
                "start": 0.0,
                "end": 60.0,
                "duration": 60.0,
            }
        ]
    )
    synthetic.to_csv(version_dir / "synthetic_schedules.csv", index=False)
    return version_dir, observed_path


@pytest.mark.parametrize(
    "model_name",
    [
        "household_discCNN_CVAE_att_interactions",
        "household_discCNN_CVAE_fullattention_interactions",
    ],
)
def test_run_enhanced_evaluation_uses_overlap_evaluator_for_interaction_model(
    tmp_path, monkeypatch, model_name
):
    _prepare_minimal_eval_environment(tmp_path)

    calls = {"overlap": 0, "joint": 0}

    monkeypatch.setattr(train_hh_disc, "ThemeEvaluator", _DummyThemeEvaluator)
    monkeypatch.setattr(train_hh_disc, "Visualizer", _DummyVisualizer)
    monkeypatch.setattr(train_hh_disc, "prepare_eval_df_from_minutes", lambda x: x)
    monkeypatch.setattr(train_hh_disc, "visualize_latent_umap", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_hh_disc, "plot_mu_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_hh_disc,
        "run_interaction_overlap_evaluation",
        lambda *args, **kwargs: calls.__setitem__("overlap", calls["overlap"] + 1),
    )
    monkeypatch.setattr(
        train_hh_disc,
        "run_joint_trip_evaluation",
        lambda *args, **kwargs: calls.__setitem__("joint", calls["joint"] + 1),
    )

    config = {
        "logging_params": {"log_dir": str(tmp_path / "logs"), "name": "exp"},
        "model_params": {"name": model_name},
        "schedules_path": str(tmp_path / "observed.csv"),
        "interactions_path": str(tmp_path / "ref_interactions.csv"),
    }
    eval_params = {
        "enabled": True,
        "generate_plots": False,
        "joint_trip_evaluation": {"enabled": True, "generate_plots": False},
        "conditional_evaluation": {"enabled": False},
    }

    train_hh_disc.run_enhanced_evaluation(config, eval_params)
    assert calls["overlap"] == 1
    assert calls["joint"] == 0


def test_run_enhanced_evaluation_keeps_joint_trip_for_non_interaction_model(
    tmp_path, monkeypatch
):
    _prepare_minimal_eval_environment(tmp_path)

    calls = {"overlap": 0, "joint": 0}

    monkeypatch.setattr(train_hh_disc, "ThemeEvaluator", _DummyThemeEvaluator)
    monkeypatch.setattr(train_hh_disc, "Visualizer", _DummyVisualizer)
    monkeypatch.setattr(train_hh_disc, "prepare_eval_df_from_minutes", lambda x: x)
    monkeypatch.setattr(train_hh_disc, "visualize_latent_umap", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_hh_disc, "plot_mu_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_hh_disc,
        "run_interaction_overlap_evaluation",
        lambda *args, **kwargs: calls.__setitem__("overlap", calls["overlap"] + 1),
    )
    monkeypatch.setattr(
        train_hh_disc,
        "run_joint_trip_evaluation",
        lambda *args, **kwargs: calls.__setitem__("joint", calls["joint"] + 1),
    )

    config = {
        "logging_params": {"log_dir": str(tmp_path / "logs"), "name": "exp"},
        "model_params": {"name": "household_discCNN_CVAE_attention"},
        "schedules_path": str(tmp_path / "observed.csv"),
    }
    eval_params = {
        "enabled": True,
        "generate_plots": False,
        "joint_trip_evaluation": {"enabled": True, "generate_plots": False},
        "conditional_evaluation": {"enabled": False},
    }

    train_hh_disc.run_enhanced_evaluation(config, eval_params)
    assert calls["overlap"] == 0
    assert calls["joint"] == 1
