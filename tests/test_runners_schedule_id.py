import pandas as pd
import torch

from caveat import runners


def test_ensure_schedule_id_column_adds_when_missing():
    df = pd.DataFrame(
        [
            {"pid": "a", "act": "home", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"pid": "a", "act": "work", "start": 10.0, "end": 20.0, "duration": 10.0},
        ]
    )
    out = runners._ensure_schedule_id_column(df)
    assert "schedule_id" in out.columns
    assert out["schedule_id"].tolist() == [0, 1]


def test_ensure_schedule_id_column_preserves_existing():
    df = pd.DataFrame(
        [
            {"schedule_id": 10, "pid": "a", "act": "home", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"schedule_id": 11, "pid": "a", "act": "work", "start": 10.0, "end": 20.0, "duration": 10.0},
        ]
    )
    out = runners._ensure_schedule_id_column(df)
    assert out["schedule_id"].tolist() == [10, 11]


def test_ensure_schedule_id_column_fills_missing_values():
    df = pd.DataFrame(
        [
            {"schedule_id": 4, "pid": "a", "act": "home", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"schedule_id": None, "pid": "a", "act": "work", "start": 10.0, "end": 20.0, "duration": 10.0},
        ]
    )
    out = runners._ensure_schedule_id_column(df)
    assert out["schedule_id"].tolist() == [4, 5]


def test_build_synthetic_interactions_from_logits():
    schedules = pd.DataFrame(
        [
            {
                "schedule_id": 7,
                "pid": "Gen_HH0_M0",
                "hid": 0,
                "member_idx": 0,
                "act": "work",
                "start": 0.0,
                "end": 480.0,
                "duration": 480.0,
            }
        ]
    )
    logits = torch.full((1, 1, 6), -10.0)
    logits[0, 0, 0] = 10.0
    logits[0, 0, 1] = 10.0
    mask = torch.tensor([[1.0]])

    out = runners._build_synthetic_interactions_from_logits(
        schedules_df=schedules,
        interaction_logits=logits,
        interaction_mask=mask,
        prob_threshold=0.5,
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["schedule_id"]) == 7
    assert row["pid"] == "Gen_HH0_M0"
    assert int(row["interaction_bin"]) == 1
    assert float(row["overlap_start"]) == 0.0
    assert float(row["overlap_end"]) == 480.0
