from pathlib import Path

import pandas as pd

from caveat.evaluate.interaction_overlap_evaluator import (
    evaluate_from_interval_csv,
    evaluate_from_test_inference,
)


def test_evaluate_from_test_inference_overlap_and_metrics(tmp_path: Path):
    test_dir = tmp_path / "test_inference"
    out_dir = tmp_path / "out"
    test_dir.mkdir(parents=True, exist_ok=True)

    # B=1, M=2, T=6 -> 12 columns flattened by member then time.
    targets = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    probs = [[0.1, 0.9, 0.1, 0, 0, 0, 0, 0, 0.8, 0, 0, 0]]
    member_mask = [[1, 1]]

    pd.DataFrame(targets).to_csv(test_dir / "interaction_targets.csv", header=False, index=False)
    pd.DataFrame(probs).to_csv(test_dir / "interaction_probs.csv", header=False, index=False)
    pd.DataFrame(member_mask).to_csv(test_dir / "interaction_mask.csv", header=False, index=False)

    schedules = pd.DataFrame(
        [
            {
                "schedule_id": 0,
                "pid": "Gen_HH0_M0",
                "hid": 0,
                "member_idx": 0,
                "act": "trip",
                "start": 240.0,
                "end": 480.0,
                "duration": 240.0,
            },
            {
                "schedule_id": 1,
                "pid": "Gen_HH0_M0",
                "hid": 0,
                "member_idx": 0,
                "act": "work",
                "start": 480.0,
                "end": 720.0,
                "duration": 240.0,
            },
            {
                "schedule_id": 2,
                "pid": "Gen_HH0_M1",
                "hid": 0,
                "member_idx": 1,
                "act": "work",
                "start": 480.0,
                "end": 720.0,
                "duration": 240.0,
            },
        ]
    )
    schedules.to_csv(test_dir / "input_schedules.csv", index=False)
    schedules.to_csv(test_dir / "inferred_schedules.csv", index=False)

    result = evaluate_from_test_inference(
        test_inference_dir=test_dir,
        write_path=out_dir,
        label_model="model_x",
        trip_activity_name="trip",
        prob_threshold=0.5,
        generate_plots=False,
    )

    summary = result["summary"]
    ref = summary.loc[summary["Label"] == "Reference"].iloc[0]
    mod = summary.loc[summary["Label"] == "model_x"].iloc[0]

    assert int(ref["Joint Trips"]) == 1
    assert int(ref["Joint Activities"]) == 1
    assert int(mod["Joint Trips"]) == 1
    assert int(mod["Joint Activities"]) == 1

    metrics = set(result["emd_results"]["Metric"].astype(str).tolist())
    assert "Escort Trip Start Time" not in metrics
    assert metrics == {
        "Joint Activity Duration",
        "Joint Activity Start Time",
        "Joint Trip Start Time",
    }


def test_evaluate_from_interval_csv_uses_strict_overlap(tmp_path: Path):
    out_dir = tmp_path / "out_interval"

    ref_sched = pd.DataFrame(
        [
            {"schedule_id": 1, "pid": "A", "act": "trip", "start": 0.0, "end": 100.0, "duration": 100.0},
            {"schedule_id": 2, "pid": "A", "act": "work", "start": 100.0, "end": 200.0, "duration": 100.0},
            {"schedule_id": 3, "pid": "B", "act": "work", "start": 200.0, "end": 300.0, "duration": 100.0},
        ]
    )
    model_sched = pd.DataFrame(
        [
            {"schedule_id": 11, "pid": "C", "act": "trip", "start": 0.0, "end": 100.0, "duration": 100.0},
            {"schedule_id": 12, "pid": "C", "act": "work", "start": 100.0, "end": 200.0, "duration": 100.0},
            {"schedule_id": 13, "pid": "D", "act": "work", "start": 200.0, "end": 300.0, "duration": 100.0},
        ]
    )

    ref_inter = pd.DataFrame(
        [
            {"schedule_id": 1, "pid": "A", "interaction_bin": 1, "overlap_start": 10.0, "overlap_end": 90.0},
            {"schedule_id": 2, "pid": "A", "interaction_bin": 1, "overlap_start": 120.0, "overlap_end": 150.0},
            {"schedule_id": 3, "pid": "B", "interaction_bin": 1, "overlap_start": 300.0, "overlap_end": 330.0},
        ]
    )
    model_inter = pd.DataFrame(
        [
            {"schedule_id": 11, "pid": "C", "interaction_bin": 1, "overlap_start": 0.0, "overlap_end": 100.0},
            {"schedule_id": 12, "pid": "C", "interaction_bin": 1, "overlap_start": 180.0, "overlap_end": 210.0},
        ]
    )

    result = evaluate_from_interval_csv(
        reference_schedules=ref_sched,
        model_schedules=model_sched,
        reference_interactions=ref_inter,
        model_interactions=model_inter,
        write_path=out_dir,
        label_model="model_y",
        trip_activity_name="trip",
        generate_plots=False,
    )

    ref_detected = result["ref_detected"]
    row_sid3 = ref_detected.loc[ref_detected["schedule_id"] == 3].iloc[0]
    assert bool(row_sid3["is_joint_activity"]) is False

    summary = result["summary"]
    ref = summary.loc[summary["Label"] == "Reference"].iloc[0]
    assert int(ref["Joint Trips"]) == 1
    assert int(ref["Joint Activities"]) == 1
