"""
évaluateur de recouvrement d'interactions pour les modèles household avec supervision d'interactions.

calcule les métriques de déplacements conjoints et d'activités conjointes directement
à partir de la supervision/prédiction d'interactions plutôt que de les inférer depuis les schedules.
les métriques d'accompagnement sont volontairement exclues ici.

utilisé par train_hh_disc pour évaluer la qualité de la branche d'interactions
du modèle dual-branch (est-ce que le modèle prédit bien quand les membres interagissent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance


def _safe_emd(a: pd.Series, b: pd.Series) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    return wasserstein_distance(a.to_list(), b.to_list())


def _ensure_time_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "start" in out.columns:
        out["start"] = pd.to_numeric(out["start"], errors="coerce")
    if "duration" not in out.columns:
        if {"start", "end"}.issubset(out.columns):
            out["end"] = pd.to_numeric(out["end"], errors="coerce")
            out["duration"] = out["end"] - out["start"]
        else:
            raise ValueError("Missing 'duration' and cannot infer it (requires 'start' and 'end').")
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    required = [c for c in ["start", "duration"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)
    return out


def _ensure_hid_member_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"hid", "member_idx"}.issubset(out.columns):
        out["_hid_eval"] = pd.to_numeric(out["hid"], errors="coerce").fillna(-1).astype(int)
        out["_member_eval"] = (
            pd.to_numeric(out["member_idx"], errors="coerce").fillna(-1).astype(int)
        )
        return out

    if "pid" not in out.columns:
        raise ValueError(
            "Schedules must contain either ['hid', 'member_idx'] or 'pid' with HH/M pattern."
        )

    extracted = out["pid"].astype(str).str.extract(r"HH(?P<hid>\d+)_M(?P<member>\d+)")
    if extracted["hid"].isna().all() or extracted["member"].isna().all():
        raise ValueError(
            "Could not derive household/member indices from pid values."
        )

    out["_hid_eval"] = pd.to_numeric(extracted["hid"], errors="coerce").fillna(-1).astype(int)
    out["_member_eval"] = (
        pd.to_numeric(extracted["member"], errors="coerce").fillna(-1).astype(int)
    )
    return out


def _activity_slot_interaction_flag(
    start: float,
    end: float,
    slot_values: np.ndarray,
    slot_edges: np.ndarray,
) -> bool:
    if end <= start:
        return False
    step_size = float(slot_edges[1] - slot_edges[0])
    s_slot = int(start / step_size)
    e_slot = int((end - 1e-4) / step_size)
    if e_slot < 0 or s_slot >= len(slot_values):
        return False
    s_slot = max(0, s_slot)
    e_slot = min(len(slot_values) - 1, e_slot)
    if s_slot > e_slot:
        return False

    edges = slot_edges[s_slot : e_slot + 2]
    overlap = np.minimum(end, edges[1:]) - np.maximum(start, edges[:-1])
    overlap = np.maximum(overlap, 0.0)
    valid = overlap > 0.0
    if not np.any(valid):
        return False
    return bool(np.any((slot_values[s_slot : e_slot + 1] > 0) & valid))


def _mark_joint_flags_from_slot_tensor(
    schedules: pd.DataFrame,
    interactions_bmt: np.ndarray,
    member_mask: np.ndarray,
    trip_activity_name: str,
) -> pd.DataFrame:
    out = _ensure_hid_member_columns(schedules)
    out = out.copy()
    out["_is_joint"] = False

    b, m, t = interactions_bmt.shape
    slot_edges = np.arange(t + 1, dtype=np.float32) * (1440.0 / float(t))
    mask_ok = member_mask.shape == (b, m)

    for idx, row in out.iterrows():
        hh = int(row["_hid_eval"])
        mm = int(row["_member_eval"])
        if hh < 0 or mm < 0 or hh >= b or mm >= m:
            continue
        if mask_ok and not bool(member_mask[hh, mm]):
            continue

        try:
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue

        slot_values = interactions_bmt[hh, mm]
        out.at[idx, "_is_joint"] = _activity_slot_interaction_flag(
            start=start,
            end=end,
            slot_values=slot_values,
            slot_edges=slot_edges,
        )

    is_trip = out["act"].astype(str).str.lower().eq(trip_activity_name.lower())
    out["is_joint_trip"] = out["_is_joint"] & is_trip
    out["is_joint_activity"] = out["_is_joint"] & ~is_trip
    return out


def _load_interaction_arrays(
    probs_path: Path,
    targets_path: Path,
    mask_path: Path,
    prob_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_2d = pd.read_csv(probs_path, header=None).to_numpy(dtype=np.float32)
    targets_2d = pd.read_csv(targets_path, header=None).to_numpy(dtype=np.float32)
    mask_2d = pd.read_csv(mask_path, header=None).to_numpy(dtype=np.float32)

    if probs_2d.shape != targets_2d.shape:
        raise ValueError(
            f"Shape mismatch probs/targets: {probs_2d.shape} vs {targets_2d.shape}"
        )
    if mask_2d.shape[0] != probs_2d.shape[0]:
        raise ValueError(
            f"Shape mismatch mask/probs rows: {mask_2d.shape[0]} vs {probs_2d.shape[0]}"
        )

    b = probs_2d.shape[0]
    members = mask_2d.shape[1]
    if members <= 0 or probs_2d.shape[1] % members != 0:
        raise ValueError(
            "Cannot reshape interaction matrix to [B, M, T]: invalid members/time dimensions."
        )

    t = probs_2d.shape[1] // members
    probs_bmt = probs_2d.reshape(b, members, t)
    targets_bmt = targets_2d.reshape(b, members, t)
    probs_bin = (probs_bmt >= float(prob_threshold)).astype(np.uint8)
    targets_bin = (targets_bmt > 0).astype(np.uint8)
    member_mask = mask_2d > 0
    return probs_bin, targets_bin, member_mask


def _build_interval_map(interactions: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
    required = {"schedule_id", "interaction_bin", "overlap_start", "overlap_end"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(
            f"Interactions must contain columns {sorted(required)}; missing={sorted(missing)}"
        )

    df = interactions.copy()
    df["schedule_id"] = pd.to_numeric(df["schedule_id"], errors="coerce")
    df["interaction_bin"] = pd.to_numeric(df["interaction_bin"], errors="coerce").fillna(0)
    df["overlap_start"] = pd.to_numeric(df["overlap_start"], errors="coerce")
    df["overlap_end"] = pd.to_numeric(df["overlap_end"], errors="coerce")

    valid = df.loc[
        (df["schedule_id"].notna())
        & (df["interaction_bin"] > 0)
        & (df["overlap_start"].notna())
        & (df["overlap_end"].notna())
        & (df["overlap_end"] > df["overlap_start"])
    ]

    interval_map: Dict[int, List[Tuple[float, float]]] = {}
    for sid, group in valid.groupby("schedule_id", sort=False):
        sid_int = int(sid)
        interval_map[sid_int] = list(
            zip(group["overlap_start"].astype(float), group["overlap_end"].astype(float))
        )
    return interval_map


def _has_interval_overlap(
    start: float, end: float, intervals: List[Tuple[float, float]]
) -> bool:
    if end <= start:
        return False
    for int_start, int_end in intervals:
        overlap = min(end, float(int_end)) - max(start, float(int_start))
        if overlap > 0.0:
            return True
    return False


def _mark_joint_flags_from_interval_map(
    schedules: pd.DataFrame,
    interval_map: Dict[int, List[Tuple[float, float]]],
    trip_activity_name: str,
) -> pd.DataFrame:
    if "schedule_id" not in schedules.columns:
        raise ValueError("Schedules must contain 'schedule_id' for interval-based evaluation.")

    out = schedules.copy()
    out["_schedule_id_eval"] = pd.to_numeric(out["schedule_id"], errors="coerce")
    out["_is_joint"] = False

    for idx, row in out.iterrows():
        sid = row["_schedule_id_eval"]
        if pd.isna(sid):
            continue
        intervals = interval_map.get(int(sid), [])
        if not intervals:
            continue
        try:
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue
        out.at[idx, "_is_joint"] = _has_interval_overlap(start, end, intervals)

    is_trip = out["act"].astype(str).str.lower().eq(trip_activity_name.lower())
    out["is_joint_trip"] = out["_is_joint"] & is_trip
    out["is_joint_activity"] = out["_is_joint"] & ~is_trip
    return out


def compute_emd_metrics(
    df_ref: pd.DataFrame,
    df_model: pd.DataFrame,
    label_model: str = "Model",
    trip_activity_name: str = "trip",
) -> pd.DataFrame:
    results = []
    total_ref = max(len(df_ref), 1)
    total_model = max(len(df_model), 1)

    def append_metric_rows(
        metric_name: str,
        filter_col: str,
        target_col: str,
        unit: str,
        split_by_act: bool = False,
    ):
        ref_sel = df_ref.loc[df_ref[filter_col] == True]
        mod_sel = df_model.loc[df_model[filter_col] == True]

        emd = _safe_emd(ref_sel[target_col], mod_sel[target_col])
        results.append(
            {
                "Metric": metric_name,
                "Purpose": "Combined",
                "Unit": unit,
                "EMD": emd,
                "N_Ref": len(ref_sel),
                f"N_{label_model}": len(mod_sel),
                "Prop_Ref": len(ref_sel) / total_ref,
                f"Prop_{label_model}": len(mod_sel) / total_model,
            }
        )

        if not split_by_act:
            return

        acts = set(ref_sel["act"].astype(str).unique()) | set(mod_sel["act"].astype(str).unique())
        for act in acts:
            if str(act).lower() == trip_activity_name.lower():
                continue
            ref_a = ref_sel.loc[ref_sel["act"].astype(str) == str(act), target_col]
            mod_a = mod_sel.loc[mod_sel["act"].astype(str) == str(act), target_col]
            emd_a = _safe_emd(ref_a, mod_a)
            results.append(
                {
                    "Metric": metric_name,
                    "Purpose": str(act),
                    "Unit": unit,
                    "EMD": emd_a,
                    "N_Ref": len(ref_a),
                    f"N_{label_model}": len(mod_a),
                    "Prop_Ref": len(ref_a) / total_ref,
                    f"Prop_{label_model}": len(mod_a) / total_model,
                }
            )

    append_metric_rows(
        metric_name="Joint Activity Duration",
        filter_col="is_joint_activity",
        target_col="duration",
        unit="Minutes",
        split_by_act=True,
    )
    append_metric_rows(
        metric_name="Joint Activity Start Time",
        filter_col="is_joint_activity",
        target_col="start",
        unit="Minutes",
        split_by_act=True,
    )
    append_metric_rows(
        metric_name="Joint Trip Start Time",
        filter_col="is_joint_trip",
        target_col="start",
        unit="Minutes",
        split_by_act=False,
    )

    cols = [
        "Metric",
        "Purpose",
        "Unit",
        "EMD",
        "N_Ref",
        f"N_{label_model}",
        "Prop_Ref",
        f"Prop_{label_model}",
    ]
    return pd.DataFrame(results, columns=cols)


def build_summary(
    df_ref: pd.DataFrame,
    df_model: pd.DataFrame,
    label_model: str = "Model",
    trip_activity_name: str = "trip",
) -> pd.DataFrame:
    def stats(df: pd.DataFrame, label: str) -> dict:
        is_trip = df["act"].astype(str).str.lower().eq(trip_activity_name.lower())
        n_total = len(df)
        n_trips = int(is_trip.sum())
        n_non_trips = max(n_total - n_trips, 1)
        n_joint_trips = int(df["is_joint_trip"].sum())
        n_joint_activities = int(df["is_joint_activity"].sum())
        return {
            "Label": label,
            "Total Activities": n_total,
            "Total Trips": n_trips,
            "Joint Trips": n_joint_trips,
            "Joint Trip %": 100.0 * n_joint_trips / max(n_trips, 1),
            "Joint Activities": n_joint_activities,
            "Joint Activity %": 100.0 * n_joint_activities / n_non_trips,
        }

    return pd.DataFrame([stats(df_ref, "Reference"), stats(df_model, label_model)])


def _plot_distributions(
    df_ref: pd.DataFrame,
    df_model: pd.DataFrame,
    emd_results: pd.DataFrame,
    write_path: Path,
    label_model: str,
    min_samples: int = 10,
) -> None:
    if emd_results.empty:
        return

    sns.set_theme(style="whitegrid")
    metric_config = {
        "Joint Activity Duration": ("is_joint_activity", "duration"),
        "Joint Activity Start Time": ("is_joint_activity", "start"),
        "Joint Trip Start Time": ("is_joint_trip", "start"),
    }
    write_path.mkdir(parents=True, exist_ok=True)

    for _, row in emd_results.iterrows():
        metric = row["Metric"]
        purpose = row["Purpose"]
        if metric not in metric_config:
            continue
        filter_col, value_col = metric_config[metric]

        ref = df_ref.loc[df_ref[filter_col] == True]
        mod = df_model.loc[df_model[filter_col] == True]
        if purpose != "Combined":
            ref = ref.loc[ref["act"].astype(str) == str(purpose)]
            mod = mod.loc[mod["act"].astype(str) == str(purpose)]

        if len(ref) < min_samples or len(mod) < min_samples:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(
            ref[value_col], ax=ax, color="black", fill=True, alpha=0.1, linewidth=2,
            label=f"Reference (N={len(ref)})",
        )
        emd_val = row["EMD"]
        lab = (
            f"{label_model} (EMD: {emd_val:.2f} min, N={len(mod)})"
            if pd.notna(emd_val)
            else f"{label_model} (N={len(mod)})"
        )
        sns.kdeplot(
            mod[value_col], ax=ax, color="tab:orange", linewidth=2, linestyle="--", label=lab
        )

        ax.set_title(f"{metric} - {purpose}")
        ax.set_ylabel("Density")
        ax.legend()
        if "Start Time" in metric:
            ax.set_xlabel("Time of Day (Hours)")
            ax.set_xlim(0, 1440)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x // 60)}h")
            )
            ax.xaxis.set_major_locator(ticker.MultipleLocator(240))
        else:
            ax.set_xlabel("Duration (Hours)")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x / 60:.1f}h")
            )

        safe_metric = metric.replace(" ", "_")
        safe_purpose = str(purpose).replace(" ", "_").replace("/", "_")
        plt.tight_layout()
        plt.savefig(write_path / f"interaction_dist_{safe_metric}_{safe_purpose}.png", dpi=150)
        plt.close()


def evaluate_from_test_inference(
    test_inference_dir: Union[str, Path],
    write_path: Union[str, Path],
    label_model: str = "Model",
    trip_activity_name: str = "trip",
    prob_threshold: float = 0.5,
    generate_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    test_dir = Path(test_inference_dir)
    write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    probs_path = test_dir / "interaction_probs.csv"
    targets_path = test_dir / "interaction_targets.csv"
    mask_path = test_dir / "interaction_mask.csv"
    input_sched_path = test_dir / "input_schedules.csv"
    inferred_sched_path = test_dir / "inferred_schedules.csv"

    missing = [
        p.name
        for p in [probs_path, targets_path, mask_path, input_sched_path, inferred_sched_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing test_inference files for interaction overlap evaluation: {missing}"
        )

    model_bin, ref_bin, member_mask = _load_interaction_arrays(
        probs_path=probs_path,
        targets_path=targets_path,
        mask_path=mask_path,
        prob_threshold=prob_threshold,
    )
    ref_sched = pd.read_csv(input_sched_path)
    model_sched = pd.read_csv(inferred_sched_path)

    ref_detected = _mark_joint_flags_from_slot_tensor(
        schedules=ref_sched,
        interactions_bmt=ref_bin,
        member_mask=member_mask,
        trip_activity_name=trip_activity_name,
    )
    model_detected = _mark_joint_flags_from_slot_tensor(
        schedules=model_sched,
        interactions_bmt=model_bin,
        member_mask=member_mask,
        trip_activity_name=trip_activity_name,
    )
    ref_detected = _ensure_time_metrics_columns(ref_detected)
    model_detected = _ensure_time_metrics_columns(model_detected)

    emd = compute_emd_metrics(
        df_ref=ref_detected,
        df_model=model_detected,
        label_model=label_model,
        trip_activity_name=trip_activity_name,
    )
    summary = build_summary(
        df_ref=ref_detected,
        df_model=model_detected,
        label_model=label_model,
        trip_activity_name=trip_activity_name,
    )

    summary.to_csv(write_path / "interaction_behavior_summary.csv", index=False)
    emd.to_csv(write_path / "interaction_behavior_emd.csv", index=False)
    if generate_plots:
        _plot_distributions(
            df_ref=ref_detected,
            df_model=model_detected,
            emd_results=emd,
            write_path=write_path,
            label_model=label_model,
        )

    return {
        "summary": summary,
        "emd_results": emd,
        "ref_detected": ref_detected,
        "model_detected": model_detected,
    }


def evaluate_from_interval_csv(
    reference_schedules: pd.DataFrame,
    model_schedules: pd.DataFrame,
    reference_interactions: pd.DataFrame,
    model_interactions: pd.DataFrame,
    write_path: Union[str, Path],
    label_model: str = "Model",
    trip_activity_name: str = "trip",
    generate_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    ref_map = _build_interval_map(reference_interactions)
    model_map = _build_interval_map(model_interactions)

    ref_detected = _mark_joint_flags_from_interval_map(
        schedules=reference_schedules,
        interval_map=ref_map,
        trip_activity_name=trip_activity_name,
    )
    model_detected = _mark_joint_flags_from_interval_map(
        schedules=model_schedules,
        interval_map=model_map,
        trip_activity_name=trip_activity_name,
    )
    ref_detected = _ensure_time_metrics_columns(ref_detected)
    model_detected = _ensure_time_metrics_columns(model_detected)

    emd = compute_emd_metrics(
        df_ref=ref_detected,
        df_model=model_detected,
        label_model=label_model,
        trip_activity_name=trip_activity_name,
    )
    summary = build_summary(
        df_ref=ref_detected,
        df_model=model_detected,
        label_model=label_model,
        trip_activity_name=trip_activity_name,
    )

    summary.to_csv(write_path / "interaction_behavior_summary.csv", index=False)
    emd.to_csv(write_path / "interaction_behavior_emd.csv", index=False)
    if generate_plots:
        _plot_distributions(
            df_ref=ref_detected,
            df_model=model_detected,
            emd_results=emd,
            write_path=write_path,
            label_model=label_model,
        )

    return {
        "summary": summary,
        "emd_results": emd,
        "ref_detected": ref_detected,
        "model_detected": model_detected,
    }


def run_interaction_overlap_evaluation(
    write_path: Union[str, Path],
    model_name: str,
    trip_activity_name: str = "trip",
    generate_plots: bool = True,
    evaluate_test_inference: bool = True,
    evaluate_synthetic: bool = True,
    prob_threshold: float = 0.5,
    test_inference_dir: Optional[Union[str, Path]] = None,
    reference_schedules: Optional[pd.DataFrame] = None,
    model_schedules: Optional[pd.DataFrame] = None,
    reference_interactions_path: Optional[Union[str, Path]] = None,
    model_interactions_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    out_dir = Path(write_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    if evaluate_test_inference:
        if test_inference_dir is None:
            print("[interaction_overlap] Skipping test_inference scope: missing directory.")
        else:
            try:
                test_dir = out_dir / "test_inference"
                print(
                    f"[interaction_overlap] Running test_inference scope from {Path(test_inference_dir)}"
                )
                results["test_inference"] = evaluate_from_test_inference(
                    test_inference_dir=test_inference_dir,
                    write_path=test_dir,
                    label_model=model_name,
                    trip_activity_name=trip_activity_name,
                    prob_threshold=prob_threshold,
                    generate_plots=generate_plots,
                )
            except Exception as e:
                print(f"[interaction_overlap] Skipping test_inference scope: {e}")

    if evaluate_synthetic:
        try:
            if reference_schedules is None or model_schedules is None:
                raise ValueError("reference_schedules/model_schedules are required.")
            if reference_interactions_path is None or model_interactions_path is None:
                raise ValueError("reference_interactions_path/model_interactions_path are required.")

            ref_path = Path(reference_interactions_path)
            mod_path = Path(model_interactions_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference interactions not found: {ref_path}")
            if not mod_path.exists():
                raise FileNotFoundError(f"Model interactions not found: {mod_path}")

            print(f"[interaction_overlap] Running synthetic scope from {mod_path}")
            synthetic_dir = out_dir / "synthetic"
            results["synthetic"] = evaluate_from_interval_csv(
                reference_schedules=reference_schedules,
                model_schedules=model_schedules,
                reference_interactions=pd.read_csv(ref_path),
                model_interactions=pd.read_csv(mod_path),
                write_path=synthetic_dir,
                label_model=model_name,
                trip_activity_name=trip_activity_name,
                generate_plots=generate_plots,
            )
        except Exception as e:
            print(f"[interaction_overlap] Skipping synthetic scope: {e}")

    return results
