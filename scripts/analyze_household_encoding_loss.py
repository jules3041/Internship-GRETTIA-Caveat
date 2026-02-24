import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caveat.encoding.household_interactions_encoder import HouseholdInteractionsEncoder


def _prepare_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    attrs = attributes.copy()
    attrs["pid"] = attrs["pid"].astype(str)
    if "age" in attrs.columns:
        attrs["_age_sort"] = pd.to_numeric(attrs["age"], errors="coerce").fillna(-1.0)
    else:
        attrs["_age_sort"] = 0.0
    if "sex" in attrs.columns:
        attrs["_sex_sort"] = attrs["sex"].astype(str)
    else:
        attrs["_sex_sort"] = ""
    return attrs


def _winner_max_overlap(candidates):
    if not candidates:
        return None

    winner_idx = 0
    winner_overlap = candidates[0][3]
    for i in range(1, len(candidates)):
        ov = candidates[i][3]
        if ov > winner_overlap:
            winner_idx = i
            winner_overlap = ov

    return candidates[winner_idx]


def _resolve_slot_winners(slot_candidates, slot_conflict_mode: str):
    winners = [_winner_max_overlap(cands) for cands in slot_candidates]
    if slot_conflict_mode != "diversity":
        return winners

    resolved = list(winners)
    n_slots = len(slot_candidates)

    def has_adjacent_same(slot_idx: int, act: int) -> bool:
        if (
            slot_idx > 0
            and winners[slot_idx - 1] is not None
            and int(winners[slot_idx - 1][2]) == act
        ):
            return True
        if (
            slot_idx + 1 < n_slots
            and winners[slot_idx + 1] is not None
            and int(winners[slot_idx + 1][2]) == act
        ):
            return True
        return False

    for t, candidates in enumerate(slot_candidates):
        if not candidates or winners[t] is None:
            continue

        acts = sorted({int(c[2]) for c in candidates})
        if len(acts) < 2:
            continue

        winner = winners[t]
        winner_act = int(winner[2])
        winner_has_adj = has_adjacent_same(t, winner_act)

        disappearing_loser_acts = []
        for act in acts:
            if act == winner_act:
                continue
            if not has_adjacent_same(t, act):
                disappearing_loser_acts.append(act)

        if len(disappearing_loser_acts) != 1 or not winner_has_adj:
            continue

        target_act = disappearing_loser_acts[0]
        chosen = None
        for cand in candidates:
            if int(cand[2]) != target_act:
                continue
            if (
                chosen is None
                or float(cand[3]) > float(chosen[3])
                or (float(cand[3]) == float(chosen[3]) and int(cand[0]) < int(chosen[0]))
            ):
                chosen = cand
        if chosen is not None:
            resolved[t] = chosen

    return resolved


def analyze_encoding_loss(
    schedules: pd.DataFrame,
    attributes: pd.DataFrame,
    time_steps: int,
    max_members: int,
    group_col: str,
    slot_conflict_mode: str = "winner",
):
    schedules = schedules.copy()
    schedules["pid"] = schedules["pid"].astype(str)
    schedules["_row_id"] = np.arange(len(schedules), dtype=np.int64)

    attrs = _prepare_attributes(attributes)

    acts = sorted(schedules["act"].astype(str).unique())
    act_to_idx = {"PAD": 0} | {act: idx for idx, act in enumerate(acts, start=1)}

    sched_cols = ["pid", "act", "start", "end", "_row_id"]
    if "schedule_id" in schedules.columns:
        sched_cols.append("schedule_id")
    attr_cols = ["pid", group_col, "_age_sort", "_sex_sort"]

    df = schedules[sched_cols].merge(attrs[attr_cols], on="pid", how="left")
    df["_act_idx"] = df["act"].astype(str).map(act_to_idx).fillna(0).astype(np.uint8)

    # Match the encoder ordering to keep member truncation behavior identical.
    df.sort_values(
        by=[group_col, "_age_sort", "_sex_sort"],
        ascending=[True, False, True],
        inplace=True,
    )

    n_rows = len(schedules)
    reason = np.array(["unprocessed"] * n_rows, dtype=object)
    has_valid_slot = np.zeros(n_rows, dtype=bool)
    kept_slots = np.zeros(n_rows, dtype=np.int32)
    kept_overlap_min = np.zeros(n_rows, dtype=np.float32)

    step_size = 1440.0 / float(time_steps)
    slot_edges = np.arange(time_steps + 1, dtype=np.float32) * step_size

    missing_group_df = df[df[group_col].isna()]
    for rid in missing_group_df["_row_id"].to_numpy(dtype=np.int64):
        reason[rid] = "missing_group"

    grouped = df[df[group_col].notna()].groupby(group_col, sort=False)

    for _, hh_group in grouped:
        pid_groups = hh_group.groupby("pid", sort=False)
        m_idx = 0

        for _, member_df in pid_groups:
            row_ids = member_df["_row_id"].to_numpy(dtype=np.int64)

            if m_idx >= max_members:
                for rid in row_ids:
                    if reason[rid] == "unprocessed":
                        reason[rid] = "dropped_max_members"
                continue

            member_df = member_df.assign(
                _start_num=pd.to_numeric(member_df["start"], errors="coerce"),
                _end_num=pd.to_numeric(member_df["end"], errors="coerce"),
            ).sort_values(by=["_start_num", "_end_num"], kind="stable")

            acts_arr = member_df["_act_idx"].to_numpy(dtype=np.int16)
            starts = member_df["_start_num"].to_numpy(dtype=np.float32)
            ends = member_df["_end_num"].to_numpy(dtype=np.float32)
            row_ids = member_df["_row_id"].to_numpy(dtype=np.int64)
            slot_candidates = [[] for _ in range(time_steps)]

            for seq_idx, (rid, act_idx, start, end) in enumerate(
                zip(row_ids, acts_arr, starts, ends)
            ):
                if np.isnan(start) or np.isnan(end) or int(act_idx) == 0:
                    continue

                s_slot = int(start / step_size)
                e_slot = int((end - 1e-4) / step_size)

                if e_slot < 0 or s_slot >= time_steps:
                    continue
                s_slot = max(0, s_slot)
                e_slot = min(time_steps - 1, e_slot)
                if s_slot > e_slot:
                    continue

                edges = slot_edges[s_slot : e_slot + 2]
                overlap = np.minimum(end, edges[1:]) - np.maximum(start, edges[:-1])
                overlap = np.maximum(overlap, 0.0)
                if not np.any(overlap > 0.0):
                    continue

                has_valid_slot[rid] = True
                for local_idx, ov in enumerate(overlap):
                    if ov <= 0.0:
                        continue
                    t = s_slot + local_idx
                    slot_candidates[t].append((seq_idx, int(rid), int(act_idx), float(ov)))

            resolved_winners = _resolve_slot_winners(
                slot_candidates, slot_conflict_mode=slot_conflict_mode
            )
            for t, winner in enumerate(resolved_winners):
                if winner is None:
                    continue
                rid = winner[1]
                kept_slots[rid] += 1
                kept_overlap_min[rid] += float(winner[3])

            for rid in row_ids:
                if reason[rid] != "unprocessed":
                    continue
                if kept_slots[rid] > 0:
                    reason[rid] = "kept"
                elif has_valid_slot[rid]:
                    reason[rid] = "overshadowed"
                else:
                    reason[rid] = "no_valid_slot"

            m_idx += 1

    reason = np.where(reason == "unprocessed", "unknown", reason)

    rows = schedules.copy()
    rows["reason"] = reason
    rows["kept_slots"] = kept_slots
    rows["kept_overlap_min"] = kept_overlap_min
    rows["kept_slot_min"] = kept_slots.astype(np.float32) * np.float32(step_size)
    rows["clipped_duration_min"] = (
        np.minimum(pd.to_numeric(rows["end"], errors="coerce").fillna(-1e9), 1440.0)
        - np.maximum(pd.to_numeric(rows["start"], errors="coerce").fillna(1e9), 0.0)
    ).clip(lower=0.0)

    by_reason = (
        rows.groupby("reason", dropna=False)
        .agg(
            rows=("pid", "size"),
            unique_pids=("pid", "nunique"),
            clipped_minutes=("clipped_duration_min", "sum"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )

    by_act = (
        rows.groupby("act", dropna=False)
        .agg(
            total_rows=("pid", "size"),
            kept_rows=("reason", lambda s: int((s == "kept").sum())),
            lost_rows=("reason", lambda s: int((s != "kept").sum())),
            clipped_minutes=("clipped_duration_min", "sum"),
            kept_overlap_min=("kept_overlap_min", "sum"),
            kept_slot_min=("kept_slot_min", "sum"),
        )
        .reset_index()
    )
    by_act["lost_rows_pct"] = 100.0 * by_act["lost_rows"] / by_act["total_rows"].clip(lower=1)
    by_act["lost_minutes_pct_overlap"] = 100.0 * (
        1.0 - by_act["kept_overlap_min"] / by_act["clipped_minutes"].clip(lower=1e-9)
    )
    by_act = by_act.sort_values("lost_rows", ascending=False)

    kept_pid_count = int(rows.loc[rows["reason"] == "kept", "pid"].nunique())

    summary = {
        "time_steps": int(time_steps),
        "step_size_min": float(step_size),
        "max_members": int(max_members),
        "group_col": group_col,
        "slot_conflict_mode": slot_conflict_mode,
        "input_rows": int(len(rows)),
        "input_unique_pids": int(rows["pid"].nunique()),
        "kept_rows": int((rows["reason"] == "kept").sum()),
        "lost_rows": int((rows["reason"] != "kept").sum()),
        "lost_rows_pct": float(100.0 * (rows["reason"] != "kept").mean()),
        "kept_unique_pids": kept_pid_count,
        "lost_unique_pids": int(rows["pid"].nunique() - kept_pid_count),
        "input_clipped_minutes": float(rows["clipped_duration_min"].sum()),
        "kept_overlap_minutes": float(rows["kept_overlap_min"].sum()),
        "kept_slot_minutes": float(rows["kept_slot_min"].sum()),
        "lost_minutes_pct_overlap": float(
            100.0
            * (
                1.0
                - rows["kept_overlap_min"].sum()
                / max(rows["clipped_duration_min"].sum(), 1e-9)
            )
        ),
    }

    return summary, by_reason, by_act, rows


def _print_summary(summary: dict, by_reason: pd.DataFrame, by_act: pd.DataFrame) -> None:
    print("\n=== Encoding Loss Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Loss by reason ===")
    print(by_reason.to_string(index=False))

    print("\n=== Top acts by lost rows ===")
    cols = [
        "act",
        "total_rows",
        "kept_rows",
        "lost_rows",
        "lost_rows_pct",
        "clipped_minutes",
        "kept_overlap_min",
        "lost_minutes_pct_overlap",
    ]
    print(by_act[cols].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activity loss caused by household interaction encoding."
    )
    parser.add_argument(
        "--schedules",
        required=True,
        help="Path to caveat_schedules.csv",
    )
    parser.add_argument(
        "--attributes",
        help="Path to caveat_attributes.csv. Defaults to sibling of schedules.",
    )
    parser.add_argument(
        "--interactions",
        default=None,
        help="Optional path to caveat_interactions.csv",
    )
    parser.add_argument("--time-steps", type=int, default=144)
    parser.add_argument("--max-members", type=int, default=5)
    parser.add_argument("--group-col", default="hid")
    parser.add_argument(
        "--slot-conflict-mode",
        default="winner",
        choices=["winner", "diversity"],
        help="Winner: max overlap per slot. Diversity: prioritize middle activity in 1-2-1 patterns.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="If set, writes <prefix>_summary.json, <prefix>_by_reason.csv, <prefix>_by_act.csv, <prefix>_rows.csv",
    )
    args = parser.parse_args()

    schedules_path = Path(args.schedules)
    if args.attributes:
        attributes_path = Path(args.attributes)
    else:
        attributes_path = schedules_path.with_name("caveat_attributes.csv")

    schedules = pd.read_csv(schedules_path)
    attributes = pd.read_csv(attributes_path)

    # Build the real dataset through HouseholdInteractionsEncoder to ensure parity with runtime pipeline.
    encoder = HouseholdInteractionsEncoder(
        time_steps=args.time_steps,
        dataset_params={
            "max_members": args.max_members,
            "group_by": args.group_col,
            "label_cols": [],
            "slot_conflict_mode": args.slot_conflict_mode,
        },
        attributes_path=str(attributes_path),
        interactions_path=args.interactions,
    )
    dataset = encoder.encode(schedules=schedules)

    summary, by_reason, by_act, rows = analyze_encoding_loss(
        schedules=schedules,
        attributes=attributes,
        time_steps=args.time_steps,
        max_members=args.max_members,
        group_col=args.group_col,
        slot_conflict_mode=args.slot_conflict_mode,
    )

    encoded_non_pad_slots = int((dataset.x_tensor > 0).sum().item())
    tracked_kept_slots = int(rows["kept_slots"].sum())
    summary["encoded_non_pad_slots"] = encoded_non_pad_slots
    summary["tracked_kept_slots"] = tracked_kept_slots
    summary["slot_tracking_gap"] = int(encoded_non_pad_slots - tracked_kept_slots)

    _print_summary(summary, by_reason, by_act)

    if args.output_prefix:
        out_prefix = Path(args.output_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        by_reason.to_csv(f"{out_prefix}_by_reason.csv", index=False)
        by_act.to_csv(f"{out_prefix}_by_act.csv", index=False)
        rows.to_csv(f"{out_prefix}_rows.csv", index=False)
        print(f"\nOutputs written with prefix: {out_prefix}")


if __name__ == "__main__":
    main()
