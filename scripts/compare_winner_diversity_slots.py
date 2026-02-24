import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from caveat.encoding.household_interactions_encoder import HouseholdInteractionsEncoder


def _build_dataset(
    schedules: pd.DataFrame,
    attributes_path: Path,
    interactions_path: Optional[str],
    time_steps: int,
    max_members: int,
    group_col: str,
    slot_conflict_mode: str,
):
    encoder = HouseholdInteractionsEncoder(
        time_steps=time_steps,
        dataset_params={
            "max_members": max_members,
            "group_by": group_col,
            "label_cols": [],
            "slot_conflict_mode": slot_conflict_mode,
        },
        attributes_path=str(attributes_path),
        interactions_path=interactions_path,
    )
    return encoder.encode(schedules=schedules)


def _idx_to_name(idx_to_act: dict, idx: int) -> str:
    return str(idx_to_act.get(int(idx), f"act_{int(idx)}"))


def compare_winner_vs_diversity(
    winner_ds,
    diversity_ds,
):
    if winner_ds.x_tensor.shape != diversity_ds.x_tensor.shape:
        raise ValueError(
            "x_tensor shape mismatch between winner and diversity datasets: "
            f"{tuple(winner_ds.x_tensor.shape)} vs {tuple(diversity_ds.x_tensor.shape)}"
        )
    if winner_ds.mask_tensor.shape != diversity_ds.mask_tensor.shape:
        raise ValueError(
            "mask_tensor shape mismatch between winner and diversity datasets: "
            f"{tuple(winner_ds.mask_tensor.shape)} vs {tuple(diversity_ds.mask_tensor.shape)}"
        )

    winner_x = winner_ds.x_tensor.cpu().numpy().astype(np.int64)
    diversity_x = diversity_ds.x_tensor.cpu().numpy().astype(np.int64)

    active_members = winner_ds.mask_tensor.cpu().numpy().astype(bool)
    active_slots_mask = np.repeat(active_members[:, :, None], winner_x.shape[2], axis=2)
    changed_slots_mask = (winner_x != diversity_x) & active_slots_mask

    total_active_slots = int(active_slots_mask.sum())
    changed_slots = int(changed_slots_mask.sum())
    changed_slots_pct = (
        100.0 * changed_slots / total_active_slots if total_active_slots > 0 else 0.0
    )

    changed_by_member = changed_slots_mask.sum(axis=2).astype(np.int64)
    members_with_changes = int(((changed_by_member > 0) & active_members).sum())
    active_members_count = int(active_members.sum())
    member_changed_pct = (
        100.0 * members_with_changes / active_members_count
        if active_members_count > 0
        else 0.0
    )

    households_with_changes = int(changed_slots_mask.any(axis=(1, 2)).sum())
    households_total = int(winner_x.shape[0])
    household_changed_pct = (
        100.0 * households_with_changes / households_total if households_total > 0 else 0.0
    )

    winner_changed = winner_x[changed_slots_mask]
    diversity_changed = diversity_x[changed_slots_mask]

    idx_to_act = dict(winner_ds.idx_to_act)
    max_idx_candidates = [0]
    if idx_to_act:
        max_idx_candidates.append(max(int(k) for k in idx_to_act.keys()))
    if winner_x.size > 0:
        max_idx_candidates.append(int(winner_x.max()))
    if diversity_x.size > 0:
        max_idx_candidates.append(int(diversity_x.max()))
    max_idx = int(max(max_idx_candidates))
    n_acts = max_idx + 1

    winner_all_counts = np.bincount(
        winner_x[active_slots_mask].astype(np.int64), minlength=n_acts
    ).astype(np.int64)
    diversity_all_counts = np.bincount(
        diversity_x[active_slots_mask].astype(np.int64), minlength=n_acts
    ).astype(np.int64)
    winner_changed_counts = np.bincount(
        winner_changed.astype(np.int64), minlength=n_acts
    ).astype(np.int64)
    diversity_changed_counts = np.bincount(
        diversity_changed.astype(np.int64), minlength=n_acts
    ).astype(np.int64)

    rows = []
    for idx in range(n_acts):
        rows.append(
            {
                "act_idx": idx,
                "act_name": _idx_to_name(idx_to_act, idx),
                "winner_slots": int(winner_all_counts[idx]),
                "diversity_slots": int(diversity_all_counts[idx]),
                "delta_slots_div_minus_winner": int(
                    diversity_all_counts[idx] - winner_all_counts[idx]
                ),
                "winner_slots_in_changed_positions": int(winner_changed_counts[idx]),
                "diversity_slots_in_changed_positions": int(diversity_changed_counts[idx]),
            }
        )
    activity_delta_df = pd.DataFrame(rows).sort_values(
        by="delta_slots_div_minus_winner", ascending=False
    )

    transition_counter = Counter(
        zip(winner_changed.astype(int).tolist(), diversity_changed.astype(int).tolist())
    )
    transition_rows = []
    for (w_idx, d_idx), count in transition_counter.items():
        transition_rows.append(
            {
                "winner_act_idx": int(w_idx),
                "winner_act_name": _idx_to_name(idx_to_act, int(w_idx)),
                "diversity_act_idx": int(d_idx),
                "diversity_act_name": _idx_to_name(idx_to_act, int(d_idx)),
                "changed_slots": int(count),
            }
        )
    transitions_df = pd.DataFrame(transition_rows)
    if not transitions_df.empty:
        transitions_df = transitions_df.sort_values(by="changed_slots", ascending=False)

    member_rows = []
    for hh_idx in range(winner_x.shape[0]):
        for m_idx in range(winner_x.shape[1]):
            if not active_members[hh_idx, m_idx]:
                continue
            slots_changed = int(changed_by_member[hh_idx, m_idx])
            member_rows.append(
                {
                    "household_idx": int(hh_idx),
                    "member_idx": int(m_idx),
                    "changed_slots": slots_changed,
                    "changed_minutes": float(slots_changed * (1440.0 / winner_x.shape[2])),
                }
            )
    member_changes_df = pd.DataFrame(member_rows)
    if not member_changes_df.empty:
        member_changes_df = member_changes_df.sort_values(
            by="changed_slots", ascending=False
        )

    distinct_acts_winner = set(winner_changed.astype(int).tolist())
    distinct_acts_diversity = set(diversity_changed.astype(int).tolist())
    distinct_acts_involved = sorted(distinct_acts_winner | distinct_acts_diversity)

    summary = {
        "time_steps": int(winner_x.shape[2]),
        "step_size_min": float(1440.0 / winner_x.shape[2]),
        "households_total": households_total,
        "active_members_total": active_members_count,
        "active_slots_total": total_active_slots,
        "changed_slots_total": changed_slots,
        "changed_slots_pct_of_active": float(changed_slots_pct),
        "members_with_changes": members_with_changes,
        "members_with_changes_pct": float(member_changed_pct),
        "households_with_changes": households_with_changes,
        "households_with_changes_pct": float(household_changed_pct),
        "distinct_activity_labels_in_changed_slots": int(len(distinct_acts_involved)),
        "distinct_activity_labels_in_changed_slots_names": [
            _idx_to_name(idx_to_act, idx) for idx in distinct_acts_involved
        ],
    }

    return summary, activity_delta_df, transitions_df, member_changes_df


def _print_report(
    summary: dict,
    activity_delta_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
) -> None:
    print("\n=== Winner vs Diversity Comparison ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Top activity deltas (diversity - winner) ===")
    top_delta = activity_delta_df[
        activity_delta_df["delta_slots_div_minus_winner"] != 0
    ].copy()
    if top_delta.empty:
        print("No net activity delta.")
    else:
        cols = [
            "act_idx",
            "act_name",
            "winner_slots",
            "diversity_slots",
            "delta_slots_div_minus_winner",
        ]
        print(top_delta[cols].head(20).to_string(index=False))

    print("\n=== Top slot transitions (winner -> diversity) ===")
    if transitions_df.empty:
        print("No slot change between modes.")
    else:
        cols = [
            "winner_act_idx",
            "winner_act_name",
            "diversity_act_idx",
            "diversity_act_name",
            "changed_slots",
        ]
        print(transitions_df[cols].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare slot-level differences between slot_conflict_mode=winner "
            "and slot_conflict_mode=diversity."
        )
    )
    parser.add_argument("--schedules", required=True, help="Path to caveat_schedules.csv")
    parser.add_argument(
        "--attributes",
        default=None,
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
        "--output-prefix",
        default=None,
        help=(
            "If set, writes <prefix>_summary.json, <prefix>_activity_slot_deltas.csv, "
            "<prefix>_slot_transitions.csv, <prefix>_member_slot_changes.csv"
        ),
    )
    args = parser.parse_args()

    schedules_path = Path(args.schedules)
    if args.attributes:
        attributes_path = Path(args.attributes)
    else:
        attributes_path = schedules_path.with_name("caveat_attributes.csv")

    schedules = pd.read_csv(schedules_path)

    winner_ds = _build_dataset(
        schedules=schedules,
        attributes_path=attributes_path,
        interactions_path=args.interactions,
        time_steps=args.time_steps,
        max_members=args.max_members,
        group_col=args.group_col,
        slot_conflict_mode="winner",
    )
    diversity_ds = _build_dataset(
        schedules=schedules,
        attributes_path=attributes_path,
        interactions_path=args.interactions,
        time_steps=args.time_steps,
        max_members=args.max_members,
        group_col=args.group_col,
        slot_conflict_mode="diversity",
    )

    summary, activity_delta_df, transitions_df, member_changes_df = compare_winner_vs_diversity(
        winner_ds=winner_ds,
        diversity_ds=diversity_ds,
    )
    _print_report(summary, activity_delta_df, transitions_df)

    if args.output_prefix:
        out_prefix = Path(args.output_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        activity_delta_df.to_csv(f"{out_prefix}_activity_slot_deltas.csv", index=False)
        transitions_df.to_csv(f"{out_prefix}_slot_transitions.csv", index=False)
        member_changes_df.to_csv(f"{out_prefix}_member_slot_changes.csv", index=False)
        print(f"\nOutputs written with prefix: {out_prefix}")


if __name__ == "__main__":
    main()
