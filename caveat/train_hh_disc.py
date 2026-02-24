"""
script d'entraînement pour les modèles household discrets (HouseholdDiscCNNVAE et variantes).

gère l'entraînement, l'inférence, la génération et l'évaluation complète.
chaque membre du ménage a sa propre représentation latente 2D.
c'est le point d'entrée principal pour tous les modèles household discrets.
"""

import argparse
import yaml
import os
import math
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from tqdm import tqdm
from caveat import runners
from caveat.models import library
from caveat.models.discrete.household_discCNN_VAE import HouseholdDiscCNNVAE
from caveat.models.discrete.household_discCNN_VAE_enhanced import HouseholdDiscCNNVAEEnhanced
from caveat.models.discrete.household_discCNN_CVAE import HouseholdDiscCNNCVAE
from caveat.models.discrete.household_discCNN_CVAE_attention import HouseholdDiscCNNCVAEAttention
from caveat.models.discrete.household_discCNN_CVAE_att_interactions import HouseholdDiscCNNCVAEAttentionInteractions
from caveat.models.discrete.household_discCNN_CVAE_fullattention_interactions import (
    HouseholdDiscCNNCVAEFullAttentionInteractions,
)
from caveat.evaluate.baseline_evaluator import (
    ThemeEvaluator,
    Visualizer,
    prepare_eval_df_from_minutes,
    ConditionalMetrics,
    ConditionalVisualizer
)
from caveat.evaluate.joint_trip_evaluator import JointTripEvaluator, run_joint_trip_evaluation
from caveat.evaluate.interaction_overlap_evaluator import run_interaction_overlap_evaluation
import polars as pl
import pandas as pd


def _resolve_latent_path(experiment_version_dir, latent_filename=None):
    test_infer_dir = Path(experiment_version_dir) / "test_inference"
    if latent_filename:
        return test_infer_dir / latent_filename

    mus_path = test_infer_dir / "mus.csv"
    if mus_path.exists():
        return mus_path

    print("Warning: 'mus.csv' not found, using 'zs.csv' (sampled latent).")
    return test_infer_dir / "zs.csv"


def _resolve_hierarchical_latent_files(experiment_version_dir, coord_file, indiv_file):
    test_infer_dir = Path(experiment_version_dir) / "test_inference"

    def _first_existing(candidates):
        for name in candidates:
            if (test_infer_dir / name).exists():
                return name
        return candidates[-1]

    coord_candidates = [coord_file, "mus_coord.csv", "mus.csv", "zs.csv"]
    indiv_candidates = [indiv_file, "mus.csv", "zs.csv"]

    coord_resolved = _first_existing(coord_candidates)
    indiv_resolved = _first_existing(indiv_candidates)

    if coord_resolved == indiv_resolved:
        print(
            f"Warning: coord/indiv latent files both resolve to '{coord_resolved}'. "
            "Plots may look identical."
        )

    print(
        f"Latent files resolved -> coord: {coord_resolved}, indiv: {indiv_resolved}"
    )
    return coord_resolved, indiv_resolved


def _load_mask(experiment_version_dir):
    mask_path = Path(experiment_version_dir) / "test_inference" / "mask.csv"
    if not mask_path.exists():
        return None
    return pd.read_csv(mask_path, header=None).values


def _prepare_latent_points_for_umap(latent_data, mask, point_mode="per_member"):
    """
    point_mode:
    - "per_member" : un point par membre actif (défaut) — donne plus de résolution dans l'espace latent
    - "household" : un point par ménage (moyenne maskée si possible) — utile pour voir la structure ménage
    """
    if mask is None:
        return latent_data, "raw"

    if mask.shape[0] != latent_data.shape[0]:
        min_len = min(mask.shape[0], latent_data.shape[0])
        print(
            f"Warning: mask/latent length mismatch ({mask.shape[0]} vs {latent_data.shape[0]}). "
            f"Truncating to {min_len}."
        )
        mask = mask[:min_len]
        latent_data = latent_data[:min_len]

    if latent_data.shape[1] % mask.shape[1] != 0:
        return latent_data, "raw"

    members = mask.shape[1]
    latent_dim = latent_data.shape[1] // members
    latent_3d = latent_data.reshape(-1, members, latent_dim)

    if point_mode == "household":
        mask_3d = mask[:, :, None].astype(float)
        denom = mask_3d.sum(axis=1)
        denom[denom == 0] = 1.0
        return (latent_3d * mask_3d).sum(axis=1) / denom, "household_mean"

    mask_bool = mask.astype(bool)
    if mask_bool.shape[:2] == latent_3d.shape[:2]:
        points = latent_3d[mask_bool].reshape(-1, latent_dim)
    else:
        points = latent_3d.reshape(-1, latent_dim)
    return points, "per_member"


def visualize_latent_umap(
    experiment_version_dir,
    model_name,
    attributes_df=None,
    latent_filename=None,
    latent_label=None,
    output_filename=None,
    point_mode="per_member",
):
    """génère la projection UMAP à partir d'un fichier csv de latents pour visualiser l'espace latent."""
    import umap

    data_path = _resolve_latent_path(
        experiment_version_dir, latent_filename=latent_filename
    )
    if not data_path.exists():
        print(f"Warning: latent file not found at {data_path}")
        return

    default_title = (
        "Latent Means - Mu" if data_path.name == "mus.csv" else "Sampled Z - Noisy"
    )
    title_suffix = latent_label or default_title
    default_output_filename = (
        "latent_means_umap.png"
        if data_path.name == "mus.csv"
        else "latent_samples_umap.png"
    )
    output_filename = output_filename or default_output_filename

    print(f"--- Generating UMAP for {title_suffix} from {data_path.name} ---")
    latent_data = pd.read_csv(data_path, header=None).values

    mask = _load_mask(experiment_version_dir)
    latent_data, projection_mode = _prepare_latent_points_for_umap(
        latent_data, mask, point_mode=point_mode
    )
    if projection_mode == "household_mean":
        title_suffix = f"{title_suffix} (masked household mean)"
    elif projection_mode == "per_member":
        title_suffix = f"{title_suffix} (per member)"

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_data)

    plt.figure(figsize=(12, 10))
    hue = None
    if attributes_df is not None and len(attributes_df.columns) > 0:
        target_col = (
            "household_size"
            if "household_size" in attributes_df.columns
            else attributes_df.columns[0]
        )
        hue = attributes_df[target_col].iloc[: len(embedding)]

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=hue,
        palette="viridis",
        s=15,
        alpha=0.8,
        edgecolor="none",
    )

    plt.title(f"UMAP projection of {title_suffix} - {model_name}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    out_dir = Path(experiment_version_dir) / "baseline_evaluation"
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / output_filename, dpi=300)
    plt.close()
    print(f"Saved UMAP visualization to: {out_dir / output_filename}")


def _load_latent_per_member(
    experiment_version_dir,
    latent_filename="mus.csv",
    max_members=None,
):
    data_path = Path(experiment_version_dir) / "test_inference" / latent_filename
    if not data_path.exists():
        print(f"Warning: latent file not found at {data_path}")
        return None, None

    latents = pd.read_csv(data_path, header=None).values
    mask = _load_mask(experiment_version_dir)
    mask_bool = mask.astype(bool) if mask is not None else None

    members = None
    if mask_bool is not None:
        members = mask_bool.shape[1]
    elif max_members is not None:
        members = int(max_members)

    if members is not None and latents.shape[1] % members == 0:
        latent_dim = latents.shape[1] // members
        latents = latents.reshape(-1, members, latent_dim)
        if mask_bool is not None:
            latents = latents[mask_bool].reshape(-1, latent_dim)
        else:
            latents = latents.reshape(-1, latent_dim)
    else:
        latent_dim = latents.shape[1]

    if latents.size == 0:
        print(f"Warning: {latent_filename} empty after masking.")
        return None, None

    return latents, latent_dim


def _load_mus_per_member(experiment_version_dir, max_members=None):
    return _load_latent_per_member(
        experiment_version_dir, latent_filename="mus.csv", max_members=max_members
    )


def plot_latent_diagnostics(
    experiment_version_dir,
    model_name,
    latent_filename="mus.csv",
    latent_label="mu",
    output_prefix="mu",
    max_members=None,
    dims_per_fig=24,
    cols=4,
):
    latents, latent_dim = _load_latent_per_member(
        experiment_version_dir,
        latent_filename=latent_filename,
        max_members=max_members,
    )
    if latents is None:
        return

    out_dir = Path(experiment_version_dir) / "baseline_evaluation"
    out_dir.mkdir(exist_ok=True, parents=True)

    for page, start in enumerate(range(0, latent_dim, dims_per_fig), 1):
        dims = list(range(start, min(start + dims_per_fig, latent_dim)))
        rows = math.ceil(len(dims) / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 3), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")

        for i, d in enumerate(dims):
            ax = axes.flat[i]
            x = latents[:, d]
            sns.histplot(x, bins=40, stat="density", kde=True, ax=ax, color="#4C72B0")
            lo, hi = np.percentile(x, [0.5, 99.5])
            grid = np.linspace(lo, hi, 200)
            ax.plot(grid, stats.norm.pdf(grid, 0, 1), color="black", lw=1, alpha=0.6)
            ax.set_title(f"{latent_label}[{d}]")
            ax.axis("on")

        fig.suptitle(f"{latent_label} densities (page {page}) - {model_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{output_prefix}_density_page_{page}.png", dpi=300)
        plt.close(fig)

    means = latents.mean(axis=0)
    stds = latents.std(axis=0)
    plt.figure(figsize=(7, 6))
    plt.scatter(means, stds, alpha=0.8)
    plt.axvline(0.0, color="black", lw=1, alpha=0.6)
    plt.axhline(1.0, color="black", lw=1, alpha=0.6)
    plt.xlabel(f"mean({latent_label})")
    plt.ylabel(f"std({latent_label})")
    plt.title(f"Mean/Std per dim ({latent_label}) - {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{output_prefix}_mean_std.png", dpi=300)
    plt.close()

    if latent_dim >= 2:
        corr = np.corrcoef(latents, rowvar=False)
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title(f"Correlation ({latent_label}) - {model_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{output_prefix}_corr_heatmap.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        sns.kdeplot(x=latents[:, 0], y=latents[:, 1], fill=True, cmap="Blues")
        plt.title(f"{latent_label}[0] vs {latent_label}[1] - {model_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{output_prefix}_2d_kde.png", dpi=300)
        plt.close()


def plot_mu_diagnostics(
    experiment_version_dir,
    model_name,
    max_members=None,
    dims_per_fig=24,
    cols=4,
):
    plot_latent_diagnostics(
        experiment_version_dir=experiment_version_dir,
        model_name=model_name,
        latent_filename="mus.csv",
        latent_label="mu",
        output_prefix="mu",
        max_members=max_members,
        dims_per_fig=dims_per_fig,
        cols=cols,
    )


def _select_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["pid", "act", "start", "end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required schedule columns: {missing}")
    return df[required].copy()


def split_reference_train_test(
    reference_schedules: pd.DataFrame,
    test_rate: float = 0.1,
    seed: int = 12345,
):
    df = _select_schedule_columns(reference_schedules)
    if df.empty:
        return df.copy(), df.copy()

    unique_pids = df["pid"].astype(str).unique()
    if len(unique_pids) < 2:
        return df.copy(), df.iloc[0:0].copy()

    test_rate = float(test_rate)
    test_rate = min(max(test_rate, 0.0), 1.0)
    n_test = int(len(unique_pids) * test_rate)
    n_test = max(1, min(n_test, len(unique_pids) - 1))

    shuffled = unique_pids.copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)

    test_pids = set(shuffled[:n_test])
    test_data = df[df["pid"].astype(str).isin(test_pids)].copy()
    train_data = df[~df["pid"].astype(str).isin(test_pids)].copy()
    return train_data, test_data


def compute_binary_codes(dataset: pd.DataFrame, t_min: int, t_max: int, activities: list):
    """
    génère des matrices binaires d'activité pour un ensemble de trajectoires.

    chaque cellule vaut 1 si l'activité est présente à ce pas de temps, 0 sinon.
    c'est utilisé pour le calcul de nouveauté/généralisation (est-ce que le modèle
    génère des séquences qu'il n'a jamais vues pendant l'entraînement).

    Returns:
        np.ndarray: matrice binaire de shape (n_trajectoires, n_pas_de_temps, n_activités).
    """
    df = _select_schedule_columns(dataset)
    if df.empty:
        n_steps = max(int(t_max) - int(t_min) + 1, 1)
        return np.zeros((0, n_steps, len(activities)), dtype=np.uint8)

    dict_activity = {a: i for i, a in enumerate(activities)}
    df["act_code"] = df["act"].map(dict_activity)
    unknown_count = int(df["act_code"].isna().sum())
    if unknown_count > 0:
        print(
            f"Warning: {unknown_count} rows contain unseen activities; "
            "they are ignored for binary coding."
        )
        df = df[df["act_code"].notna()].copy()
    if df.empty:
        n_steps = max(int(t_max) - int(t_min) + 1, 1)
        return np.zeros((0, n_steps, len(activities)), dtype=np.uint8)

    n_steps = max(int(t_max) - int(t_min) + 1, 1)
    df["act_code"] = df["act_code"].astype(int)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    if df.empty:
        return np.zeros((0, n_steps, len(activities)), dtype=np.uint8)

    df["start"] = df["start"].astype(int) - int(t_min)
    df["end"] = df["end"].astype(int) - int(t_min)
    df["start"] = df["start"].clip(lower=0, upper=n_steps - 1)
    df["end"] = df["end"].clip(lower=0, upper=n_steps - 1)

    trajectories_dataset = [group for _, group in df.groupby("pid", sort=False)]
    binary_codes = np.zeros((len(trajectories_dataset), n_steps, len(activities)), dtype=np.uint8)

    for i, trajectory in enumerate(trajectories_dataset):
        for idx in trajectory.index:
            id_activity = int(trajectory.loc[idx, "act_code"])
            t_start = int(trajectory.loc[idx, "start"])
            t_end = int(trajectory.loc[idx, "end"])
            if t_end < t_start:
                continue
            binary_codes[i, t_start:t_end + 1, id_activity] = 1

    return binary_codes


def flat_binary_codes(binary_codes):
    return binary_codes.reshape(len(binary_codes), -1)


def distance_to_closest_record_score_closest_set(
    data_trajectory_training: pd.DataFrame,
    data_trajectory_testing: pd.DataFrame,
    data_trajectory_generated: pd.DataFrame,
    batch_size: int = 100,
):
    n_gen = len(data_trajectory_generated["pid"].astype(str).unique())
    n_train = len(data_trajectory_training["pid"].astype(str).unique())
    n_test = len(data_trajectory_testing["pid"].astype(str).unique())
    if n_gen == 0 or n_train == 0 or n_test == 0:
        return np.nan

    train_start = pd.to_numeric(data_trajectory_training["start"], errors="coerce").dropna()
    train_end = pd.to_numeric(data_trajectory_training["end"], errors="coerce").dropna()
    if train_start.empty or train_end.empty:
        return np.nan
    t_min = int(train_start.min())
    t_max = int(train_end.max())

    activities = list(pd.Series(data_trajectory_training["act"]).dropna().unique())
    if not activities:
        return np.nan

    binary_codes_training = compute_binary_codes(
        data_trajectory_training, t_min, t_max, activities
    )
    binary_codes_testing = compute_binary_codes(
        data_trajectory_testing, t_min, t_max, activities
    )
    binary_codes_generated = compute_binary_codes(
        data_trajectory_generated, t_min, t_max, activities
    )

    if (
        len(binary_codes_training) == 0
        or len(binary_codes_testing) == 0
        or len(binary_codes_generated) == 0
    ):
        return np.nan

    binary_codes_training_flat = flat_binary_codes(binary_codes_training).astype(np.float32, copy=False)
    binary_codes_testing_flat = flat_binary_codes(binary_codes_testing).astype(np.float32, copy=False)
    binary_codes_generated_flat = flat_binary_codes(binary_codes_generated).astype(np.float32, copy=False)

    n_steps = max(int(t_max) - int(t_min) + 1, 1)
    batch_size = max(1, int(batch_size))
    n_batch = int(np.ceil(n_gen / batch_size))
    res = 0.0

    for batch in tqdm(range(n_batch), desc="Closest-record metric"):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        binary_codes_generated_flat_batch = binary_codes_generated_flat[start_idx:end_idx]
        if len(binary_codes_generated_flat_batch) == 0:
            continue

        dot_products_training_generated = np.dot(
            binary_codes_training_flat, binary_codes_generated_flat_batch.T
        )
        dot_products_testing_generated = np.dot(
            binary_codes_testing_flat, binary_codes_generated_flat_batch.T
        )

        distances_training_generated = 1.0 - (dot_products_training_generated / n_steps)
        distances_testing_generated = 1.0 - (dot_products_testing_generated / n_steps)

        min_distance_generated_sample_training_set = np.min(distances_training_generated, axis=0)
        min_distance_generated_sample_testing_set = np.min(distances_testing_generated, axis=0)

        for i in range(len(binary_codes_generated_flat_batch)):
            dist_min_training = min_distance_generated_sample_training_set[i]
            dist_min_testing = min_distance_generated_sample_testing_set[i]
            if dist_min_testing < dist_min_training:
                res += 1.0
            elif dist_min_training == dist_min_testing:
                n_occur_training = np.sum(dist_min_training == distances_training_generated[:, i])
                n_occur_testing = np.sum(dist_min_testing == distances_testing_generated[:, i])
                denom = n_occur_testing + n_occur_training
                if denom > 0:
                    res += n_occur_testing / denom

    score = 1.0 - (res / n_gen) * ((n_train + n_test) / n_test)
    score = max(0.0, float(score))
    print(f"Distance closest record score: {score}")
    return score


def generated_sequences_activities(dataset: pd.DataFrame):
    df = _select_schedule_columns(dataset)
    if df.empty:
        return np.array([])

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.sort_values(by=["pid", "start", "end"], kind="stable")

    res = []
    for _, traj in df.groupby("pid", sort=False):
        res.append("!".join(traj["act"].astype(str).to_numpy()))
    return np.array(res)


def unseen_sequences(
    data_trajectory_training: pd.DataFrame,
    data_trajectory_testing: pd.DataFrame,
    data_trajectory_generated: pd.DataFrame,
):
    sequences_training = generated_sequences_activities(data_trajectory_training)
    sequences_testing = generated_sequences_activities(data_trajectory_testing)
    sequences_generated = generated_sequences_activities(data_trajectory_generated)

    set_training = set(sequences_training)
    set_testing = set(sequences_testing)
    set_generated = set(sequences_generated)

    unique_test_unseen_train = set_testing.difference(set_training)
    unique_test_unseen_train_in_generated = unique_test_unseen_train.intersection(set_generated)
    unique_generated_unseen_train = set_generated.difference(set_training)
    unique_generated_unseen_all = set_generated.difference(set_testing.union(set_training))

    print(
        "Number of unique sequences in testing set, unseen in training set: "
        f"{len(unique_test_unseen_train)}"
    )
    print(
        f"{len(unique_test_unseen_train_in_generated)} of them are in the generated data"
    )
    print(
        "Number of unique sequences in generated set, unseen in training set: "
        f"{len(unique_generated_unseen_train)}"
    )

    rate_true_new_unique_sequences = np.nan
    if len(unique_generated_unseen_train):
        rate_true_new_unique_sequences = (
            len(unique_test_unseen_train_in_generated) / len(unique_generated_unseen_train)
        )
        print(
            "Rate of unique true newly generated sequences, seen in testing and generated "
            f"but unseen in training set, among unseen in generated: "
            f"{rate_true_new_unique_sequences}"
        )

    proportion_unseen_in_data = np.nan
    if len(sequences_generated):
        proportion_unseen_in_data = float(
            np.sum(
                np.isin(
                    sequences_generated,
                    np.array(list(unique_generated_unseen_all)),
                )
            )
            / len(sequences_generated)
        )
    print(
        "Proportion of sequences that are unseen in the data (test+train) "
        f"{proportion_unseen_in_data}"
    )

    return {
        "unique_test_unseen_train": len(unique_test_unseen_train),
        "unique_test_unseen_train_in_generated": len(unique_test_unseen_train_in_generated),
        "unique_generated_unseen_train": len(unique_generated_unseen_train),
        "rate_true_new_unique_sequences": rate_true_new_unique_sequences,
        "proportion_generated_sequences_unseen_in_train_test": proportion_unseen_in_data,
    }


def run_novelty_evaluation(
    observed_schedules: pd.DataFrame,
    synthetic_schedules: pd.DataFrame,
    write_path: Path,
    test_rate: float = 0.1,
    seed: int = 12345,
    batch_size: int = 100,
):
    write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)

    ref_df = _select_schedule_columns(observed_schedules)
    gen_df = _select_schedule_columns(synthetic_schedules)
    train_df, test_df = split_reference_train_test(
        ref_df, test_rate=test_rate, seed=seed
    )

    print(
        f"Novelty split -> train pids: {train_df['pid'].nunique()}, "
        f"test pids: {test_df['pid'].nunique()}, generated pids: {gen_df['pid'].nunique()}"
    )

    closest_score = distance_to_closest_record_score_closest_set(
        data_trajectory_training=train_df,
        data_trajectory_testing=test_df,
        data_trajectory_generated=gen_df,
        batch_size=batch_size,
    )

    unseen_metrics = unseen_sequences(
        data_trajectory_training=train_df,
        data_trajectory_testing=test_df,
        data_trajectory_generated=gen_df,
    )

    summary = {
        "distance_closest_record_score": closest_score,
        **unseen_metrics,
        "test_rate": float(test_rate),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "n_train_pids": int(train_df["pid"].nunique()),
        "n_test_pids": int(test_df["pid"].nunique()),
        "n_generated_pids": int(gen_df["pid"].nunique()),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = write_path / "novelty_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved novelty metrics to: {summary_path}")
    return summary_df


def _ensure_hid_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hid" in out.columns:
        out["hid"] = pd.to_numeric(out["hid"], errors="coerce")
        return out
    if "pid" not in out.columns:
        return out
    extracted = out["pid"].astype(str).str.extract(r"HH(?P<hid>\d+)_M(?P<member>\d+)")
    if extracted["hid"].isna().all():
        return out
    out["hid"] = pd.to_numeric(extracted["hid"], errors="coerce")
    return out


def _prepare_interaction_events(
    df_interactions: pd.DataFrame,
    df_schedules: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["hid", "act_norm", "overlap_start", "overlap_end", "pid"]
    if df_interactions is None or df_interactions.empty:
        return pd.DataFrame(columns=cols)

    dfi = df_interactions.copy()
    if "interaction_bin" in dfi.columns:
        dfi = dfi[dfi["interaction_bin"].fillna(0).astype(int) > 0].copy()

    schedules = _ensure_hid_column(df_schedules)
    meta_cols = ["schedule_id", "pid", "hid", "act", "start", "end"]
    sched_meta = schedules[[c for c in meta_cols if c in schedules.columns]].copy()
    if {"schedule_id", "pid"}.issubset(dfi.columns) and {"schedule_id", "pid"}.issubset(
        sched_meta.columns
    ):
        dfi = dfi.merge(sched_meta, on=["schedule_id", "pid"], how="left")

    if "hid" not in dfi.columns:
        dfi = _ensure_hid_column(dfi)
    dfi["hid"] = pd.to_numeric(dfi.get("hid"), errors="coerce")

    if "overlap_start" not in dfi.columns and "start" in dfi.columns:
        dfi["overlap_start"] = dfi["start"]
    if "overlap_end" not in dfi.columns and "end" in dfi.columns:
        dfi["overlap_end"] = dfi["end"]
    dfi["overlap_start"] = pd.to_numeric(dfi.get("overlap_start"), errors="coerce")
    dfi["overlap_end"] = pd.to_numeric(dfi.get("overlap_end"), errors="coerce")

    if "act" in dfi.columns:
        dfi["act_norm"] = dfi["act"].astype(str).str.lower().str.strip()
    else:
        dfi["act_norm"] = "unknown"

    if "pid" in dfi.columns:
        dfi["pid"] = dfi["pid"].astype(str)
    else:
        dfi["pid"] = ""
    dfi = dfi[
        dfi["hid"].notna()
        & dfi["pid"].ne("")
        & dfi["overlap_start"].notna()
        & dfi["overlap_end"].notna()
        & (dfi["overlap_end"] > dfi["overlap_start"])
    ].copy()

    # on arrondit les bornes pour éviter les soucis de comparaison float dans le groupby
    dfi["overlap_start"] = dfi["overlap_start"].round(6)
    dfi["overlap_end"] = dfi["overlap_end"].round(6)
    dfi["hid"] = dfi["hid"].astype(int)
    dfi = dfi[cols].drop_duplicates()
    return dfi


def _compute_theme_interaction_rows(
    reference_schedules: pd.DataFrame,
    model_schedules: pd.DataFrame,
    reference_interactions_path: Path,
    model_interactions_path: Path,
    model_name: str,
) -> pd.DataFrame:
    out_cols = ["Theme", "Metric", "Label", "Score"]
    if not reference_interactions_path.exists():
        print(f"[theme_interactions] Missing reference interactions file: {reference_interactions_path}")
        return pd.DataFrame(columns=out_cols)
    if not model_interactions_path.exists():
        print(f"[theme_interactions] Missing model interactions file: {model_interactions_path}")
        return pd.DataFrame(columns=out_cols)

    try:
        ref_interactions = pd.read_csv(reference_interactions_path)
        mod_interactions = pd.read_csv(model_interactions_path)
    except Exception as e:
        print(f"[theme_interactions] Could not read interactions files: {e}")
        return pd.DataFrame(columns=out_cols)

    ref_events = _prepare_interaction_events(ref_interactions, reference_schedules)
    mod_events = _prepare_interaction_events(mod_interactions, model_schedules)
    event_cols = ["hid", "act_norm", "overlap_start", "overlap_end"]

    n_ref_events = int(len(ref_events[event_cols].drop_duplicates())) if not ref_events.empty else 0
    n_mod_events = int(len(mod_events[event_cols].drop_duplicates())) if not mod_events.empty else 0

    mod_singleton = 0
    if not mod_events.empty:
        mod_singleton = int(
            mod_events.groupby(event_cols)["pid"].nunique().eq(1).sum()
        )

    ratio_vs_ref = 100.0 * float(n_mod_events) / float(max(n_ref_events, 1))
    rows = [
        {
            "Theme": "Interaction",
            "Metric": "Singleton Interactions (count)",
            "Label": model_name,
            "Score": float(mod_singleton),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present (count)",
            "Label": "Reference",
            "Score": float(n_ref_events),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present (count)",
            "Label": model_name,
            "Score": float(n_mod_events),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present vs Reference (%)",
            "Label": model_name,
            "Score": ratio_vs_ref,
        },
    ]
    return pd.DataFrame(rows, columns=out_cols)


def _compute_theme_interaction_rows_from_slots(
    test_inference_dir: Path,
    model_name: str,
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    out_cols = ["Theme", "Metric", "Label", "Score"]
    probs_path = Path(test_inference_dir) / "interaction_probs.csv"
    targets_path = Path(test_inference_dir) / "interaction_targets.csv"
    mask_path = Path(test_inference_dir) / "interaction_mask.csv"

    if not (probs_path.exists() and targets_path.exists() and mask_path.exists()):
        return pd.DataFrame(columns=out_cols)

    try:
        probs_2d = pd.read_csv(probs_path, header=None).to_numpy(dtype=np.float32)
        targets_2d = pd.read_csv(targets_path, header=None).to_numpy(dtype=np.float32)
        mask_2d = pd.read_csv(mask_path, header=None).to_numpy(dtype=np.float32)
    except Exception as e:
        print(f"[theme_interactions_slots] Could not read slot arrays: {e}")
        return pd.DataFrame(columns=out_cols)

    if probs_2d.shape != targets_2d.shape:
        print(
            "[theme_interactions_slots] Shape mismatch probs/targets: "
            f"{probs_2d.shape} vs {targets_2d.shape}"
        )
        return pd.DataFrame(columns=out_cols)
    if mask_2d.shape[0] != probs_2d.shape[0]:
        print(
            "[theme_interactions_slots] Shape mismatch mask/probs rows: "
            f"{mask_2d.shape[0]} vs {probs_2d.shape[0]}"
        )
        return pd.DataFrame(columns=out_cols)

    b = probs_2d.shape[0]
    members = mask_2d.shape[1]
    if members <= 0 or probs_2d.shape[1] % members != 0:
        print(
            "[theme_interactions_slots] Invalid [B, M, T] reshape with "
            f"probs shape {probs_2d.shape} and mask shape {mask_2d.shape}."
        )
        return pd.DataFrame(columns=out_cols)

    t = probs_2d.shape[1] // members
    probs_bmt = probs_2d.reshape(b, members, t)
    targets_bmt = targets_2d.reshape(b, members, t)
    member_mask = (mask_2d > 0)[:, :, None]

    model_bin = (probs_bmt >= float(prob_threshold)).astype(np.uint8)
    target_bin = (targets_bmt >= 0.5).astype(np.uint8)

    # on garde seulement les membres actifs ; les membres masqués ne doivent pas compter dans les slots
    model_bin = np.where(member_mask, model_bin, 0)
    target_bin = np.where(member_mask, target_bin, 0)

    # agrégation au niveau slot par ménage en sommant sur les membres
    model_slot_sum = model_bin.sum(axis=1)
    target_slot_sum = target_bin.sum(axis=1)

    mod_singleton = int((model_slot_sum == 1).sum())
    n_ref_present = int((target_slot_sum >= 1).sum())
    n_mod_present = int((model_slot_sum >= 1).sum())
    ratio_vs_ref = 100.0 * float(n_mod_present) / float(max(n_ref_present, 1))

    rows = [
        {
            "Theme": "Interaction",
            "Metric": "Singleton Interactions (slot count)",
            "Label": model_name,
            "Score": float(mod_singleton),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present (slot count)",
            "Label": "Reference",
            "Score": float(n_ref_present),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present (slot count)",
            "Label": model_name,
            "Score": float(n_mod_present),
        },
        {
            "Theme": "Interaction",
            "Metric": "Interactions Present vs Reference (%)",
            "Label": model_name,
            "Score": ratio_vs_ref,
        },
    ]
    return pd.DataFrame(rows, columns=out_cols)


def main():
    """fonction principale d'entraînement, d'inférence et d'évaluation."""
    parser = argparse.ArgumentParser(
        description="Train or run inference/generation/evaluation from a checkpoint."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "ckpt", "checkpoint"],
        default=None,
        help="Run mode. Use 'ckpt' to run from an existing checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: configs/household_discCNN_VAE.yaml).",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint tag/name (last/latest/best or filename in checkpoints dir).",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Full path to a checkpoint file.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version directory name under logs/<name>/ (overrides config).",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Output subdirectory name under the version folder.",
    )
    parser.add_argument(
        "--infer",
        dest="infer",
        action="store_true",
        help="Enable inference (checkpoint mode).",
    )
    parser.add_argument(
        "--no-infer",
        dest="infer",
        action="store_false",
        help="Disable inference (checkpoint mode).",
    )
    parser.add_argument(
        "--gen",
        dest="gen",
        action="store_true",
        help="Enable generation (checkpoint mode).",
    )
    parser.add_argument(
        "--no-gen",
        dest="gen",
        action="store_false",
        help="Disable generation (checkpoint mode).",
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="Enable evaluation (checkpoint mode).",
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Disable evaluation (checkpoint mode).",
    )
    parser.set_defaults(infer=None, gen=None, eval=None)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (root_dir / config_path).resolve()
    else:
        config_path = root_dir / "configs" / "household_discCNN_VAE.yaml"

    print(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # enregistrement des modèles dans la bibliothèque pour que le runner puisse les instancier par nom
    library["household_discCNN_VAE"] = HouseholdDiscCNNVAE
    library["household_discCNN_VAE_enhanced"] = HouseholdDiscCNNVAEEnhanced
    library["household_discCNN_CVAE"] = HouseholdDiscCNNCVAE
    library["household_discCNN_CVAE_attention"] = HouseholdDiscCNNCVAEAttention
    library["household_discCNN_CVAE_att_interactions"] = HouseholdDiscCNNCVAEAttentionInteractions
    library["household_discCNN_CVAE_fullattention_interactions"] = HouseholdDiscCNNCVAEFullAttentionInteractions

    print(f"   Model: {config['model_params']['name']}")
    print(
        f"   Epochs: {config.get('trainer_params', {}).get('max_epochs', 'default')}"
    )
    print(f"   Latent dim: {config.get('model_params', {}).get('latent_dim', 32)}")
    print(
        f"   2D Latent: [B, {config.get('model_params', {}).get('max_members', 5)}, latent_dim]"
    )

    checkpoint_params = config.get("checkpoint_params", {})
    mode = args.mode or checkpoint_params.get("mode", "train")
    if mode == "checkpoint":
        mode = "ckpt"

    if mode == "train":
        runners.run_command(
            config=config,
            verbose=True,
            gen=True,
            test=True,
            infer=True
        )

        # lance l'évaluation si activée dans la config
        eval_params = config.get("baseline_evaluation", {})
        if eval_params.get("enabled", True):
            print("\n" + "=" * 70)
            print("   RUNNING BASELINE EVALUATION")
            print("=" * 70)
            run_enhanced_evaluation(config, eval_params)
    elif mode == "ckpt":
        run_infer = (
            args.infer
            if args.infer is not None
            else checkpoint_params.get("run_infer", True)
        )
        run_gen = (
            args.gen
            if args.gen is not None
            else checkpoint_params.get("run_gen", True)
        )
        run_eval = (
            args.eval
            if args.eval is not None
            else checkpoint_params.get("run_eval", True)
        )

        runners.run_checkpoint_command(
            config=config,
            ckpt=args.ckpt or checkpoint_params.get("ckpt", None),
            ckpt_path=args.ckpt_path or checkpoint_params.get("ckpt_path", None),
            version_name=args.version
            or checkpoint_params.get("version_name", None),
            output_tag=args.output_tag
            or checkpoint_params.get("output_tag", None),
            infer=run_infer,
            gen=run_gen,
            eval=run_eval,
            verbose=True,
        )

        # évaluation baseline optionnelle sur les derniers schedules synthétiques générés
        eval_params = config.get("baseline_evaluation", {})
        if run_eval and eval_params.get("enabled", True):
            print("\n" + "=" * 70)
            print("   RUNNING BASELINE EVALUATION")
            print("=" * 70)
            run_enhanced_evaluation(config, eval_params)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_enhanced_evaluation(config, eval_params):
    """lance l'évaluation complète (thématique + conjointe + conditionnelle) sur les schedules générés."""

    # chemin du logger pour retrouver les fichiers de sortie
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(logger_params.get("name", "household_discCNN_VAE"))

    # on cherche le fichier de schedules synthétiques le plus récent
    experiment_dir = log_dir / name
    if not experiment_dir.exists():
        print(f"Warning: Experiment directory {experiment_dir} not found. Skipping evaluation.")
        return

    # recherche récursive de tous les fichiers synthetic_schedules*.csv dans les sous-répertoires
    synthetic_files = list(experiment_dir.rglob("synthetic_schedules*.csv"))

    if not synthetic_files:
        print("Warning: No synthetic schedules found. Skipping evaluation.")
        return

    # on prend le plus récemment modifié
    latest_synthetic = max(synthetic_files, key=lambda f: f.stat().st_mtime)
    latest_version = latest_synthetic.parent
    print(f"Using results from: {latest_version}")

    # chargement des schedules synthétiques
    synthetic_path = latest_version / "synthetic_schedules_with_attributes.csv"
    if not synthetic_path.exists():
        synthetic_path = latest_version / "synthetic_schedules.csv"

    if not synthetic_path.exists():
        print(f"Warning: Synthetic schedules not found at {synthetic_path}")
        return

    print(f"Loading synthetic schedules from {synthetic_path}")
    df_synthetic_schedules = pd.read_csv(synthetic_path)

    # chargement des schedules observés (référence)
    schedules_path = Path(config["schedules_path"])
    print(f"Loading observed schedules from {schedules_path}")
    df_observed_schedules = pd.read_csv(schedules_path)

    # chargement des attributs si disponibles (pour enrichir les schedules observés)
    attributes_path = config.get("attributes_path")
    if attributes_path and Path(attributes_path).exists():
        df_observed_attributes = pd.read_csv(attributes_path)
        df_observed_schedules = df_observed_schedules.merge(
            df_observed_attributes, on="pid", how="left"
        )
    if 'age' in df_synthetic_schedules.columns:
        df_synthetic_schedules['age'] = df_synthetic_schedules['age'].astype(str)

    # 2. Gestion de COMMUTE_MODE
    # Le problème : 'nan' est un float.
    # Solution : On remplace les trous par "UNKNOWN" (ou "PAD") puis on convertit en str
    if 'commute_mode' in df_synthetic_schedules.columns:
        df_synthetic_schedules['commute_mode'] = df_synthetic_schedules['commute_mode'].fillna("UNKNOWN").astype(str)
    print(df_synthetic_schedules)
    print(df_observed_schedules)
    # conversion en polars et préparation pour l'évaluation (format minutes)
    df_syn_pl = pl.from_pandas(df_synthetic_schedules)
    df_obs_pl = pl.from_pandas(df_observed_schedules)

    df_syn_continuous = prepare_eval_df_from_minutes(df_syn_pl)
    df_obs_continuous = prepare_eval_df_from_minutes(df_obs_pl)

    # initialisation des évaluateurs
    evaluator = ThemeEvaluator(config=eval_params)
    visualizer = Visualizer(config=eval_params)

    print("\n" + "-" * 70)
    print("   THEME EVALUATOR METRICS")
    print("-" * 70)

    model_name = config["model_params"]["name"]
    df_scores = evaluator.evaluate_all(
        df_ref=df_obs_continuous,
        df_model=df_syn_continuous,
        label_model=model_name
    )
    test_infer_dir = latest_version / "test_inference"
    interaction_eval_params = eval_params.get("interaction_overlap_evaluation", {})
    prob_threshold = float(interaction_eval_params.get("prob_threshold", 0.5))

    df_interaction_scores = _compute_theme_interaction_rows_from_slots(
        test_inference_dir=test_infer_dir,
        model_name=model_name,
        prob_threshold=prob_threshold,
    )

    if df_interaction_scores.empty:
        reference_interactions_path = config.get("interactions_path")
        model_interactions_path = latest_version / "synthetic_interactions.csv"
        if reference_interactions_path:
            df_interaction_scores = _compute_theme_interaction_rows(
                reference_schedules=df_observed_schedules,
                model_schedules=df_synthetic_schedules,
                reference_interactions_path=Path(reference_interactions_path),
                model_interactions_path=model_interactions_path,
                model_name=model_name,
            )

    if not df_interaction_scores.empty:
        df_scores = pd.concat([df_scores, df_interaction_scores], ignore_index=True)

    # affichage des résultats
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n--- Evaluation Scores ---")
    print(df_scores.to_string(index=False))

    # sauvegarde des résultats
    eval_output_dir = latest_version / "baseline_evaluation"
    eval_output_dir.mkdir(exist_ok=True, parents=True)

    scores_path = eval_output_dir / "theme_evaluation_scores.csv"
    df_scores.to_csv(scores_path, index=False)
    print(f"\nSaved evaluation scores to: {scores_path}")

    novelty_params = eval_params.get("novelty_evaluation", {})
    if novelty_params.get("enabled", False):
        print("\n" + "-" * 70)
        print("   NOVELTY / GENERALIZATION EVALUATION")
        print("-" * 70)

        novelty_dir = eval_output_dir / "novelty_evaluation"
        run_novelty_evaluation(
            observed_schedules=df_observed_schedules,
            synthetic_schedules=df_synthetic_schedules,
            write_path=novelty_dir,
            test_rate=float(novelty_params.get("test_rate", 0.1)),
            seed=int(novelty_params.get("seed", config.get("seed", 12345))),
            batch_size=int(novelty_params.get("batch_size", 100)),
        )

    # génération des plots si activée
    if eval_params.get("generate_plots", True):
        print("\n" + "-" * 70)
        print("   GENERATING DISTRIBUTION PLOTS")
        print("-" * 70)

        print("Creating marginal distributions...")
        visualizer.plot_marginal_distributions(
            df_ref=df_obs_continuous,
            df_model=df_syn_continuous,
            write_path=eval_output_dir,
            model_name=model_name
        )

        print("Creating joint distributions...")
        target_acts = eval_params.get("target_activities", ["work", "shop", "education", "home"])
        visualizer.plot_joint_distribution_2d(
            df_ref=df_obs_continuous,
            df_model=df_syn_continuous,
            write_path=eval_output_dir,
            model_name=model_name,
            target_acts=target_acts
        )

        resolution_min = 10
        print(f"Creating stacked tempograms (resolution={resolution_min} min)...")
        visualizer.plot_stacked_tempograms(
            df_ref=df_obs_continuous,
            df_model=df_syn_continuous,
            write_path=eval_output_dir,
            model_name=model_name,
            resolution_min=resolution_min,
        )

    # évaluation des déplacements et activités conjoints
    joint_trip_params = eval_params.get("joint_trip_evaluation", {})
    interaction_model_names = {
        "household_discCNN_CVAE_att_interactions",
        "household_discCNN_CVAE_fullattention_interactions",
    }
    has_test_inference_interaction_files = (
        (test_infer_dir / "interaction_targets.csv").exists()
        and (test_infer_dir / "interaction_mask.csv").exists()
    )
    has_synthetic_interactions_file = (latest_version / "synthetic_interactions.csv").exists()
    has_interaction_files = (
        has_test_inference_interaction_files or has_synthetic_interactions_file
    )

    interaction_eval_enabled = interaction_eval_params.get(
        "enabled",
        model_name in interaction_model_names or has_interaction_files,
    )

    if interaction_eval_enabled:
        print("\n" + "-" * 70)
        print("   INTERACTION OVERLAP EVALUATION (NO ESCORT)")
        print("-" * 70)

        trip_activity_name = interaction_eval_params.get(
            "trip_activity_name",
            joint_trip_params.get("trip_activity_name", "trip"),
        )
        synthetic_interactions_path = latest_version / "synthetic_interactions.csv"
        reference_interactions_path = config.get("interactions_path")
        print(f"Reference interactions source: {reference_interactions_path}")
        print(f"Model interactions source: {synthetic_interactions_path}")
        print(f"Test inference source: {test_infer_dir}")

        overlap_eval_dir = eval_output_dir / "interaction_overlap_evaluation"
        run_interaction_overlap_evaluation(
            write_path=overlap_eval_dir,
            model_name=model_name,
            trip_activity_name=trip_activity_name,
            generate_plots=interaction_eval_params.get("generate_plots", True),
            evaluate_test_inference=interaction_eval_params.get(
                "evaluate_test_inference", False
            ),
            evaluate_synthetic=interaction_eval_params.get("evaluate_synthetic", True),
            prob_threshold=float(interaction_eval_params.get("prob_threshold", 0.5)),
            test_inference_dir=test_infer_dir,
            reference_schedules=df_observed_schedules,
            model_schedules=df_synthetic_schedules,
            reference_interactions_path=reference_interactions_path,
            model_interactions_path=synthetic_interactions_path,
        )
        print(f"\nInteraction overlap evaluation results saved to: {overlap_eval_dir}")
    elif joint_trip_params.get("enabled", True):
        print("\n" + "-" * 70)
        print("   JOINT TRIP/ACTIVITY EVALUATION")
        print("-" * 70)

        # vérification que l'activité 'trip' existe dans les données (sinon on ne peut pas évaluer les déplacements)
        trip_activity_name = joint_trip_params.get("trip_activity_name", "trip")
        has_trips_syn = df_synthetic_schedules['act'].str.lower().str.contains(trip_activity_name.lower()).any()
        has_trips_obs = df_observed_schedules['act'].str.lower().str.contains(trip_activity_name.lower()).any()

        if has_trips_syn and has_trips_obs:
            print(f"Trip activity '{trip_activity_name}' found in both datasets.")

            joint_eval_dir = eval_output_dir / "joint_trip_evaluation"

            run_joint_trip_evaluation(
                reference_schedules=df_observed_schedules,
                model_schedules=df_synthetic_schedules,
                write_path=joint_eval_dir,
                model_name=model_name,
                buffer_time_min=joint_trip_params.get("buffer_time_min", 5.0),
                buffer_duration_min=joint_trip_params.get("buffer_duration_min", 5.0),
                trip_activity_name=trip_activity_name,
                generate_plots=joint_trip_params.get("generate_plots", True),
            )

            print(f"\nJoint trip evaluation results saved to: {joint_eval_dir}")
        else:
            if not has_trips_obs:
                print(f"Warning: Trip activity '{trip_activity_name}' not found in observed schedules.")
            if not has_trips_syn:
                print(f"Warning: Trip activity '{trip_activity_name}' not found in synthetic schedules.")
            print("Skipping joint trip evaluation. Joint activity analysis will still run.")

            joint_eval_dir = eval_output_dir / "joint_activity_evaluation"

            evaluator_joint = JointTripEvaluator(
                buffer_time_min=joint_trip_params.get("buffer_time_min", 5.0),
                buffer_duration_min=joint_trip_params.get("buffer_duration_min", 5.0),
                trip_activity_name=trip_activity_name,
            )

            evaluator_joint.evaluate(
                df_ref=df_observed_schedules,
                df_model=df_synthetic_schedules,
                write_path=joint_eval_dir,
                label_model=model_name,
                generate_plots=joint_trip_params.get("generate_plots", True),
            )
            print(f"\nJoint activity evaluation results saved to: {joint_eval_dir}")

    # évaluation conditionnelle par attribut socio-démo
    conditional_params = eval_params.get("conditional_evaluation", {})
    if conditional_params.get("enabled", False):
        print("\n" + "-" * 70)
        print("   CONDITIONAL EVALUATION (BY DEMOGRAPHICS)")
        print("-" * 70)

        # chargement des attributs si disponibles (pour le conditionnement)
        attributes_path = config.get("attributes_path")
        synthetic_attributes_path = config.get("synthetic_attributes_path")

        if attributes_path and Path(attributes_path).exists():
            print(f"Loading observed attributes from {attributes_path}")
            df_obs_attrs = pd.read_csv(attributes_path)

            # chargement des attributs synthétiques
            if synthetic_attributes_path and Path(synthetic_attributes_path).exists():
                print(f"Loading synthetic attributes from {synthetic_attributes_path}")
                df_syn_attrs = pd.read_csv(synthetic_attributes_path)
            else:
                # si pas d'attributs synthétiques, on utilise ceux de la référence (même population)
                print("Warning: No synthetic attributes found, using observed attributes")
                df_syn_attrs = df_obs_attrs.copy()

            # récupération des paramètres d'évaluation conditionnelle
            condition_on = conditional_params.get("condition_on", ["age", "sex"])
            age_bins = conditional_params.get("age_bins", [6, 11, 15, 18, 30, 40, 50, 60, 70])

            # initialisation des évaluateurs
            cond_eval = ConditionalMetrics(config=eval_params, evaluator=evaluator)
            cond_vis = ConditionalVisualizer(config=eval_params)

            print(f"Computing conditional metrics for: {condition_on}")

            # calcul des métriques conditionnelles par attribut
            df_cond = cond_eval.compute_conditional_metrics(
                df_ref=df_obs_continuous,
                df_model=df_syn_continuous,
                ref_attributes=df_obs_attrs,
                model_attributes=df_syn_attrs,
                condition_on=condition_on,
                age_bins=age_bins,
                label_model=model_name
            )

            # sauvegarde des résultats conditionnels
            cond_eval_dir = eval_output_dir / "conditional_evaluation"
            cond_eval_dir.mkdir(exist_ok=True, parents=True)

            cond_scores_path = cond_eval_dir / "conditional_evaluation_scores.csv"
            df_cond.to_csv(cond_scores_path, index=False)
            print(f"\nSaved conditional evaluation scores to: {cond_scores_path}")

            # affichage du récapitulatif
            if not df_cond.empty:
                print("\n--- Conditional Evaluation Summary ---")
                print(df_cond.head(20).to_string(index=False))
                print(f"\n... ({len(df_cond)} total rows)")

                # génération des plots si activée
                if conditional_params.get("generate_plots", True):
                    print("\nGenerating conditional evaluation plots...")

                    # heatmaps pour chaque attribut conditionnel
                    for attr in condition_on:
                        if attr == "age":
                            attr = "age_group"  # Use age_group after binning

                        # heatmap pour les métriques de timing
                        cond_vis.plot_heatmap(
                            df_cond, attr, "EMD",
                            model_name=model_name,
                            write_path=cond_eval_dir
                        )

                    # distributions par condition pour les activités clés
                    target_activities = conditional_params.get(
                        "target_activities", ["work", "education"]
                    )
                    for attr in condition_on:
                        if attr == "age":
                            attr = "age_group"

                        for activity in target_activities:
                            try:
                                cond_vis.plot_distributions_by_condition(
                                    df_ref=df_obs_continuous,
                                    df_model=df_syn_continuous,
                                    ref_attrs=df_obs_attrs,
                                    mod_attrs=df_syn_attrs,
                                    condition_attr=attr,
                                    age_bins=age_bins,
                                    activity=activity,
                                    write_path=cond_eval_dir
                                )
                            except Exception as e:
                                print(f"Warning: Could not plot {activity} by {attr}: {e}")

                    print(f"Saved conditional plots to: {cond_eval_dir}")
            else:
                print("Warning: No conditional metrics computed (insufficient data)")
        else:
            print(f"Warning: Attributes file not found at {attributes_path}")
            print("Skipping conditional evaluation.")
            print("To enable conditional evaluation, add 'attributes_path' to your config.")

    model_params = config.get("model_params", {})
    max_members = model_params.get("max_members", None)
    is_hierarchical = bool(model_params.get("use_hierarchical", False))
    latent_plot_cfg = eval_params.get("latent_plots", {})
    latent_plots_enabled = bool(latent_plot_cfg.get("enabled", True))
    plot_both_hierarchical_latents = bool(
        latent_plot_cfg.get("plot_both_hierarchical_latents", True)
    )

    if latent_plots_enabled:
        if is_hierarchical and plot_both_hierarchical_latents:
            print("\nHierarchical model detected: generating latent plots for z_coord and z_indiv.")

            coord_file = latent_plot_cfg.get("coord_latent_file", "mus_coord.csv")
            indiv_file = latent_plot_cfg.get("indiv_latent_file", "mus.csv")
            coord_file, indiv_file = _resolve_hierarchical_latent_files(
                latest_version, coord_file, indiv_file
            )
            coord_mode = str(model_params.get("coord_mode", "2d")).lower()
            coord_point_mode = "household" if coord_mode == "1d" else "per_member"

            visualize_latent_umap(
                latest_version,
                model_name,
                df_synthetic_schedules,
                latent_filename=coord_file,
                latent_label="z_coord",
                output_filename="coord_latent_umap.png",
                point_mode=coord_point_mode,
            )
            plot_latent_diagnostics(
                latest_version,
                model_name,
                latent_filename=coord_file,
                latent_label="z_coord",
                output_prefix="coord_latent",
                max_members=max_members,
            )

            visualize_latent_umap(
                latest_version,
                model_name,
                df_synthetic_schedules,
                latent_filename=indiv_file,
                latent_label="z_indiv",
                output_filename="indiv_latent_umap.png",
                point_mode="per_member",
            )
            plot_latent_diagnostics(
                latest_version,
                model_name,
                latent_filename=indiv_file,
                latent_label="z_indiv",
                output_prefix="indiv_latent",
                max_members=max_members,
            )
        else:
            visualize_latent_umap(latest_version, model_name, df_synthetic_schedules)
            plot_mu_diagnostics(
                latest_version,
                model_name,
                max_members=max_members,
            )
    else:
        print("Latent plots are disabled by baseline_evaluation.latent_plots.enabled=false")
    print(f"\n{'='*70}")
    print("   EVALUATION COMPLETED")
    print(f"{'='*70}\n")



if __name__ == "__main__":
    main()

