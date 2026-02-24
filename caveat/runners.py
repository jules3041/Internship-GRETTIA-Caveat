"""
Orchestrateur principal du pipeline CAVEAT

Ce fichier coordonne toutes les étapes du pipeline de génération synthétique :
    1. Chargement des données (emplois du temps + attributs socio-démographiques)
    2. Encodage des données via les encodeurs spécialisés (encoding/ et label_encoding/)
    3. Construction du DataModule PyTorch Lightning (data/module.py)
    4. Instanciation et entraînement du modèle VAE (models/)
    5. Inférence sur le jeu de test (reconstruction des schedules observés)
    6. Génération de nouvelles populations synthétiques
    7. Évaluation comparative entre données réelles et synthétiques

Architecture des liens entre fichiers :
    - cli.py appelle les fonctions *_command() de ce fichier (run_command, batch_command, etc.)
    - load_data() utilise data/loaders.py pour charger les CSV bruts
    - encode_input_labels() instancie un encodeur depuis label_encoding/ (onehot ou tokens)
    - encode_schedules() instancie un encodeur depuis encoding/ (discrete, continuous, household...)
    - build_model() instancie un modèle depuis models/__init__.py via le registre global
    - Le modèle hérite de experiment.py (LightningModule), qui gère training/validation/loss
    - generate() produit de nouvelles séquences en échantillonnant l'espace latent
    - evaluate_synthetics() compare les sorties avec evaluate/

"""

import datetime
import pickle
import re
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.random import seed as seeder

from caveat import cuda_available, data, encoding, label_encoding, models
from caveat.callbacks import LinearLossScheduler
from caveat.data.module import DataModule
from caveat.data.household_dataset import bin_age_value, DEFAULT_AGE_BINS
from caveat.encoding import BaseDataset, BaseEncoder
from caveat.evaluate import evaluate
from caveat.label_encoding.base import BaseLabelEncoder
import inspect
class DelayedEarlyStopping(EarlyStopping):
    """EarlyStopping différé : empêche l'arrêt prématuré pendant la phase de warm-up.

    Pourquoi : les VAE ont souvent une phase initiale chaotique où la loss oscille
    fortement (le KLD collapse ou l'activité domine). Activer l'early stopping trop
    tôt provoquerait un arrêt avant que le modèle ait pu stabiliser ses gradients.
    On attend donc 'start_epoch' époques avant de commencer à surveiller la val_loss.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _run_early_stopping_check(self, trainer):
        # On bloque complètement l'exécution du check avant l'époque cible,
        # car durant les premières époques la loss est instable (warm-up du KLD)
        if trainer.current_epoch < self.start_epoch:
            return

        # Une fois l'époque atteinte, on délègue à la logique standard de Lightning
        super()._run_early_stopping_check(trainer)

def safe_decode(encoder, schedules, argmax=True, mask=None):
    """Décodage adaptatif des tenseurs en DataFrame de schedules.

    Pourquoi cette indirection : le pipeline supporte deux familles d'encodeurs :
      - Les encodeurs classiques (discrete, continuous) qui n'ont pas de notion de masque
      - Le HouseholdEncoder qui a besoin d'un masque pour filtrer les membres fantômes
        (les ménages sont paddés à max_members, les membres inexistants doivent être ignorés)
    On inspecte dynamiquement la signature de decode() pour rester compatible avec les deux.
    """
    sig = inspect.signature(encoder.decode)
    if 'mask' in sig.parameters:
        return encoder.decode(schedules, argmax=argmax, mask=mask)
    else:
        return encoder.decode(schedules, argmax=argmax)


def safe_decode_attributes(encoder, attributes, mask=None):
    """Décodage adaptatif des attributs socio-démographiques (âge, sexe, emploi...).

    Pourquoi : dans le mode Household, les attributs sont encodés par le HouseholdEncoder
    lui-même (pas par un label_encoder séparé). Le décodage inverse nécessite le masque
    pour savoir quels membres du ménage sont réels vs fantômes (padding).
    En mode classique, decode_attributes n'existe pas sur l'encodeur → on retourne None.
    """
    if not hasattr(encoder, 'decode_attributes'):
        return None

    sig = inspect.signature(encoder.decode_attributes)
    if 'mask' in sig.parameters:
        return encoder.decode_attributes(attributes, mask=mask)
    else:
        return encoder.decode_attributes(attributes)


def _ensure_schedule_id_column(df: DataFrame, start_id: int = 0) -> DataFrame:
    out = df.copy()
    if "schedule_id" not in out.columns:
        out.insert(0, "schedule_id", pd.RangeIndex(start_id, start_id + len(out), 1))
        return out

    schedule_ids = pd.to_numeric(out["schedule_id"], errors="coerce")
    if schedule_ids.isna().any():
        missing = schedule_ids.isna()
        next_id = int(schedule_ids.max()) + 1 if schedule_ids.notna().any() else int(start_id)
        fill_vals = pd.Series(
            range(next_id, next_id + int(missing.sum())),
            index=schedule_ids.index[missing],
            dtype="int64",
        )
        schedule_ids.loc[missing] = fill_vals
    out["schedule_id"] = schedule_ids.astype("int64")
    return out


def _normalize_dataloader(dataloaders):
    if dataloaders is None:
        return None
    if isinstance(dataloaders, dict):
        return next(iter(dataloaders.values()), None)
    if isinstance(dataloaders, (list, tuple)):
        return dataloaders[0] if dataloaders else None
    return dataloaders


def _extract_mask_from_dataloader(dataloaders):
    loader = _normalize_dataloader(dataloaders)
    if loader is None:
        return None
    masks = []
    for batch in loader:
        if isinstance(batch, dict) and "mask" in batch:
            masks.append(batch["mask"])
    if masks:
        return torch.concat(masks)
    return None


def _save_interaction_overview_image(
    write_dir: Path,
    interaction_probs: torch.Tensor,
    interaction_targets: Optional[torch.Tensor] = None,
    interaction_mask: Optional[torch.Tensor] = None,
) -> list[Path]:
    if interaction_probs is None or interaction_probs.numel() == 0:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Could not generate interaction image (matplotlib unavailable): {e}")
        return []

    probs = interaction_probs.detach().float().cpu()
    targets = (
        interaction_targets.detach().float().cpu()
        if interaction_targets is not None
        else None
    )
    mask = (
        interaction_mask.detach().bool().cpu()
        if interaction_mask is not None
        else None
    )

    out_dir = Path(write_dir) / "interaction_maps"
    out_dir.mkdir(exist_ok=True, parents=True)

    n_samples = probs.shape[0]
    indices = torch.arange(n_samples)
    if mask is not None and mask.ndim == 2 and mask.shape[0] == n_samples:
        valid_hh = mask.sum(dim=1) > 0
        indices = torch.where(valid_hh)[0]
    if targets is not None and targets.ndim == 3 and targets.shape[0] == n_samples:
        has_pos = targets.sum(dim=(1, 2)) > 0
        pos_indices = torch.where(has_pos)[0]
        if pos_indices.numel() > 0:
            if indices.numel() > 0:
                indices = indices[torch.isin(indices, pos_indices)]
            else:
                indices = pos_indices

    if indices.numel() == 0:
        indices = torch.arange(min(n_samples, 10))
    else:
        indices = indices[:10]

    saved_paths: list[Path] = []
    cmap = "viridis"
    for idx in indices.tolist():
        prob_map = probs[idx]
        true_map = targets[idx] if targets is not None and targets.ndim == 3 else None

        if mask is not None and mask.ndim == 2 and idx < mask.shape[0]:
            active_members = torch.where(mask[idx])[0]
            if active_members.numel() > 0:
                prob_map = prob_map[active_members]
                if true_map is not None:
                    true_map = true_map[active_members]

        pred_map = (prob_map >= 0.5).float()

        panels = []
        if true_map is not None:
            panels.append(("Ground Truth", true_map.numpy(), cmap, 0.0, 1.0))
        panels.append(("Prediction", pred_map.numpy(), cmap, 0.0, 1.0))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, arr, panel_cmap, vmin, vmax) in zip(axes, panels):
            ax.imshow(arr, aspect="auto", cmap=panel_cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Member")

        fig.suptitle(f"Interaction Maps (sample #{idx})")
        fig.tight_layout()
        out_path = out_dir / f"interaction_overview_{idx:04d}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def _infer_hh_member_from_row(row: pd.Series) -> tuple[Optional[int], Optional[int]]:
    if "hid" in row.index and "member_idx" in row.index:
        try:
            return int(row["hid"]), int(row["member_idx"])
        except Exception:
            return None, None
    if "pid" in row.index:
        match = re.search(r"HH(\d+)_M(\d+)", str(row["pid"]))
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def _merge_intervals(intervals: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + eps:
            cur_end = max(cur_end, end)
        else:
            if cur_end > cur_start:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_end > cur_start:
        merged.append((cur_start, cur_end))
    return merged


def _build_synthetic_interactions_from_logits(
    schedules_df: pd.DataFrame,
    interaction_logits: Optional[torch.Tensor],
    interaction_mask: Optional[torch.Tensor],
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    columns = ["schedule_id", "pid", "interaction_bin", "overlap_start", "overlap_end"]
    if interaction_logits is None or not isinstance(interaction_logits, torch.Tensor):
        return pd.DataFrame(columns=columns)
    if interaction_logits.ndim != 3:
        return pd.DataFrame(columns=columns)

    probs = torch.sigmoid(interaction_logits.detach()).cpu().numpy()
    pred = (probs >= float(prob_threshold)).astype(np.uint8)
    b, m, t = pred.shape
    slot_edges = np.arange(t + 1, dtype=np.float32) * (1440.0 / float(t))
    step_size = 1440.0 / float(t)

    if isinstance(interaction_mask, torch.Tensor) and interaction_mask.ndim == 2:
        mask_arr = interaction_mask.detach().cpu().numpy() > 0
    else:
        mask_arr = np.ones((b, m), dtype=bool)

    rows = []
    for _, row in schedules_df.iterrows():
        if "schedule_id" not in row.index:
            continue
        try:
            schedule_id = int(row["schedule_id"])
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue
        if end <= start:
            continue

        hh_idx, member_idx = _infer_hh_member_from_row(row)
        if hh_idx is None or member_idx is None:
            continue
        if hh_idx < 0 or member_idx < 0 or hh_idx >= b or member_idx >= m:
            continue
        if not bool(mask_arr[hh_idx, member_idx]):
            continue

        s_slot = int(start / step_size)
        e_slot = int((end - 1e-4) / step_size)
        if e_slot < 0 or s_slot >= t:
            continue
        s_slot = max(0, s_slot)
        e_slot = min(t - 1, e_slot)
        if s_slot > e_slot:
            continue

        intervals = []
        for slot in range(s_slot, e_slot + 1):
            if pred[hh_idx, member_idx, slot] <= 0:
                continue
            slot_start = float(slot_edges[slot])
            slot_end = float(slot_edges[slot + 1])
            overlap_start = max(start, slot_start)
            overlap_end = min(end, slot_end)
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))

        for overlap_start, overlap_end in _merge_intervals(intervals):
            rows.append(
                {
                    "schedule_id": schedule_id,
                    "pid": str(row.get("pid", "")),
                    "interaction_bin": 1,
                    "overlap_start": float(overlap_start),
                    "overlap_end": float(overlap_end),
                }
            )

    return pd.DataFrame(rows, columns=columns)


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


def _prepare_overlap_interactions(
    df_schedules: pd.DataFrame,
    df_interactions: pd.DataFrame,
) -> pd.DataFrame:
    if df_schedules is None or df_schedules.empty or df_interactions is None:
        return pd.DataFrame()

    cols = ["schedule_id", "pid", "act", "hid", "start", "end"]
    schedules = _ensure_hid_column(df_schedules)
    sched_meta = schedules[[c for c in cols if c in schedules.columns]].copy()

    dfi = df_interactions.copy()
    if dfi.empty:
        return dfi

    if "interaction_bin" in dfi.columns:
        dfi = dfi[dfi["interaction_bin"].fillna(0).astype(int) > 0].copy()

    if {"schedule_id", "pid"}.issubset(dfi.columns) and {"schedule_id", "pid"}.issubset(
        sched_meta.columns
    ):
        dfi = dfi.merge(sched_meta, on=["schedule_id", "pid"], how="left")

    if "interaction_type" not in dfi.columns:
        if "act" in dfi.columns:
            dfi["interaction_type"] = (
                dfi["act"].astype(str).str.lower() == "trip"
            ).map({True: 2, False: 1}).astype(int)
        else:
            dfi["interaction_type"] = 1

    if "overlap_start" not in dfi.columns and "start" in dfi.columns:
        dfi["overlap_start"] = dfi["start"]
    if "overlap_end" not in dfi.columns and "end" in dfi.columns:
        dfi["overlap_end"] = dfi["end"]

    if "hid" not in dfi.columns:
        dfi = _ensure_hid_column(dfi)
    if "hid" in dfi.columns:
        dfi["hid"] = pd.to_numeric(dfi["hid"], errors="coerce")

    if {"overlap_start", "overlap_end"}.issubset(dfi.columns):
        dfi["overlap_start"] = pd.to_numeric(dfi["overlap_start"], errors="coerce")
        dfi["overlap_end"] = pd.to_numeric(dfi["overlap_end"], errors="coerce")
        dfi = dfi[
            dfi["overlap_start"].notna()
            & dfi["overlap_end"].notna()
            & (dfi["overlap_end"] > dfi["overlap_start"])
        ].copy()

    return dfi


def _plot_household_joint_overlap_gantt(
    hid: int,
    df_schedules: pd.DataFrame,
    dfi_overlap: pd.DataFrame,
    df_attributes: Optional[pd.DataFrame] = None,
    interaction_logits: Optional[torch.Tensor] = None,
    interaction_mask: Optional[torch.Tensor] = None,
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        print(f"Could not generate interaction gantt (matplotlib unavailable): {e}")
        return None

    hh = df_schedules[df_schedules["hid"] == hid].copy()
    if hh.empty:
        return None

    if "act" in hh.columns:
        hh["act"] = hh["act"].astype(str)

    act_colors = {
        "home": "#f2f2f2",
        "work": "#AA3F2A",
        "shop": "#9639ee",
        "education": "#1f77b4",
        "leisure": "#2ca02c",
        "escort": "#ff7f0e",
        "other": "#d62728",
        "trip": "#f8e750",
        "unknown": "#333333",
    }

    members = sorted(hh["pid"].dropna().astype(str).unique()) if "pid" in hh.columns else []
    if not members:
        return None
    pid_to_y = {pid: i for i, pid in enumerate(members)}

    pred_panel = None
    if (
        isinstance(interaction_logits, torch.Tensor)
        and interaction_logits.ndim == 3
        and hid >= 0
        and hid < interaction_logits.shape[0]
    ):
        logits_h = interaction_logits[int(hid)].detach().float().cpu()
        mask_h = None
        if (
            isinstance(interaction_mask, torch.Tensor)
            and interaction_mask.ndim == 2
            and hid < interaction_mask.shape[0]
        ):
            mask_h = interaction_mask[int(hid)].detach().bool().cpu()

        member_idx_by_pid: dict[str, int] = {}
        if {"pid", "member_idx"}.issubset(hh.columns):
            mm = hh[["pid", "member_idx"]].dropna().drop_duplicates(subset=["pid"])
            for _, row in mm.iterrows():
                try:
                    member_idx_by_pid[str(row["pid"])] = int(row["member_idx"])
                except Exception:
                    continue

        pred_rows: list[np.ndarray] = []
        for pid in members:
            member_idx = member_idx_by_pid.get(str(pid))
            if member_idx is None:
                m = re.search(r"_M(\d+)", str(pid))
                if m:
                    member_idx = int(m.group(1))
            valid = (
                member_idx is not None
                and member_idx >= 0
                and member_idx < logits_h.shape[0]
                and (mask_h is None or member_idx >= mask_h.shape[0] or bool(mask_h[member_idx]))
            )
            if valid:
                row = (torch.sigmoid(logits_h[int(member_idx)]) >= 0.5).float().numpy()
            else:
                row = np.zeros(logits_h.shape[1], dtype=np.float32)
            pred_rows.append(row)

        if pred_rows:
            pred_panel = np.stack(pred_rows, axis=0)

    if pred_panel is not None:
        fig, (ax, ax_pred) = plt.subplots(
            1,
            2,
            figsize=(24, 6),
            gridspec_kw={"width_ratios": [3, 2]},
        )
    else:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax_pred = None

    present_acts = set()

    sort_cols = [c for c in ["pid", "start"] if c in hh.columns]
    rows = hh.sort_values(sort_cols).iterrows() if sort_cols else hh.iterrows()
    for _, r in rows:
        pid = str(r.get("pid", ""))
        if pid not in pid_to_y:
            continue
        y = pid_to_y[pid]
        try:
            start = float(r["start"])
            end = float(r["end"])
        except Exception:
            continue
        width = max(0.0, end - start)
        if width <= 0:
            continue
        act_key = str(r.get("act", "unknown")).lower().strip()
        present_acts.add(act_key)
        color = act_colors.get(act_key, act_colors["unknown"])
        ax.barh(y=y, width=width, left=start, height=0.8, color=color, edgecolor="white")

    hh_i = dfi_overlap[dfi_overlap["hid"] == hid].copy() if "hid" in dfi_overlap.columns else pd.DataFrame()
    style = {
        1: {"hatch": "///", "edgecolor": "black", "label": "Activite conjointe (overlap)"},
        2: {"hatch": "...", "edgecolor": "black", "label": "Trajet conjoint"},
    }

    for _, r in hh_i.iterrows():
        pid = str(r.get("pid", ""))
        if pid not in pid_to_y:
            continue
        y = pid_to_y[pid]
        try:
            s = float(r["overlap_start"])
            e = float(r["overlap_end"])
        except Exception:
            continue
        w = max(0.0, e - s)
        if w <= 0:
            continue
        it = int(r["interaction_type"]) if pd.notna(r.get("interaction_type")) else 1
        st = style.get(it, style[1])
        ax.barh(
            y=y,
            width=w,
            left=s,
            height=0.8,
            facecolor="none",
            edgecolor=st["edgecolor"],
            hatch=st["hatch"],
            linewidth=1.3,
        )

    y_labels = []
    attrs = df_attributes.copy() if isinstance(df_attributes, pd.DataFrame) else None
    if attrs is not None and "pid" in attrs.columns:
        attrs["pid"] = attrs["pid"].astype(str)
    for pid in members:
        label = f"PID: {pid}"
        if attrs is not None:
            attr = attrs[attrs["pid"] == pid]
            if not attr.empty:
                age = attr.iloc[0].get("age", "?")
                sex = attr.iloc[0].get("sex", "?")
                employed = attr.iloc[0].get("employed", "?")
                vehicles = attr.iloc[0].get("number_of_vehicles", "?")
                emp_txt = "Empl." if bool(employed) else "Chom."
                label = f"{pid}\n({age} ans, {sex}, {emp_txt}, {vehicles} veh.)"
        y_labels.append(label)

    ax.set_yticks(list(pid_to_y.values()))
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 1440)
    ax.set_xticks(np.arange(0, 1441, 120))
    ax.set_xticklabels([f"{int(h // 60):02d}h" for h in np.arange(0, 1441, 120)])
    ax.set_xlabel("Heure de la journee")
    ax.set_ylabel("Membres")
    ax.set_title(f"Menage {hid}: trips/activites joints (overlap)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    sorted_acts = sorted(list(present_acts))
    if sorted_acts:
        leg_act = [
            mpatches.Patch(color=act_colors.get(a, act_colors["unknown"]), label=a.capitalize())
            for a in sorted_acts
        ]
        legend1 = ax.legend(handles=leg_act, title="Activites", loc="upper left", bbox_to_anchor=(1, 1))
    else:
        legend1 = None
    leg_int = [
        mpatches.Patch(facecolor="white", edgecolor=v["edgecolor"], hatch=v["hatch"], label=v["label"])
        for v in style.values()
    ]
    if leg_int:
        ax.legend(handles=leg_int, title="Types d'interaction", loc="upper left", bbox_to_anchor=(1, 0.5))
        if legend1 is not None:
            ax.add_artist(legend1)

    if ax_pred is not None and pred_panel is not None:
        ax_pred.imshow(
            pred_panel,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=[0.0, 1440.0, float(len(members)), 0.0],
        )
        ax_pred.set_title("Interactions predites (binaire)")
        ax_pred.set_xlabel("Heure de la journee")
        ax_pred.set_ylabel("Membres")
        ax_pred.set_yticks(np.arange(len(members)) + 0.5)
        ax_pred.set_yticklabels(y_labels)
        ax_pred.set_xlim(0, 1440)
        ax_pred.set_xticks(np.arange(0, 1441, 120))
        ax_pred.set_xticklabels([f"{int(h // 60):02d}h" for h in np.arange(0, 1441, 120)])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _export_generated_interaction_gantts(
    write_dir: Path,
    schedules_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    seed: int,
    num_households: int = 10,
    attributes_df: Optional[pd.DataFrame] = None,
    interaction_logits: Optional[torch.Tensor] = None,
    interaction_mask: Optional[torch.Tensor] = None,
) -> list[Path]:
    out_dir = Path(write_dir) / "interaction_gantt"
    out_dir.mkdir(exist_ok=True, parents=True)

    schedules = _ensure_hid_column(schedules_df)
    if "hid" not in schedules.columns:
        print("Skipping interaction gantt export: missing hid/pid information.")
        return []

    dfi_overlap = _prepare_overlap_interactions(schedules, interactions_df)
    if dfi_overlap.empty:
        print("Skipping interaction gantt export: no valid interaction overlaps found.")
        return []

    hids = (
        schedules["hid"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    if not hids:
        print("Skipping interaction gantt export: no households available.")
        return []

    n_select = min(max(int(num_households), 0), len(hids))
    if n_select == 0:
        return []

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.array(hids, dtype=np.int64), size=n_select, replace=False)

    logits = (
        interaction_logits.detach().float().cpu()
        if isinstance(interaction_logits, torch.Tensor)
        else None
    )
    mask = (
        interaction_mask.detach().bool().cpu()
        if isinstance(interaction_mask, torch.Tensor)
        else None
    )

    saved_paths: list[Path] = []
    for hid in selected.tolist():
        fig = _plot_household_joint_overlap_gantt(
            hid=int(hid),
            df_schedules=schedules,
            dfi_overlap=dfi_overlap,
            df_attributes=attributes_df,
            interaction_logits=logits,
            interaction_mask=mask,
        )
        if fig is None:
            continue
        out_path = out_dir / f"gantt_hid_{int(hid)}.png"
        fig.savefig(out_path, dpi=200)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
        saved_paths.append(out_path)
    return saved_paths


def _extract_generation_tensors(
    synth: list,
    with_attributes: bool,
) -> tuple[
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if not synth:
        raise ValueError("Empty generation predictions.")
    first = synth[0]
    if not isinstance(first, (tuple, list)) or len(first) < 3:
        raise ValueError("Unexpected generation prediction format.")

    attributes = torch.concat([batch[0] for batch in synth]) if with_attributes else None
    schedules = torch.concat([batch[1] for batch in synth])
    zs = torch.concat([batch[2] for batch in synth])

    interaction_logits = None
    interaction_mask = None
    if len(first) >= 5:
        try:
            logits_list = [batch[3] for batch in synth]
            mask_list = [batch[4] for batch in synth]
            if all(isinstance(x, torch.Tensor) and x.ndim == 3 for x in logits_list) and all(
                isinstance(x, torch.Tensor) and x.ndim == 2 for x in mask_list
            ):
                interaction_logits = torch.concat(logits_list)
                interaction_mask = torch.concat(mask_list)
        except Exception:
            interaction_logits = None
            interaction_mask = None

    return attributes, schedules, zs, interaction_logits, interaction_mask


def _validate_gmm_config_migration(config: dict) -> None:
    model_params = config.get("model_params", {})
    if "use_gmm_prior" in model_params:
        raise ValueError(
            "`model_params.use_gmm_prior` a ete supprime. "
            "Utiliser `checkpoint_params.sample_with_train_gmm` et "
            "`checkpoint_params.gmm_n_components`."
        )


def _build_train_inference_dataloader(data_loader: DataModule, config: dict):
    if not hasattr(data_loader, "train_dataset"):
        data_loader.setup()
    if not hasattr(data_loader, "train_dataset"):
        raise ValueError("DataModule train_dataset is unavailable after setup().")

    loader_params = config.get("loader_params", {})
    batch_size = loader_params.get("val_batch_size", loader_params.get("train_batch_size", 128))
    return torch.utils.data.DataLoader(
        data_loader.train_dataset,
        batch_size=batch_size,
        num_workers=getattr(data_loader, "num_workers", 0),
        shuffle=False,
        pin_memory=getattr(data_loader, "pin_memory", False),
        collate_fn=getattr(data_loader, "collate_fn", None),
        persistent_workers=(
            getattr(data_loader, "persistent_workers", False)
            if getattr(data_loader, "num_workers", 0) > 0
            else False
        ),
        prefetch_factor=(
            getattr(data_loader, "prefetch_factor", None)
            if getattr(data_loader, "num_workers", 0) > 0
            else None
        ),
    )


def _fit_train_mu_indiv_gmm(
    train_infer_dir: Path,
    model: LightningModule,
    n_components: int,
    write_dir: Path,
    seed: int = 42,
):
    from sklearn.mixture import GaussianMixture

    if not getattr(model, "use_hierarchical", False):
        raise ValueError("Train GMM sampling requires a hierarchical model.")

    mus_path = train_infer_dir / "mus.csv"
    mask_path = train_infer_dir / "mask.csv"
    if not mus_path.exists():
        raise FileNotFoundError(f"Missing train latent file for GMM: {mus_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing train mask file for GMM: {mask_path}")

    mus_np = pd.read_csv(mus_path, header=None).values
    mask_np = pd.read_csv(mask_path, header=None).values

    max_members = int(getattr(model, "max_members"))
    indiv_dim = int(getattr(model, "indiv_latent_dim"))
    if mus_np.ndim != 2:
        raise ValueError(f"Expected mus.csv to be 2D, got shape {mus_np.shape}")
    if mus_np.shape[1] != max_members * indiv_dim:
        raise ValueError(
            "Unexpected mus.csv width for hierarchical indiv latents: "
            f"got {mus_np.shape[1]}, expected {max_members * indiv_dim} "
            f"(M={max_members}, D={indiv_dim})"
        )
    if mask_np.shape[0] != mus_np.shape[0] or mask_np.shape[1] != max_members:
        raise ValueError(
            "Unexpected mask.csv shape for hierarchical GMM fit: "
            f"got {mask_np.shape}, expected ({mus_np.shape[0]}, {max_members})"
        )

    B = mus_np.shape[0]
    mus_3d = mus_np.reshape(B, max_members, indiv_dim)
    active = mask_np > 0.5
    mu_active = mus_3d[active]
    if mu_active.shape[0] < int(n_components):
        raise ValueError(
            f"Not enough active-member train mus to fit GMM({n_components}): "
            f"{mu_active.shape[0]} samples"
        )

    print(
        f"[GMM] Fitting external train GMM on mu_indiv from {train_infer_dir} "
        f"(N_active={len(mu_active)}, D={indiv_dim}, K={int(n_components)})."
    )
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type="diag",
        random_state=int(seed),
        max_iter=300,
        n_init=3,
    )
    gmm.fit(mu_active)

    artifacts_dir = Path(write_dir) / "gmm_train_mu_indiv"
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(gmm.weights_).to_csv(artifacts_dir / "gmm_weights.csv", index=False, header=False)
    pd.DataFrame(gmm.means_).to_csv(artifacts_dir / "gmm_means.csv", index=False, header=False)
    gmm_stds = np.sqrt(gmm.covariances_)
    pd.DataFrame(gmm_stds).to_csv(artifacts_dir / "gmm_stds.csv", index=False, header=False)
    metadata = {
        "source_split": "train",
        "source_latent_file": "mus.csv",
        "source_mask_file": "mask.csv",
        "covariance_type": "diag",
        "n_components": int(n_components),
        "max_iter": 300,
        "n_init": 3,
        "random_state": int(seed),
        "max_members": max_members,
        "indiv_latent_dim": indiv_dim,
        "n_households": int(B),
        "n_active_members": int(mu_active.shape[0]),
    }
    with open(artifacts_dir / "gmm_metadata.yaml", "w") as fp:
        yaml.safe_dump(metadata, fp, sort_keys=False)
    print(f"[GMM] Saved train GMM artifacts to {artifacts_dir}")
    return gmm


def _sample_z_indiv_from_gmm(
    gmm,
    n_households: int,
    max_members: int,
    indiv_latent_dim: int,
    seed: int,
) -> torch.Tensor:
    n_samples = int(n_households) * int(max_members)
    original_random_state = getattr(gmm, "random_state", None)
    gmm.random_state = int(seed)
    z_np, _ = gmm.sample(n_samples=n_samples)
    if original_random_state is not None:
        gmm.random_state = original_random_state
    z_np = np.asarray(z_np, dtype=np.float32)
    if z_np.shape != (n_samples, int(indiv_latent_dim)):
        raise ValueError(
            f"Unexpected sampled GMM shape {z_np.shape}, expected {(n_samples, int(indiv_latent_dim))}"
        )
    z_np = z_np.reshape(int(n_households), int(max_members), int(indiv_latent_dim))
    return torch.tensor(z_np, dtype=torch.float32)


def resolve_version_dir(
    log_dir: Path, name: str, version_name: Optional[str] = None
) -> Path:
    experiment_dir = Path(log_dir) / name
    if version_name:
        candidate = experiment_dir / version_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Version directory not found: {candidate}"
        )

    versions = [d for d in experiment_dir.iterdir() if d.is_dir()]
    if not versions:
        raise FileNotFoundError(
            f"No versions found under {experiment_dir}"
        )
    return max(versions, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint_path(
    checkpoints_dir: Path,
    ckpt: Optional[str] = None,
    ckpt_path: Optional[str] = None,
) -> Path:
    if ckpt_path:
        candidate = Path(ckpt_path)
        if candidate.is_file():
            return candidate
        candidate = checkpoints_dir / ckpt_path
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = ckpt or "last"
    ckpt_lower = ckpt.lower()

    if ckpt_lower in {"last", "latest"}:
        last_files = sorted(
            checkpoints_dir.glob("last*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if last_files:
            return last_files[0]

    if ckpt_lower == "best":
        non_last = [
            p
            for p in checkpoints_dir.glob("*.ckpt")
            if not p.name.startswith("last")
        ]
        non_last = sorted(
            non_last, key=lambda p: p.stat().st_mtime, reverse=True
        )
        if non_last:
            return non_last[0]

    candidate = checkpoints_dir / ckpt
    if candidate.is_file():
        return candidate

    all_ckpts = sorted(
        checkpoints_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if all_ckpts:
        return all_ckpts[0]

    raise FileNotFoundError(
        f"No checkpoints found under {checkpoints_dir}"
    )


def build_predict_trainer(config: dict) -> Trainer:
    trainer_config = config.get("trainer_params", {}).copy()
    allowed_keys = {
        "accelerator",
        "devices",
        "precision",
        "log_every_n_steps",
        "enable_progress_bar",
    }
    predict_config = {
        key: value
        for key, value in trainer_config.items()
        if key in allowed_keys
    }
    return Trainer(
        logger=False,
        enable_checkpointing=False,
        **predict_config,
    )


def prepare_synthetic_population(
    synthetic_attributes: Optional[DataFrame],
    attribute_encoder: Optional[BaseLabelEncoder],
    encoded_schedules: BaseDataset,
    input_schedules: DataFrame,
    verbose: bool = False,
):
    # CAS 1 : Architecture Standard (Encoder séparé défini dans le YAML)
    # Dans ce cas, les attributs (âge, sexe, emploi...) sont encodés par un label_encoder
    # indépendant (onehot ou tokens), configuré dans la section 'labels_encoder' du YAML.
    # C'est le cas classique pour les modèles individuels (CVAEContLSTM, etc.)
    if synthetic_attributes is not None and attribute_encoder is not None:
        # Sanitize synthetic attributes against training categories for token encoder
        if attribute_encoder.__class__.__name__ == "TokenAttributeEncoder":
            synthetic_attributes = synthetic_attributes.copy()
            for col, cfg in attribute_encoder.config.items():
                if col not in synthetic_attributes.columns:
                    continue
                nominal_map = cfg.get("nominal")
                if nominal_map is None:
                    continue

                # Work on strings to avoid pandas dtype warnings
                series = synthetic_attributes[col].astype(str)
                valid_keys = set(str(k) for k in nominal_map.keys())

                # Try numeric clamp if mapping keys look numeric
                numeric_keys = []
                for key in valid_keys:
                    try:
                        numeric_keys.append(int(float(key)))
                    except (TypeError, ValueError):
                        continue

                if numeric_keys:
                    min_val = min(numeric_keys)
                    max_val = max(numeric_keys)
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    invalid_mask = numeric_series.notna() & (
                        (numeric_series < min_val) | (numeric_series > max_val)
                    )
                    if invalid_mask.any():
                        clamped = numeric_series.clip(lower=min_val, upper=max_val)
                        series.loc[numeric_series.notna()] = (
                            clamped.loc[numeric_series.notna()]
                            .round()
                            .astype(int)
                            .astype(str)
                        )
                        count = int(invalid_mask.sum())
                        print(
                            f"Warning: clamped {count} synthetic values for '{col}' "
                            f"to [{min_val}, {max_val}] to match training categories."
                        )
                else:
                    invalid_mask = ~series.isin(valid_keys)
                    if invalid_mask.any():
                        count = int(invalid_mask.sum())
                        print(
                            f"Warning: replaced {count} synthetic values for '{col}' "
                            "with 'PAD' to match training categories."
                        )
                        series.loc[invalid_mask] = "PAD"

                synthetic_attributes[col] = series

        synthetic_population, _ = attribute_encoder.encode(
            synthetic_attributes
        )
        return synthetic_population

    # CAS 2 : Architecture HOUSEHOLD (Encoder intégré au Dataset)
    # Ici, les attributs sont encodés directement par le HouseholdEncoder,
    # pas par un label_encoder séparé. Le Dataset contient les dictionnaires d'encodage
    # (label_dics) construits pendant l'entraînement. On les réutilise pour encoder
    # les attributs de la population synthétique fournie en CSV.
    # Pourquoi ce cas existe : le modèle Household a besoin d'encoder les attributs
    # de TOUS les membres d'un ménage dans un tenseur 3D [N_ménages, max_members, N_attrs],
    # ce qui est incompatible avec les label_encoders individuels classiques.
    if synthetic_attributes is not None and hasattr(encoded_schedules, "label_dics"):
        print("[Household Mode] Encodage des attributs synthétiques")

        # on récupère les outils du dataset déjà instancié lors de l'entraînement
        # pour garantir la cohérence des mappings (même index = même catégorie)
        label_dics = encoded_schedules.label_dics
        label_cols = encoded_schedules.label_cols
        max_members = encoded_schedules.max_members
        group_col = encoded_schedules.group_col  # 'hid'

        # on regroupe le CSV synthétique par ménage (hid) pour construire les tenseurs 3D
        synthetic_attributes[group_col] = synthetic_attributes[group_col].astype(
            str
        )
        grouped = synthetic_attributes.groupby(group_col, sort=False)

        # liste pour stocker les tenseurs de chaque ménage
        hh_tensors = []

        # pour chaque ménage dans le fichier synthétique
        for hid, group in grouped:
            # création de la grille vide pour ce ménage [max_members, N_Attributs]
            # les membres non remplis resteront à 0 (= PAD token)
            hh_grid = torch.zeros(
                (max_members, len(label_cols)), dtype=torch.long
            )

            # tri par âge (desc), puis sexe, pour rester cohérent avec l'ordre du dataset
            # d'entraînement. C'est important car le modèle a appris la position des membres
            # dans cet ordre (le plus âgé en premier)
            members = group.copy()
            if "age" in members.columns:
                members = members.assign(
                    _age_sort=pd.to_numeric(members["age"], errors="coerce").fillna(-1)
                )
                if "sex" in members.columns:
                    members = members.assign(
                        _sex_sort=members["sex"].astype(str)
                    ).sort_values(
                        by=["_age_sort", "_sex_sort"],
                        ascending=[False, True],
                        kind="stable",
                    )
                else:
                    members = members.sort_values(
                        by=["_age_sort"], ascending=[False], kind="stable"
                    )

            # on prend les 5 premiers membres max apres tri
            members = members.iloc[:max_members]

            # remplissage
            for i in range(len(members)):
                member_row = members.iloc[i]
                for j, col in enumerate(label_cols):
                    raw_val = str(member_row[col])
                    if (
                        col == "age"
                        and getattr(encoded_schedules, "discretize_age", False)
                    ):
                        raw_val = bin_age_value(
                            raw_val,
                            getattr(
                                encoded_schedules, "age_bins", DEFAULT_AGE_BINS
                            ),
                        )
                    # on utilise le dictionnaire appris pendant le Training
                    mapping = label_dics[col]
                    hh_grid[i, j] = mapping.get(raw_val, 0)  # 0 = PAD

            hh_tensors.append(hh_grid)

        # on empile tout -> [N_Menages, 5, N_Attributs]
        if hh_tensors:
            synthetic_population = torch.stack(hh_tensors)
            print(
                f">>> Attributs encodés avec succès : {synthetic_population.shape}"
            )
            return synthetic_population

    if verbose:
        print(
            "Warning: No attribute encoder found, generating unconditional samples."
        )
    return input_schedules.pid.nunique()



def run_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
) -> None:
    """Point d'entrée principal pour un run complet : entraînement + inférence + génération + évaluation.

    C'est la commande 'caveat run' de la CLI. Elle enchaîne toutes les étapes du pipeline :
      1. Chargement des données brutes (load_data)
      2. Encodage des labels puis des schedules (encode_input_labels → encode_schedules)
      3. Entraînement du VAE (train → experiment.py gère les epochs via Lightning)
      4. Test d'inférence : reconstruction des schedules observés (test_inference)
      5. Génération d'une population synthétique (generate)
      6. Évaluation comparative réel vs synthétique (evaluate_synthetics)

    Pourquoi cette fonction orchestre tout : elle garantit que chaque étape utilise
    les mêmes encodeurs, le même seed, et le même logger pour la traçabilité.
    """
    _validate_gmm_config_migration(config)
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    version_name = logger_params.get("version_name", None)
    logger = initiate_logger(log_dir, name, version_name)
    seed = config.pop("seed", seeder())

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_labels, label_weights = encode_input_labels(
        logger.log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        logger.log_dir, input_schedules, encoded_labels, label_weights, config
    )

    # train
    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        label_encoder=attribute_encoder,
        config=config,
        test=test,
        gen=gen,
        logger=logger,
        seed=seed,
    )

    if test:
        # test the model
        test_path = Path(f"{logger.log_dir}/test")
        test_path.mkdir(exist_ok=True, parents=True)
        run_test(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            write_dir=test_path,
            seed=seed,
        )

    if infer:
        test_infer_path = Path(f"{logger.log_dir}/test_inference")
        test_infer_path.mkdir(exist_ok=True, parents=True)

        test_inference(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            write_dir=test_infer_path,
            seed=seed,
        )
    if gen:
        synthetic_population = prepare_synthetic_population(
            synthetic_attributes=synthetic_attributes,
            attribute_encoder=attribute_encoder,
            encoded_schedules=encoded_schedules,
            input_schedules=input_schedules,
            verbose=verbose,
        )

        # generate synthetic schedules
        gen_output = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                config=config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
        synthetic_schedules = gen_output[0] # Le premier élément est toujours les schedules
    
            # evaluate synthetic schedules
            # Note: pour run_command, on passe directement les dictionnaires
        evaluate_synthetics(
                synthetic_schedules={name: synthetic_schedules},
                synthetic_labels={name: synthetic_attributes},
                default_eval_schedules=input_schedules,
                default_eval_attributes=input_attributes,
                write_path=Path(logger.log_dir),
                eval_params=config.get("evaluation_params", {}),
                stats=False, # Souvent False pour un run unique
                verbose=verbose,
            )
 

def _fit_gmm_prior_from_inference(model, test_infer_dir: Path) -> bool:
    """Load z_indiv samples saved by test-inference and fit a GMM on the model.

    Reads ``zs.csv`` (shape [B, M*D], reparameterised z_indiv) and ``mask.csv``
    (shape [B, M]) from *test_infer_dir*, extracts the active-member rows, and
    legacy helper no longer used.

    Returns True if the GMM was fitted successfully, False otherwise.
    """
    raise RuntimeError(
        "Deprecated helper `_fit_gmm_prior_from_inference` should not be called."
    )
    if not getattr(model, "use_hierarchical", False):
        print("[GMM] Skipping: model is not hierarchical.")
        return False

    zs_path = test_infer_dir / "zs.csv"
    mask_path = test_infer_dir / "mask.csv"

    if not zs_path.exists():
        print(f"[GMM] Warning: {zs_path} not found — cannot fit GMM prior.")
        return False
    if not mask_path.exists():
        print(f"[GMM] Warning: {mask_path} not found — cannot fit GMM prior.")
        return False

    print(f"[GMM] Loading z_indiv samples from {zs_path} …")
    zs_np = pd.read_csv(zs_path, header=None).values      # [B, M*D]
    mask_np = pd.read_csv(mask_path, header=None).values  # [B, M]

    indiv_dim = model.indiv_latent_dim
    M = model.max_members
    B = zs_np.shape[0]

    # Unflatten [B, M*D] → [B, M, D]
    zs_3d = zs_np.reshape(B, M, indiv_dim)

    # on garde seulement les membres actifs (les masqués sont des fantômes)
    mask_bool = mask_np > 0.5          # [B, M]
    z_active = zs_3d[mask_bool]        # [N_active, D]

    z_tensor = torch.tensor(z_active, dtype=torch.float32)
    raise RuntimeError(
        "Deprecated helper `_fit_gmm_prior_from_inference` should not be called."
    )
    return True


def run_checkpoint_command(
    config: dict,
    ckpt: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    version_name: Optional[str] = None,
    output_tag: Optional[str] = None,
    verbose: bool = False,
    gen: bool = True,
    infer: bool = True,
    eval: bool = True,
) -> None:
    """Reprend un modèle existant depuis un checkpoint pour inférence/génération/évaluation.

    Pourquoi cette commande existe : l'entraînement d'un VAE peut prendre des heures.
    Une fois le modèle entraîné, on veut pouvoir re-générer avec différents paramètres
    (autre population cible, autre seed, GMM activé...) sans ré-entraîner.
    Supporte aussi le sampling avec GMM (sample_with_train_gmm) pour les modèles hiérarchiques :
    on infère d'abord les latents sur le train, on ajuste un GMM, puis on échantillonne
    depuis ce GMM au lieu du prior N(0,I) standard.
    """
    _validate_gmm_config_migration(config)
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )

    if version_name is None:
        version_name = logger_params.get("version_name", None)

    version_dir = resolve_version_dir(log_dir, name, version_name)
    checkpoints_dir = version_dir / "checkpoints"
    ckpt_file = resolve_checkpoint_path(
        checkpoints_dir, ckpt=ckpt, ckpt_path=ckpt_path
    )

    seed = config.get("seed", seeder())
    checkpoint_params = config.get("checkpoint_params", {})
    sample_with_train_gmm = bool(checkpoint_params.get("sample_with_train_gmm", False))
    gmm_n_components = int(checkpoint_params.get("gmm_n_components", 20))
    if gmm_n_components <= 0:
        raise ValueError(f"`checkpoint_params.gmm_n_components` must be > 0, got {gmm_n_components}")

    if output_tag is None:
        output_tag = f"postprocess_{ckpt_file.stem}"
    output_dir = version_dir / output_tag
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Using checkpoint: {ckpt_file}")
    print(f"Outputs will be written to: {output_dir}")

    apply_performance_params(config)

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_labels, label_weights = encode_input_labels(
        output_dir, input_attributes, config
    )
    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        output_dir, input_schedules, encoded_labels, label_weights, config
    )

    # load model and trainer
    model = load_model(ckpt_file, config)
    trainer = build_predict_trainer(config)
    if sample_with_train_gmm and not getattr(model, "use_hierarchical", False):
        raise ValueError(
            "`checkpoint_params.sample_with_train_gmm=true` requires a hierarchical model."
        )

    if infer:
        test_infer_path = output_dir / "test_inference"
        test_infer_path.mkdir(exist_ok=True, parents=True)
        test_inference(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            write_dir=test_infer_path,
            seed=seed,
            ckpt_path=str(ckpt_file),
            dataloaders=data_loader.test_dataloader(),
            datamodule=data_loader,
            model=model,
        )

    # Le GMM (Gaussian Mixture Model) est une alternative au prior standard N(0,I).
    # Pourquoi : le prior gaussien standard est trop simpliste pour les modèles hiérarchiques
    # où chaque membre d'un ménage a son propre z_indiv. En ajustant un GMM sur les mu_indiv
    # de l'entraînement, on capture la vraie structure multi-modale de l'espace latent
    # (ex: clusters d'enfants vs adultes vs personnes âgées). Ça permet de tester
    # la capacité réelle du decoder avec un prior plus fidèle aux données.

    synthetic_schedules = None
    if gen:
        synthetic_population = prepare_synthetic_population(
            synthetic_attributes=synthetic_attributes,
            attribute_encoder=attribute_encoder,
            encoded_schedules=encoded_schedules,
            input_schedules=input_schedules,
            verbose=verbose,
        )

        z_override = None
        z_override_kind = None
        if sample_with_train_gmm:
            train_infer_path = output_dir / "train_inference"
            train_infer_path.mkdir(exist_ok=True, parents=True)
            train_loader = _build_train_inference_dataloader(data_loader, config)
            test_inference(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=train_infer_path,
                seed=seed,
                ckpt_path=str(ckpt_file),
                dataloaders=train_loader,
                datamodule=None,
                model=model,
            )
            gmm = _fit_train_mu_indiv_gmm(
                train_infer_dir=train_infer_path,
                model=model,
                n_components=gmm_n_components,
                write_dir=output_dir,
                seed=42,
            )
            n_households = (
                int(synthetic_population)
                if isinstance(synthetic_population, int)
                else int(len(synthetic_population))
            )
            z_override = _sample_z_indiv_from_gmm(
                gmm=gmm,
                n_households=n_households,
                max_members=int(getattr(model, "max_members")),
                indiv_latent_dim=int(getattr(model, "indiv_latent_dim")),
                seed=int(seed),
            )
            z_override_kind = "z_indiv"
            z_override_np = z_override.detach().cpu().numpy().reshape(n_households, -1)
            DataFrame(z_override_np).to_csv(
                output_dir / "synthetic_zs_indiv_gmm.csv", index=False, header=False
            )
            print(
                f"[GMM] Sampled z_indiv overrides saved to {output_dir / 'synthetic_zs_indiv_gmm.csv'}"
            )

        gen_output = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            config=config,
            write_dir=output_dir,
            seed=seed,
            ckpt_path=str(ckpt_file),
            model=model,
            z_override=z_override,
            z_override_kind=z_override_kind,
        )
        synthetic_schedules = gen_output[0]

    if eval:
        if synthetic_schedules is None:
            print(
                "Warning: Evaluation requested but no synthetic schedules were generated."
            )
        else:
            evaluate_synthetics(
                synthetic_schedules={name: synthetic_schedules},
                synthetic_labels={name: synthetic_attributes},
                default_eval_schedules=input_schedules,
                default_eval_attributes=input_attributes,
                write_path=output_dir,
                eval_params=config.get("evaluation_params", {}),
                stats=False,
                verbose=verbose,
            )


def batch_command(
    batch_config: dict,
    stats: bool = True,
    verbose: bool = False,
    test: bool = False,
    infer: bool = True,
    gen: bool = True,
) -> None:
    """
    Runs a batch of training and reporting runs based on the provided configuration.

    Args:
        batch_config (dict[dict]): A dictionary containing the configuration for each training job.

    Returns:
        None
    """
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs"), name)

    synthetic_schedules = {}
    synthetic_attributes_all = {}

    for name, config in batch_config.items():
        name = str(name)
        logger = initiate_logger(log_dir, name)

        # build config for this run
        combined_config = global_config.copy()
        combined_config.update(config)
        seed = combined_config.pop("seed", seeder())

        # load data
        input_schedules, input_attributes, synthetic_attributes = load_data(
            combined_config
        )

        # encode data
        attribute_encoder, encoded_labels, label_weights = encode_input_labels(
            logger.log_dir, input_attributes, combined_config
        )

        schedule_encoder, encoded_schedules, data_loader = encode_schedules(
            logger.log_dir,
            input_schedules,
            encoded_labels,
            label_weights,
            combined_config,
        )

        # train
        trainer = train(
            name=name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            label_encoder=attribute_encoder,
            config=combined_config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
        )
        if test:
            # test the model
            run_test(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
        if infer:
            test_infer_path = Path(f"{logger.log_dir}/test_inference")
            test_infer_path.mkdir(exist_ok=True, parents=True)

            test_inference(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=test_infer_path,
                seed=seed,
            )
        if gen:
            # prepare synthetic attributes
            if synthetic_attributes is not None:
                synthetic_population, _ = attribute_encoder.encode(
                    synthetic_attributes
                )
            else:
                synthetic_population = input_schedules.pid.nunique()

            # record synthetic attributes for evaluation
            synthetic_attributes_all[name] = synthetic_attributes

            # generate synthetic schedules
            synthetic_schedules[name] = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                config=combined_config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )[0]
    if gen:
        # evaluate synthetic schedules
        evaluate_synthetics(
            synthetic_schedules=synthetic_schedules,
            synthetic_labels=synthetic_attributes_all,
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_attributes,
            write_path=log_dir,
            eval_params=global_config.get("evaluation_params", {}),
            stats=stats,
            verbose=verbose,
        )


def nrun_command(
    config: dict,
    n: int = 5,
    stats: bool = True,
    verbose: bool = False,
    test: bool = False,
    infer: bool = True,
    gen: bool = True,
) -> None:
    """
    Repeat a single model training while varying the seed.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        n (int, optional): The number of times to repeat the run. Defaults to 5.
    """
    logger_params = config.get("logging_params", {})

    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs")) / name
    log_dir.mkdir(exist_ok=True, parents=True)

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_labels, label_weights = encode_input_labels(
        log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        log_dir, input_schedules, encoded_labels, label_weights, config
    )

    synthetic_schedules = {}
    all_synthetic_attributes = {}

    for i in range(n):
        run_name = f"{name}_nrun{i}"
        logger = initiate_logger(log_dir, run_name)
        seed = seeder()
        trainer = train(
            name=run_name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            label_encoder=attribute_encoder,
            config=config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
        )
        if test:
            run_test(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )
        if infer:
            test_infer_path = Path(f"{logger.log_dir}/test_inference")
            test_infer_path.mkdir(exist_ok=True, parents=True)

            test_inference(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=test_infer_path,
                seed=seed,
            )
        if gen:
            # prepare synthetic attributes
            if synthetic_attributes is not None:
                synthetic_population, _ = attribute_encoder.encode(
                    synthetic_attributes
                )
            else:
                synthetic_population = input_schedules.pid.nunique()

            all_synthetic_attributes[run_name] = synthetic_attributes

            synthetic_schedules[run_name] = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                config=config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )[0]

    if gen:
        evaluate_synthetics(
            synthetic_schedules=synthetic_schedules,
            synthetic_labels=all_synthetic_attributes,
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_attributes,
            write_path=log_dir,
            eval_params=config.get("evaluation_params", {}),
            stats=stats,
            verbose=verbose,
        )


def ngen_command(
    config: dict,
    n: int = 5,
    infer: bool = True,
    stats: bool = False,
    verbose: bool = False,
) -> None:
    """
    Repeat a single run with multiple samples varying the seed.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        n (int, optional): The number of times to repeat the run. Defaults to 5.
    """
    logger_params = config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs"))
    training_logger = initiate_logger(log_dir, name)

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    attribute_encoder, encoded_labels, label_weights = encode_input_labels(
        log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        log_dir, input_schedules, encoded_labels, label_weights, config
    )

    seed = config.pop("seed", seeder())

    # train
    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        label_encoder=attribute_encoder,
        config=config,
        test=False,
        gen=True,
        logger=training_logger,
        seed=seed,
    )

    synthetic_schedules = {}
    all_synthetic_attributes = {}

    # prepare synthetic attributes
    if synthetic_attributes is not None:
        synthetic_population = attribute_encoder.encode(synthetic_attributes)
    else:
        synthetic_population = input_schedules.pid.nunique()

    for i in range(n):
        logger = initiate_logger(training_logger.log_dir, f"nsample{i}")
        seed = seeder()
        if infer:
            test_infer_path = Path(f"{logger.log_dir}/test_inference")
            test_infer_path.mkdir(exist_ok=True, parents=True)

            test_inference(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=test_infer_path,
                seed=seed,
            )

        synthetic_schedules[f"nsample{i}"] = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            config=config,
            write_dir=Path(logger.log_dir),
            seed=seed,
        )[0]
        all_synthetic_attributes[f"nsample{i}"] = synthetic_attributes

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=all_synthetic_attributes,
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_attributes,
        write_path=log_dir,
        eval_params=config.get("evaluation_params", {}),
        stats=stats,
        verbose=verbose,
    )


def eval_command(
    config: dict,
    schedules_name: str = "synthetic_schedules.csv",
    labels_name: Optional[str] = None,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    """
    Runs the evaluation process using the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    logger_params = config.get("logging_params")
    log_dir = Path(logger_params.get("log_dir"))
    schedules_name = str(logger_params.get("name"))

    # load data
    input_schedules, input_attributes, synthetic_labels = load_data(
        config, verbose=False
    )

    # get most recent version
    version = sorted([d for d in log_dir.iterdir() if d.is_dir()])[-1]
    outputs_dir = log_dir / version.name
    schedules_path = outputs_dir / schedules_name
    synthetic_schedules = {
        log_dir.name: data.load_and_validate_schedules(schedules_path)
    }
    print(
        f"> Loaded {synthetic_schedules[log_dir.name].pid.nunique()} synthetic schedules from {schedules_path}"
    )

    if labels_name is not None:
        synthetic_labels_path = outputs_dir / labels_name
        synthetic_labels = load_labels(synthetic_labels_path)

    elif "synthetic_labels" in outputs_dir.iterdir():
        synthetic_labels_path = outputs_dir / "synthetic_labels.csv"
        synthetic_labels = load_labels(synthetic_labels_path)

    elif "synthetic_attributes" in outputs_dir.iterdir():
        synthetic_labels_path = outputs_dir / "synthetic_attributes.csv"
        synthetic_labels = load_labels(synthetic_labels_path)

    synthetic_labels = {log_dir.name: synthetic_labels}

    # evaluate synthetic schedules
    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_attributes,
        write_path=log_dir,
        eval_params=config.get("evaluation_params", {}),
        stats=stats,
        verbose=verbose,
    )


def load_labels(path):
    synthetic_attributes = pd.read_csv(path)
    print(f"> Loaded {len(synthetic_attributes)} synthetic labels from {path}")
    if synthetic_attributes.empty:
        raise UserWarning(f"No labels found in {path}.")
    return synthetic_attributes


def batch_eval_command(
    batch_config: dict,
    schedules_name: str = "synthetic_schedules.csv",
    labels_name: Optional[str] = None,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    """
    Runs a batch of evaluation based on the provided configuration.

    Args:
        batch_config (dict[dict]): A dictionary containing the configuration for each training job.

    Returns:
        None
    """
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params")
    name = str(logger_params.get("name"))
    batch_dir = Path(logger_params.get("log_dir"), name)

    synthetic_schedules_all = {}
    synthetic_labels_all = {}

    for name, config in batch_config.items():
        name = str(name)
        print(f"\n======= Loading {name} =======")
        log_dir = batch_dir / name

        # build config for this run
        combined_config = global_config.copy()
        combined_config.update(config)

        # load data
        input_schedules, input_labels, synthetic_labels = load_data(
            combined_config, verbose=verbose
        )
        print(
            f"> Loaded {input_schedules.pid.nunique()} target schedules for evaluation"
        )
        if input_labels is not None:
            print(
                f"> Loaded {input_labels.pid.nunique()} target labels for evaluation"
            )
        else:
            print("No target labels provided.")

        # get most recent version
        version = sorted([d for d in log_dir.iterdir() if d.is_dir()])[-1]
        outputs_dir = log_dir / version.name
        schedules_path = outputs_dir / schedules_name
        synthetic_schedules_all[
            log_dir.name
        ] = data.load_and_validate_schedules(schedules_path)
        print(
            f"> Loaded {synthetic_schedules_all[log_dir.name].pid.nunique()} synthetic schedules from {schedules_path}"
        )

        synthetic_labels_path = (
            outputs_dir / "synthetic_labels.csv"
        )  # todo: make this consistent across all models
        synthetic_attributes_path = outputs_dir / "synthetic_attributes.csv"
        if labels_name is not None:
            synthetic_labels_path = outputs_dir / labels_name
            synthetic_labels = load_labels(synthetic_labels_path)

        elif synthetic_labels_path.exists():
            synthetic_labels = load_labels(synthetic_labels_path)

        elif synthetic_attributes_path.exists():
            synthetic_labels = load_labels(synthetic_attributes_path)

        synthetic_labels_all[log_dir.name] = synthetic_labels

    # evaluate synthetic schedules
    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules_all,
        synthetic_labels=synthetic_labels_all,
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_labels,
        write_path=batch_dir,
        eval_params=global_config.get("evaluation_params", {}),
        stats=stats,
        verbose=verbose,
    )


def report_command(
    observed_path: Path,
    log_dir: Path,
    name: str = "synthetic_schedules.csv",
    verbose: bool = False,
    head: int = 10,
    batch: bool = False,
    stats: bool = True,
):
    observed_path = Path(observed_path)
    log_dir = Path(log_dir)
    observed = data.load_and_validate_schedules(observed_path)
    synthetic_schedules = {}
    if batch:
        paths = [p for p in log_dir.iterdir() if p.is_dir()]
    else:
        paths = [log_dir]

    for experiment in paths:
        # get most recent version
        version = sorted([d for d in experiment.iterdir() if d.is_dir()])[-1]
        path = experiment / version.name / name
        synthetic_schedules[experiment.name] = data.load_and_validate_schedules(
            path
        )

    reports = evaluate.evaluate(
        target_schedules=observed,
        synthetic_schedules=synthetic_schedules,
        report_stats=stats,
    )
    evaluate.report(reports, log_dir=log_dir, head=head, verbose=verbose)


def load_data(
    config: dict, verbose: bool = False
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Charge les données brutes depuis les CSV configurés dans le YAML.

    Retourne 3 DataFrames :
      - schedules : emplois du temps observés (pid, act, start, end, ...)
      - attributes : attributs socio-démographiques liés aux schedules (âge, sexe, emploi...)
      - synthetic_attributes : attributs de la population cible pour la génération
        (peut être None si on génère sans conditionnement)

    Pourquoi séparer attributes et synthetic_attributes :
      Les attributes servent à entraîner le modèle conditionnel (apprendre la relation
      entre profil socio-démo et emploi du temps). Les synthetic_attributes définissent
      la population qu'on veut générer (ex: 1000 ménages avec telle distribution d'âge).
    """
    # chargement des emplois du temps observés (données d'entraînement)
    schedules_path = Path(config["schedules_path"])
    schedules = data.load_and_validate_schedules(schedules_path)
    if verbose:
        print(
            f"Loaded {schedules.pid.nunique()} schedules from {schedules_path}"
        )

    # load attributes data (conditional case)
    attributes, synthetic_attributes = data.load_and_validate_attributes(
        config, schedules, verbose=verbose
    )
    return schedules, attributes, synthetic_attributes


def encode_schedules(
    log_dir: Path,
    schedules: DataFrame,
    attributes: Optional[Tensor],
    label_weights: Optional[Tuple[Tensor, Tensor]],
    config: dict,
) -> Tuple[BaseEncoder, BaseDataset, DataModule]:
    """Transforme les DataFrames bruts en tenseurs prêts pour l'entraînement.

    Étapes :
      1. Instanciation de l'encodeur depuis encoding/ (ex: 'household', 'discrete', 'continuous')
         selon la clé 'encoder' du YAML. L'encodeur apprend le vocabulaire d'activités.
      2. Appel à encoder.encode() qui produit un Dataset PyTorch contenant :
         - Les tenseurs de schedules encodés (ex: [N, T, C] pour discrete, [N, M, T, C] pour household)
         - Les poids par activité/séquence/label pour la loss pondérée
         - Les labels encodés (passés depuis encode_input_labels ou intégrés au dataset Household)
      3. Construction du DataModule Lightning qui gère le split train/val/test et les DataLoaders

    Pourquoi sérialiser l'encodeur (pickle) : il contient le mapping activités→indices
    appris sur les données d'entraînement, nécessaire pour décoder les sorties du modèle
    lors de la génération ou de l'inférence depuis un checkpoint.
    """
    schedule_encoder = build_encoder(config)
    encoded_schedules = schedule_encoder.encode(
        schedules=schedules, labels=attributes, label_weights=label_weights
    )
    data_loader = build_dataloader(config, encoded_schedules)

    pickle.dump(schedule_encoder, open(f"{log_dir}/schedule_encoder.pkl", "wb"))

    return (schedule_encoder, encoded_schedules, data_loader)


def encode_input_labels(
    log_dir: Path, input_labels: Optional[DataFrame], config: dict
) -> Tuple[BaseEncoder, BaseDataset, DataModule, Tensor]:
    """Encode les attributs socio-démographiques pour le conditionnement du VAE.

    Deux cas possibles selon l'architecture :
      - Mode Standard : un label_encoder séparé (onehot ou tokens) est défini dans
        la section 'labels_encoder' du YAML. Il encode les attributs en tenseurs
        et calcule des poids d'équilibrage (pour gérer le déséquilibre des classes,
        ex: plus de femmes que d'hommes dans l'échantillon).
      - Mode Household : pas de 'labels_encoder' dans le YAML car le HouseholdEncoder
        gère lui-même l'encodage des attributs dans son Dataset. On retourne (None, None, None)
        et les attributs seront encodés plus tard dans encode_schedules().

    Pourquoi sérialiser l'encodeur : les mappings (ex: 'male'→0, 'female'→1)
    doivent être identiques entre entraînement et génération/évaluation.
    """
    attribute_encoder = None
    # encodage optionnel des attributs socio-démographiques
    encoded_attributes = None
    weights = None
    if input_labels is not None:
        encoder_config = config.get("labels_encoder", {})
        if not encoder_config:
            # pas de labels_encoder → on est probablement en mode Household
            # où l'encodage des attributs est intégré au HouseholdEncoder
            print("Warning: Input labels found but no 'labels_encoder' config. Skipping generic label encoding (assuming custom encoder handles it).")
            return None, None, None
        encoder_name = encoder_config.get("name", "onehot")
        labels_config = encoder_config.get("labels", None)
        if labels_config is None:
            raise UserWarning(
                "You have specified input labels, config must contain label encoder configuration with labels defined."
            )
        attribute_encoder = label_encoding.library[encoder_name](
            config=labels_config, **encoder_config
        )
        encoded_attributes, weights = attribute_encoder.encode(input_labels)
        if encoded_attributes is not None:
            attribute_encoder.label_kwargs["labels_size"] = int(
                encoded_attributes.shape[-1]
            )

    pickle.dump(
        attribute_encoder, open(f"{log_dir}/attribute_encoder.pkl", "wb")
    )

    return (attribute_encoder, encoded_attributes, weights)


def apply_performance_params(config: dict) -> None:
    if not cuda_available():
        return

    perf = config.get("performance_params", {})
    torch.set_float32_matmul_precision(perf.get("matmul_precision", "medium"))
    if perf.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if "cudnn_benchmark" in perf:
        torch.backends.cudnn.benchmark = perf["cudnn_benchmark"]
    if "cudnn_deterministic" in perf:
        torch.backends.cudnn.deterministic = perf["cudnn_deterministic"]


def train(
    name: str,
    data_loader: DataModule,
    encoded_schedules: BaseDataset,
    config: dict,
    test: bool,
    gen: bool,
    logger: TensorBoardLogger,
    seed: Optional[int] = None,
    ckpt_path: Optional[Path] = None,
    label_encoder: Optional[BaseLabelEncoder] = None,
) -> Tuple[Trainer, encoding.BaseEncoder]:
    """
    Trains a model on the observed data. Return model trainer (which includes model) and encoder.

    Args:
        name (str): The name of the experiment.
        schedules (pandas.DataFrame): The "observed" population data to train the model on.
        conditionals (pandas.DataFrame): The "conditionals" data to train the model on.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        logger (TensorBoardLogger): Logger.

    Returns:
        Tuple(pytorch.Trainer, BaseEncoder).
    """
    print(f"\n======= Training {name} =======")

    torch.manual_seed(seed)

    apply_performance_params(config)

    torch.cuda.empty_cache()
    if ckpt_path is not None:
        experiment = load_model(ckpt_path, config)
    else:
        label_kwargs = label_encoder.label_kwargs if label_encoder else {}
        experiment = build_model(
            encoded_schedules, config, test, gen, label_kwargs
        )
    trainer = build_trainer(logger, config)
    time = datetime.datetime.now()
    trainer.fit(experiment, datamodule=data_loader)
    duration = datetime.datetime.now() - time
    print(
        f">>> Training took {duration.seconds + duration.microseconds / 1e6} seconds"
    )
    with open(Path(logger.log_dir) / "timer.csv", "a") as fp:
        fp.write(
            f"train,{duration.seconds + duration.microseconds / 1e6},seconds\n"
        )
    return trainer


def run_test(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    print("\n======= Testing =======")
    if ckpt_path is None:
        ckpt_path = "last"

    time = datetime.datetime.now()

    with open(write_dir / f"{ckpt_path}_loss.txt", "w") as fp:
        loss = trainer.test(
            ckpt_path=ckpt_path,
            dataloaders=trainer.datamodule.train_dataloader(),
        )
        fp.write(f"TRAIN: {loss[0]}\n")

        loss = trainer.test(
            ckpt_path=ckpt_path, dataloaders=trainer.datamodule.val_dataloader()
        )
        fp.write(f"VAL: {loss[0]}\n")

        loss = trainer.test(
            ckpt_path=ckpt_path,
            dataloaders=trainer.datamodule.test_dataloader(),
        )
        fp.write(f"TEST: {loss[0]}\n")

        print(f">> Losses saved to {write_dir / f'{ckpt_path}_loss.txt'}")

    duration = datetime.datetime.now() - time
    print(
        f">>> Testing took {duration.seconds + duration.microseconds / 1e6} seconds"
    )
    with open(write_dir / "timer.csv", "a") as fp:
        fp.write(
            f"test,{duration.seconds + duration.microseconds / 1e6},seconds\n"
        )


def test_inference(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
    dataloaders=None,
    datamodule: Optional[DataModule] = None,
    model: Optional[LightningModule] = None,
):
    """Passe d'inférence sur le jeu de test : reconstruit les schedules à partir des observations.

    Pourquoi cette étape est cruciale :
      - Elle mesure la capacité du modèle à RECONSTRUIRE fidèlement les données observées
        (contrairement à generate() qui INVENTE de nouvelles séquences).
      - Elle exporte les espaces latents (zs, mus, stds) pour l'analyse post-hoc :
        visualisation t-SNE, clustering, ajustement GMM pour la génération enrichie.
      - Pour les modèles hiérarchiques (Household), elle exporte aussi les statistiques
        par niveau (coord vs indiv) et les prédictions d'interactions entre membres.

    Sorties sauvegardées dans write_dir :
      - input_schedules.csv : les vrais emplois du temps décodés
      - inferred_schedules.csv : les reconstructions du modèle
      - zs.csv : les échantillons latents (pour analyse ou GMM)
      - mus.csv / stds.csv : moyennes et écarts-types latents (si le modèle les retourne)
      - interaction_probs.csv : probabilités d'interaction prédites (modèles household)
    """
    torch.manual_seed(seed)
    if ckpt_path is None:
        ckpt_path = "last"

    print("\n======= Testing Inference =======")
    if dataloaders is None:
        if datamodule is None:
            datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise ValueError(
                "No dataloaders or datamodule available for inference."
            )
        dataloaders = datamodule.test_dataloader()

    if model is None:
        inference = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    else:
        # si un modèle est déjà fourni en mémoire (ex: après ajustement GMM),
        # on évite de recharger les poids depuis le checkpoint
        inference = trainer.predict(model=model, dataloaders=dataloaders)
    if not inference:
        raise ValueError("Empty inference predictions.")
    batch_outputs = list(zip(*inference))

    # Les 4 premiers éléments sont toujours présents (contrat de l'API predict_step) :
    # [0] input original, [1] reconstruction, [2] latent z, [3] conditionals/labels
    input_schedules = torch.concat(batch_outputs[0])
    inferred_schedules = torch.concat(batch_outputs[1])
    zs = torch.concat(batch_outputs[2])
    conditionals = torch.concat(batch_outputs[3])

    # Détection dynamique des sorties additionnelles (Mu/Logvar/interactions)
    # car chaque architecture retourne un nombre différent de tenseurs
    mus = None
    logvars = None
    mus_coord = None
    logvars_coord = None

    # Format hiérarchique attendu:
    # (x, log_probs, z_indiv, labels, mu_coord, logvar_coord,
    #  interaction_logits, interactions, mask, z_coord, mu_indiv, logvar_indiv)
    if len(batch_outputs) >= 12:
        print(">> Modèle hiérarchique détecté: export stats indiv + coord.")
        mus_coord = torch.concat(batch_outputs[4])
        logvars_coord = torch.concat(batch_outputs[5])
        mus = torch.concat(batch_outputs[10])
        logvars = torch.concat(batch_outputs[11])
    # Format standard:
    # (x, log_probs, z, labels, mu, logvar, ...)
    elif len(batch_outputs) >= 6:
        print(">> Modèle avec sorties statistiques (Mu/Logvar) détecté.")
        mus = torch.concat(batch_outputs[4])
        logvars = torch.concat(batch_outputs[5])
    else:
        print(">> Modèle standard (pas de Mu/Logvar retourné).")

    interaction_logits = None
    interaction_targets = None
    interaction_mask = None
    if len(batch_outputs) >= 9:
        try:
            cand_logits = torch.concat(batch_outputs[6])
            cand_targets = torch.concat(batch_outputs[7])
            cand_mask = torch.concat(batch_outputs[8])
            if (
                cand_logits.ndim == 3
                and cand_targets.ndim == 3
                and cand_mask.ndim == 2
            ):
                interaction_logits = cand_logits
                interaction_targets = cand_targets
                interaction_mask = cand_mask
                print(">> Sorties d'interactions détectées.")
        except Exception as e:
            print(f"Impossible de récupérer les sorties interactions: {e}")

    mask = None
    try:
        mask_loader = dataloaders
        mask = _extract_mask_from_dataloader(mask_loader)
        if mask is not None:
            print(f"Masque récupéré : shape {mask.shape}")
            DataFrame(mask.cpu().numpy()).to_csv(
                Path(write_dir, "mask.csv"), index=False, header=False
            )
    except Exception as e:
        print(f"Impossible de récupérer le masque : {e}")

    input_schedules_df = safe_decode(schedule_encoder, input_schedules, argmax=False, mask=mask)
    input_schedules_df = _ensure_schedule_id_column(input_schedules_df)
    data.validate_schedules(input_schedules_df)
    input_schedules_df.to_csv(write_dir / "input_schedules.csv", index=False)

    inferred_schedules_df = safe_decode(schedule_encoder, inferred_schedules, argmax=True, mask=mask)
    inferred_schedules_df = _ensure_schedule_id_column(inferred_schedules_df)
    data.validate_schedules(inferred_schedules_df)
    inferred_schedules_df.to_csv(write_dir / "inferred_schedules.csv", index=False)

    # Gestion des espaces latents 2D et 3D :
    # - Modèles individuels : z de forme [B, latent_dim] (un vecteur par personne)
    # - Modèles household : z de forme [B, M, latent_dim] (un vecteur par MEMBRE par ménage)
    # On aplatit le 3D en 2D pour l'export CSV, car les outils d'analyse (t-SNE, GMM)
    # travaillent sur des matrices 2D
    zs_np = zs.cpu().numpy()
    if zs_np.ndim == 3:
        # 3D latent [B, M, latent_dim] -> flatten to [B, M*latent_dim]
        B, M, latent_dim = zs_np.shape
        zs_np = zs_np.reshape(B, M * latent_dim)
        print(f"Latent 3D détecté: aplati de [{B}, {M}, {latent_dim}] à [{B}, {M * latent_dim}]")
 
    DataFrame(zs_np).to_csv(
        Path(write_dir, "zs.csv"), index=False, header=False
    )
    if mus is not None:
        print("Sauvegarde des Moyennes (Mu)...")
        mus_np = mus.cpu().numpy()
        if mus_np.ndim == 3:
            B, M, latent_dim = mus_np.shape
            mus_np = mus_np.reshape(B, M * latent_dim)
        
        DataFrame(mus_np).to_csv(Path(write_dir, "mus.csv"), index=False, header=False)

    # Sauvegarde Std (seulement si dispo)
    if logvars is not None:
        print("Sauvegarde des Ecart-types (Std)...")
        # Calcul de l'écart-type : exp(0.5 * logvar)
        stds_tensor = torch.exp(0.5 * logvars)
        stds_np = stds_tensor.cpu().numpy()
        
        if stds_np.ndim == 3:
            B, M, latent_dim = stds_np.shape
            stds_np = stds_np.reshape(B, M * latent_dim)

        DataFrame(stds_np).to_csv(Path(write_dir, "stds.csv"), index=False, header=False)

    if mus_coord is not None:
        print("Sauvegarde des Moyennes coordination (Mu_coord)...")
        mus_coord_np = mus_coord.cpu().numpy()
        if mus_coord_np.ndim == 3:
            B, M, latent_dim = mus_coord_np.shape
            mus_coord_np = mus_coord_np.reshape(B, M * latent_dim)
        DataFrame(mus_coord_np).to_csv(
            Path(write_dir, "mus_coord.csv"), index=False, header=False
        )

    if logvars_coord is not None:
        print("Sauvegarde des Ecart-types coordination (Std_coord)...")
        stds_coord_tensor = torch.exp(0.5 * logvars_coord)
        stds_coord_np = stds_coord_tensor.cpu().numpy()
        if stds_coord_np.ndim == 3:
            B, M, latent_dim = stds_coord_np.shape
            stds_coord_np = stds_coord_np.reshape(B, M * latent_dim)
        DataFrame(stds_coord_np).to_csv(
            Path(write_dir, "stds_coord.csv"), index=False, header=False
        )

    if interaction_logits is not None:
        interaction_probs = torch.sigmoid(interaction_logits)
        DataFrame(
            interaction_probs.cpu().numpy().reshape(interaction_probs.shape[0], -1)
        ).to_csv(
            Path(write_dir, "interaction_probs.csv"), index=False, header=False
        )
        DataFrame(
            interaction_targets.cpu().numpy().reshape(interaction_targets.shape[0], -1)
        ).to_csv(
            Path(write_dir, "interaction_targets.csv"), index=False, header=False
        )
        DataFrame(interaction_mask.cpu().numpy()).to_csv(
            Path(write_dir, "interaction_mask.csv"), index=False, header=False
        )
        image_paths = _save_interaction_overview_image(
            write_dir=write_dir,
            interaction_probs=interaction_probs,
            interaction_targets=interaction_targets,
            interaction_mask=interaction_mask,
        )
        if image_paths:
            print(f"Images interactions sauvegardées: {len(image_paths)} dans {image_paths[0].parent}")

    attributes_df = None
    if attribute_encoder is not None:
        attributes_df = attribute_encoder.decode(conditionals)
        attributes_df.to_csv(write_dir / "input_attributes.csv", index=False)
    elif isinstance(conditionals, torch.Tensor) and hasattr(schedule_encoder, 'decode_attributes'):
        # utiliser decode_attributes du HouseholdEncoder
        attributes_df = safe_decode_attributes(schedule_encoder, conditionals, mask=mask)
        if attributes_df is not None:
            attributes_df.to_csv(write_dir / "input_attributes.csv", index=False)


    if attributes_df is not None and not attributes_df.empty:
        try:
            print(" Merge automatique des inferred_schedules et attributs...")

            # Identifier les colonnes de jointure
            join_cols = []
            if 'hid' in inferred_schedules_df.columns and 'hid' in attributes_df.columns:
                join_cols.append('hid')
            if 'pid' in inferred_schedules_df.columns and 'pid' in attributes_df.columns:
                join_cols.append('pid')
            if 'member_idx' in inferred_schedules_df.columns and 'member_idx' in attributes_df.columns:
                join_cols.append('member_idx')

            if join_cols:
                #  inferred schedules + attributes
                merged_inferred = pd.merge(inferred_schedules_df, attributes_df, on=join_cols, how='left')
                output_path = write_dir / "inferred_schedules_with_attributes.csv"
                merged_inferred.to_csv(output_path, index=False)
                print(f" Fichier mergé créé : {output_path}")
                print(f"   {len(merged_inferred)} lignes, {len(merged_inferred.columns)} colonnes")

                #  input_schedules + attributes
                merged_input = pd.merge(input_schedules_df, attributes_df, on=join_cols, how='left')
                output_path_input = write_dir / "input_schedules_with_attributes.csv"
                merged_input.to_csv(output_path_input, index=False)
                print(f"Fichier input mergé créé : {output_path_input}")
            else:
                print(" Pas de colonnes de jointure communes, merge impossible")
        except Exception as e:
            print(f"Erreur lors du merge automatique : {e}")
            import traceback
            traceback.print_exc()


def generate(
    trainer: Trainer,
    population: Union[int, Tensor],
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
    model: Optional[LightningModule] = None,
    z_override: Optional[torch.Tensor] = None,
    z_override_kind: Optional[str] = None,
) -> DataFrame:
    """Génère de nouveaux emplois du temps synthétiques à partir du modèle entraîné.

    Deux modes de fonctionnement :
      - population = int : génération INCONDITIONNELLE de N schedules (échantillonnage z ~ N(0,I))
      - population = Tensor : génération CONDITIONNELLE à partir d'attributs socio-démo encodés
        (le modèle utilise les labels pour guider la génération via le decoder conditionnel)

    z_override permet de remplacer l'échantillonnage standard par des latents pré-calculés
    (ex: depuis un GMM ajusté sur les données d'entraînement) pour les modèles hiérarchiques.

    Retourne (synthetic_schedules_df, synthetic_attributes, zs).
    Sauvegarde aussi : synthetic_schedules.csv, synthetic_zs.csv, synthetic_interactions.csv
    """
    torch.manual_seed(seed)
    if ckpt_path is None:
        ckpt_path = "last"
    latent_dims = config.get("model_params", {}).get("latent_dim")
    if latent_dims is None:
        latent_dims = config.get("experiment_params", {}).get("latent_dims", 2)
        # default of 2
    latent_categories = config.get("model_params", {}).get("latent_categories")
    batch_size = config.get("generator_params", {}).get("batch_size", 256)
    gantt_cfg = config.get("generator_params", {}).get("interaction_gantt", {})
    gantt_enabled = bool(gantt_cfg.get("enabled", False))
    gantt_num_households = int(gantt_cfg.get("num_households", 10))

    generated_interaction_logits = None
    generated_interaction_mask = None
    attributes_df = None
    gantt_labels_df = None

    if isinstance(population, int):
        print(f"\n======= Sampling {population} new schedules =======")
        synthetic_schedules, zs, generated_interaction_logits, generated_interaction_mask = generate_n(
            trainer,
            n=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
            write_dir=write_dir,
            latent_categories=latent_categories,
            model=model,
            z_override=z_override,
            z_override_kind=z_override_kind,
        )
        synthetic_attributes = None
    elif isinstance(population, Tensor):
        print(
            f"\n======= Sampling {len(population)} new schedules from synthetic attributes ======="
        )
        (
            synthetic_attributes,
            synthetic_schedules,
            zs,
            generated_interaction_logits,
            generated_interaction_mask,
        ) = generate_from_attributes(
            trainer,
            attributes=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
            write_dir=write_dir,
            cats=latent_categories,
            model=model,
            z_override=z_override,
            z_override_kind=z_override_kind,
        )
        if attribute_encoder is not None:
            synthetic_attributes = attribute_encoder.decode(synthetic_attributes)
            synthetic_attributes.to_csv(write_dir / "synthetic_attributes.csv", index=False)
        else:
            # Mode Household : on va décoder les attributs plus tard avec le masque
            if isinstance(synthetic_attributes, torch.Tensor):
                # récupère les dimensions
                n_households, n_members, n_attribs = synthetic_attributes.shape

                attr_np = synthetic_attributes.detach().cpu().numpy()

                # on met tous les membres les uns à la suite des autres
                flat_attr = attr_np.reshape(-1, n_attribs)

                # Création du DataFrame
                df = pd.DataFrame(flat_attr)

                hids = [i for i in range(n_households) for _ in range(n_members)]
                m_ids = [j for _ in range(n_households) for j in range(n_members)]

                df.insert(0, "member_idx", m_ids)
                df.insert(0, "hid", hids)


                print(f"Sauvegarde des attributs bruts (encodés) : {df.shape}")
                df.to_csv(write_dir / "synthetic_attributes_raw.csv", index=False)

    # Reconstruire le masque depuis les schedules générés (avant decode).
    # Pourquoi : les modèles Household génèrent des tenseurs paddés à max_members,
    # mais certains ménages ont moins de membres. Les positions non-utilisées
    # contiennent des activités PAD (index 0). On crée un masque binaire [B, M]
    # pour identifier les membres réels (somme d'activités > 0) vs fantômes.
    # Sans ce masque, le décodage inclurait des lignes de bruit dans le CSV final.
    mask = None
    if isinstance(synthetic_schedules, torch.Tensor):
        # Pour HouseholdVAE : shape [B, M, T] ou [B, M, T, Classes]
        if synthetic_schedules.dim() == 4:  # [B, M, T, Classes]
            # Prendre l'argmax pour avoir [B, M, T]
            sched_indices = torch.argmax(synthetic_schedules, dim=-1)
        else:
            sched_indices = synthetic_schedules

        # Créer masque : membre réel si sum(activités) > 0 (pas que du PAD)
        mask = (sched_indices.sum(dim=-1) > 0).float()  # [B, M]
        print(f"Masque reconstruit depuis schedules : {mask.shape}, {mask.sum().item():.0f} membres actifs")

    # Décoder les schedules
    synthetic_schedules_df = safe_decode(schedule_encoder, synthetic_schedules, argmax=True, mask=mask)
    synthetic_schedules_df = _ensure_schedule_id_column(synthetic_schedules_df)
    data.validate_schedules(synthetic_schedules_df)
    synthetic_schedules_df.to_csv(write_dir / "synthetic_schedules.csv", index=False)

    interaction_eval_params = (
        config.get("baseline_evaluation", {})
        .get("interaction_overlap_evaluation", {})
    )
    inter_threshold = float(interaction_eval_params.get("prob_threshold", 0.5))
    synthetic_interactions_df = _build_synthetic_interactions_from_logits(
        schedules_df=synthetic_schedules_df,
        interaction_logits=generated_interaction_logits,
        interaction_mask=generated_interaction_mask,
        prob_threshold=inter_threshold,
    )
    synthetic_interactions_path = write_dir / "synthetic_interactions.csv"
    synthetic_interactions_df.to_csv(synthetic_interactions_path, index=False)
    print(
        f"Synthetic interactions exported: {synthetic_interactions_path} "
        f"({len(synthetic_interactions_df)} intervals)"
    )

    # sauvegarder les latents
    zs_np = zs.cpu().numpy()
    if zs_np.ndim == 3:
        # 3D latent [B, M, latent_dim] -> flatten to [B, M*latent_dim]
        B, M, latent_dim = zs_np.shape
        zs_np = zs_np.reshape(B, M * latent_dim)
        print(f"Latent 3D détecté: aplati de [{B}, {M}, {latent_dim}] à [{B}, {M * latent_dim}]")
 
    DataFrame(zs_np).to_csv(
        Path(write_dir, "synthetic_zs.csv"), index=False, header=False
    )

    # Merger automatiquement schedules + attributes si possible
    if synthetic_attributes is not None:
        try:
            if isinstance(synthetic_attributes, pd.DataFrame):
                attributes_df = synthetic_attributes
            elif isinstance(synthetic_attributes, torch.Tensor) and hasattr(schedule_encoder, 'decode_attributes'):
                print("📝 Décodage des attributs synthétiques...")
                attributes_df = safe_decode_attributes(schedule_encoder, synthetic_attributes, mask=mask)
                if attributes_df is not None:
                    # Sauvegarder les attributs décodés dans le fichier principal
                    attributes_df.to_csv(write_dir / "synthetic_attributes.csv", index=False)
                    print(f"✓ Attributs décodés sauvegardés : {attributes_df.shape}")
            else:
                # Fallback : on ne peut pas décoder
                attributes_df = None
                print("Impossible de décoder les attributs (pas de decode_attributes)")

            # Merger schedules + attributes
            if attributes_df is not None and not attributes_df.empty:
                print("Merge automatique des schedules et attributs...")

                # Identifier les colonnes de jointure
                join_cols = []
                if 'hid' in synthetic_schedules_df.columns and 'hid' in attributes_df.columns:
                    join_cols.append('hid')
                if 'pid' in synthetic_schedules_df.columns and 'pid' in attributes_df.columns:
                    join_cols.append('pid')
                if 'member_idx' in synthetic_schedules_df.columns and 'member_idx' in attributes_df.columns:
                    join_cols.append('member_idx')

                if join_cols:
                    merged_df = pd.merge(synthetic_schedules_df, attributes_df, on=join_cols, how='left')
                    output_path = write_dir / "synthetic_schedules_with_attributes.csv"
                    merged_df.to_csv(output_path, index=False)
                    gantt_labels_df = merged_df
                    print(f"Fichier mergé créé : {output_path}")
                    print(f"   {len(merged_df)} lignes, {len(merged_df.columns)} colonnes")
                else:
                    print("Pas de colonnes de jointure communes, merge impossible")
        except Exception as e:
            print(f"Erreur lors du merge automatique : {e}")
            import traceback
            traceback.print_exc()

    if gantt_labels_df is None and isinstance(attributes_df, pd.DataFrame) and "pid" in attributes_df.columns:
        gantt_labels_df = attributes_df

    if gantt_enabled:
        saved_gantt_paths = _export_generated_interaction_gantts(
            write_dir=write_dir,
            schedules_df=synthetic_schedules_df,
            interactions_df=synthetic_interactions_df,
            seed=seed,
            num_households=gantt_num_households,
            attributes_df=gantt_labels_df,
            interaction_logits=generated_interaction_logits,
            interaction_mask=generated_interaction_mask,
        )
        print(
            f"Interaction gantt export: {len(saved_gantt_paths)} image(s) in {Path(write_dir) / 'interaction_gantt'}"
        )

    return synthetic_schedules_df, synthetic_attributes, zs


def generate_n(
    trainer: Trainer,
    n: int,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
    write_dir: Path,
    latent_categories: Optional[Tensor] = None,
    model: Optional[LightningModule] = None,
    z_override: Optional[torch.Tensor] = None,
    z_override_kind: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Génération INCONDITIONNELLE de N schedules (sans attributs socio-démo).

    Échantillonne N vecteurs latents z ~ N(0,I) et les décode via le modèle.
    Utilisé quand aucune population synthétique n'est fournie (pas de CSV d'attributs).
    Le z_override permet de substituer les latents standard par des échantillons GMM
    pour les modèles hiérarchiques (voir run_checkpoint_command).
    """
    torch.manual_seed(seed)
    if z_override is not None:
        if z_override_kind != "z_indiv":
            raise ValueError(f"Unsupported z_override_kind for generate_n: {z_override_kind}")
        if len(z_override) != int(n):
            raise ValueError(
                f"z_override length mismatch for generate_n: len(z_override)={len(z_override)} != n={int(n)}"
            )
        dataloaders = data.build_custom_indiv_gen_dataloader(
            z_indiv=z_override,
            attributes=None,
            batch_size=batch_size,
        )
    else:
        dataloaders = data.build_latent_dataloader(
            n, latent_dims, latent_categories, batch_size
        )
    time = datetime.datetime.now()
    if model is None:
        synth = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    else:
        # on garde l'état du modèle en mémoire (ex: priors GMM fittés a posteriori)
        synth = trainer.predict(model=model, dataloaders=dataloaders)
    duration = datetime.datetime.now() - time
    print(
        f">>> Generation took {duration.seconds + duration.microseconds / 1e6} seconds"
    )
    with open(write_dir / "timer.csv", "a") as fp:
        fp.write(
            f"generate,{duration.seconds + duration.microseconds / 1e6},seconds\n"
        )

    (
        _,
        synthetic_schedules,
        zs,
        interaction_logits,
        interaction_mask,
    ) = _extract_generation_tensors(synth, with_attributes=False)
    return synthetic_schedules, zs, interaction_logits, interaction_mask


def generate_from_attributes(
    trainer: Trainer,
    attributes: Tensor,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
    write_dir: Path,
    cats: Optional[Tensor] = None,
    model: Optional[LightningModule] = None,
    z_override: Optional[torch.Tensor] = None,
    z_override_kind: Optional[str] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Génération CONDITIONNELLE : produit des schedules guidés par des attributs socio-démo.

    Chaque ligne d'attributs (ex: homme, 35 ans, employé) est associée à un z aléatoire.
    Le decoder du VAE conditionnel utilise les deux (z + attributs) pour générer un
    emploi du temps cohérent avec le profil socio-démographique demandé.
    Pour les modèles Household, les attributs sont 3D : [N_ménages, max_members, N_attrs].
    """
    torch.manual_seed(seed)
    if z_override is not None:
        if z_override_kind != "z_indiv":
            raise ValueError(
                f"Unsupported z_override_kind for generate_from_attributes: {z_override_kind}"
            )
        if len(z_override) != len(attributes):
            raise ValueError(
                "z_override length mismatch for conditional generation: "
                f"len(z_override)={len(z_override)} != len(attributes)={len(attributes)}"
            )
        dataloaders = data.build_custom_indiv_gen_dataloader(
            z_indiv=z_override,
            attributes=attributes,
            batch_size=batch_size,
        )
    else:
        dataloaders = data.build_latent_conditional_dataloader(
            attributes, latent_dims, batch_size
        )
    time = datetime.datetime.now()
    if model is None:
        synth = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    else:
        # on garde l'état du modèle en mémoire (ex: priors GMM fittés a posteriori)
        synth = trainer.predict(model=model, dataloaders=dataloaders)
    duration = datetime.datetime.now() - time
    print(
        f">>> Generation took {duration.seconds + duration.microseconds / 1e6} seconds"
    )
    with open(write_dir / "timer.csv", "a") as fp:
        fp.write(
            f"generate,{duration.seconds + duration.microseconds / 1e6},seconds\n"
        )

    (
        synthetic_attributes,
        synthetic_schedules,
        zs,
        interaction_logits,
        interaction_mask,
    ) = _extract_generation_tensors(synth, with_attributes=True)
    if synthetic_attributes is None:
        raise ValueError("Expected generated attributes tensor but received None.")
    return (
        synthetic_attributes,
        synthetic_schedules,
        zs,
        interaction_logits,
        interaction_mask,
    )


def evaluate_synthetics(
    synthetic_schedules: dict[str, DataFrame],
    synthetic_labels: dict[str, DataFrame],
    default_eval_schedules: DataFrame,
    default_eval_attributes: DataFrame,
    write_path: Path,
    eval_params: dict,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    """Compare les emplois du temps synthétiques aux données réelles.

    Pourquoi c'est l'étape finale critique : toute la valeur de CAVEAT repose sur
    la qualité des données générées. Cette fonction utilise evaluate/ pour calculer
    des métriques de similarité (distributions d'activités, durées, transitions...),
    optionnellement conditionnées par des attributs socio-démo (split_on).
    Supporte aussi l'évaluation en batch (plusieurs modèles comparés ensemble).
    """
    print("\n======= Evaluating synthetic schedules =======")
    head = eval_params.get("head", 10)

    eval_schedules_path = eval_params.get("schedules_path", None)
    if eval_schedules_path:
        eval_schedules = data.load_and_validate_schedules(eval_schedules_path)
        print(
            f"<!> Loaded {len(eval_schedules)} schedules for evaluation from {eval_schedules_path}"
        )
    else:
        eval_schedules = default_eval_schedules
        print("Evaluating synthetic schedules against target schedules")

    split_on = eval_params.get("split_on", [])
    if split_on:
        print(
            f"Conditional Evaluation using: {split_on}, writing to {write_path}"
        )
        eval_attributes_path = eval_params.get("attributes_path", None)
        if eval_attributes_path:
            eval_attributes = data.load_and_validate_attributes(
                eval_params, eval_schedules
            )
            print(
                f"<!> Loaded {len(eval_attributes)} attributes for evaluation from {eval_attributes_path}"
            )
        else:
            eval_attributes = default_eval_attributes

        sub_reports = evaluate.subsample_and_evaluate(
            synthetic_schedules=synthetic_schedules,
            synthetic_attributes=synthetic_labels,
            target_schedules=eval_schedules,
            target_attributes=eval_attributes,
            split_on=split_on,
            report_stats=stats,
            verbose=verbose,
        )
        evaluate.report_splits(
            sub_reports,
            log_dir=write_path,
            head=head,
            verbose=verbose,
            suffix="_subs",
            ranking=len(synthetic_schedules) > 1,
        )
    print(f"Evaluating schedules, writing results to {write_path}.")
    reports = evaluate.evaluate(
        target_schedules=eval_schedules,
        synthetic_schedules=synthetic_schedules,
        report_stats=stats,
    )
    evaluate.report(
        reports,
        log_dir=write_path,
        head=head,
        verbose=verbose,
        ranking=len(synthetic_schedules) > 1,
    )


def conditional_sample(
    trainer: Trainer,
    population_size: int,
    data_encoder: encoding.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
) -> DataFrame:
    torch.manual_seed(seed)
    print("\n======= Sampling =======")
    predict_loader = data.build_latent_dataloader(
        population_size, config["model_params"]["latent_dim"], 256
    )
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    synthetic = data_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def build_encoder(config: dict) -> encoding.BaseEncoder:
    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoding.library[encoder_name](**config["encoder_params"])
    return data_encoder


def build_dataloader(
    config: dict, dataset: encoding.BaseDataset
) -> data.DataModule:
    data_loader_params = config.get("loader_params", {})
    collate_fn = None
    if hasattr(dataset, "get_interaction_pairs_for_batch"):
        try:
            from caveat.data.household_dataset_interactions import InteractionAwareCollator
            collate_fn = InteractionAwareCollator(dataset)
        except Exception as exc:
            print(f"Warning: could not create InteractionAwareCollator: {exc}")
            collate_fn = None
    datamodule = data.DataModule(
        data=dataset, collate_fn=collate_fn, **data_loader_params
    )
    datamodule.setup()
    return datamodule

def build_model(
    dataset: encoding.BaseDataset,
    config: dict,
    test: bool,
    gen: bool,
    label_kwargs: dict,
) -> LightningModule:
    model_name = config["model_params"]["name"]
    model_cls = models.library[model_name]

    # gestion des poids de classe
    encoding_weights = None
    use_weights = config.get("experiment_params", {}).get("use_activity_weights", True)

    if use_weights:
        if hasattr(dataset, "activity_weights"):
            encoding_weights = dataset.activity_weights
        elif hasattr(dataset, "encoding_weights"):
            encoding_weights = dataset.encoding_weights

    # configuration automatique de la taille du ménage
    hh_size_col_idx = None
    hh_size_map = None

    if hasattr(dataset, "label_cols") and "household_size" in dataset.label_cols:
        print("Configuration automatique de la taille du ménage...")
        
        hh_size_col_idx = dataset.label_cols.index("household_size")
        
        mapping_dict = dataset.label_dics["household_size"]
        max_idx = max(mapping_dict.values())
        hh_size_map = [0] * (max_idx + 1)
        
        for val_str, idx in mapping_dict.items():
            if idx == 0: continue 
            try:
                val_int = int(float(val_str)) 
                hh_size_map[idx] = val_int
            except ValueError:
                hh_size_map[idx] = dataset.max_members

        print(f" Index colonne : {hh_size_col_idx}")
        print(f" Mapping détecté : {hh_size_map}")

    # on fait une copie pour ne pas modifier le dictionnaire original
    model_config = config["model_params"].copy()
    
    # injection des infos calculées (Household Size)
    if hh_size_col_idx is not None:
        model_config["hh_size_col_idx"] = hh_size_col_idx
        model_config["hh_size_map"] = hh_size_map

    # injection des dimensions du dataset 
    if hasattr(dataset, "time_steps"):
        model_config["time_steps"] = dataset.time_steps
    if hasattr(dataset, "max_members"):
        model_config["max_members"] = dataset.max_members


    # label_cardinalities (pour modèles household)
    model_kwargs = {
        "in_shape": dataset.shape(),
        "encodings": dataset.activity_encodings,
        "encoding_weights": encoding_weights,
        **model_config,
        **config.get("experiment_params", {}),
        "scheduler_params": config.get("scheduler_params", None),
        "test": test,
        "gen": gen,
        **label_kwargs,
    }

    #  label_cardinalities (pour modèles household)
    if hasattr(dataset, "label_cardinalities"):
        model_kwargs["label_cardinalities"] = dataset.label_cardinalities
    model = model_cls(**model_kwargs)
    return model

def load_model(ckpt_path: Path, config: dict) -> LightningModule:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    return model.load_from_checkpoint(ckpt_path)


def build_trainer(logger: TensorBoardLogger, config: dict) -> Trainer:
    trainer_config = config.get("trainer_params", {}).copy()
    patience = trainer_config.pop("patience", 5)
    early_stopping_start_epoch = trainer_config.pop(
        "early_stopping_start_epoch", None
    )
    if early_stopping_start_epoch is None:
        model_params = config.get("model_params", {})
        kld_type = model_params.get("kld_annealing_type")
        kld_epochs = model_params.get("kld_annealing_epochs", 0)
        if kld_type in {"monotonic", "cyclical"} and kld_epochs:
            try:
                early_stopping_start_epoch = int(kld_epochs)
            except (TypeError, ValueError):
                early_stopping_start_epoch = 0
    if early_stopping_start_epoch is None:
        early_stopping_start_epoch = 0
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=2,
        save_weights_only=False,
        save_last=True
    )
    loss_scheduling = trainer_config.pop("loss_scheduling", {})
    custom_loss_scheduler = LinearLossScheduler(loss_scheduling)
    return Trainer(
        logger=logger,
        callbacks=[
            DelayedEarlyStopping(
                monitor="val_loss",
                patience=patience,
                stopping_threshold=0.0,
                start_epoch=early_stopping_start_epoch,
            ),
            LearningRateMonitor(),
            checkpoint_callback,
            custom_loss_scheduler,
        ],
        **trainer_config,
    )


def initiate_logger(save_dir: Union[Path, str], name: str, version_name: str = None) -> TensorBoardLogger:
    """
    Initializes a TensorBoardLogger object for logging training progress.

    Args:
        save_dir (str): The directory where the logs will be saved.
        name (str): The name of the logger.
        version_name (str, optional): Custom version name. If None, PyTorch Lightning will auto-generate version_xx.

    Returns:
        TensorBoardLogger: The initialized TensorBoardLogger object.
    """
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name, version=version_name)
    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(
        exist_ok=True, parents=True
    )
    Path(f"{tb_logger.log_dir}/val_z").mkdir(exist_ok=True, parents=True)
    return tb_logger


