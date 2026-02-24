"""
HouseholdInteractionsEncoder

Encoder qui charge les schedules + attributs + interactions et produit
un HouseholdDatasetWithInteractions.

Hérite de la logique de HouseholdEncoder pour réutiliser decode() et decode_attributes().
"""

import pandas as pd
import torch
from caveat.encoding.base import BaseEncoder
from caveat.data.household_dataset_interactions import HouseholdDatasetWithInteractions


class HouseholdInteractionsEncoder(BaseEncoder):
    """
    Encoder pour le modèle dual-branch avec interactions sociales.

    Charge:
    - schedules:    CSV avec colonnes [pid, act, start, end, schedule_id (optionnel)]
    - attributes:   CSV avec colonnes [pid, hid, age, sex, ...]
    - interactions: CSV avec colonnes [schedule_id, interaction_bin, pid]
                    ou format étendu [schedule_id, interaction_bin, pid, overlap_start, overlap_end]

    Produit un HouseholdDatasetWithInteractions qui retourne par ménage:
    - x:            [M, T] activités encodées
    - mask:         [M]    membres réels
    - labels:       [M, L] attributs socio-démo encodés
    - interactions: [M, T] interaction binaire par slot (0 ou 1)
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.idx_to_act = {}
        self.idx_to_label = {}
        self.label_cols = []
        self.time_steps = kwargs.get("time_steps", 144)

        # dataset_params.slot_conflict_mode:
        # - "winner" (default): activité avec overlap max dans le slot
        # - "diversity": si motif 1-2-1 dans un slot, l'activité intermédiaire est priorisée

    def encode(self, schedules, labels=None, **kwargs):
        """Encode les données en créant le dataset avec interactions."""

        attr_path = self.config.get("attributes_path")
        if attr_path is None:
            raise ValueError("attributes_path requis dans encoder_params")

        print(f"Chargement des attributs depuis {attr_path}")
        attributes = pd.read_csv(attr_path)

        # Interactions (optionnel)
        interactions_path = self.config.get("interactions_path")
        interactions = None
        if interactions_path:
            print(f"Chargement des interactions depuis {interactions_path}")
            interactions = pd.read_csv(interactions_path)
        else:
            print("Pas de fichier d'interactions — mode sans supervision sociale")

        # Création du dataset
        dataset = HouseholdDatasetWithInteractions(
            config=self.config,
            schedules=schedules,
            attributes=attributes,
            interactions=interactions,
        )

        # Récupérer les mappings pour le decode
        if hasattr(dataset, "idx_to_act"):
            self.idx_to_act = dataset.idx_to_act

        if hasattr(dataset, "label_dics") and hasattr(dataset, "label_cols"):
            self.idx_to_label = {}
            for col in dataset.label_cols:
                if col in dataset.label_dics:
                    self.idx_to_label[col] = {
                        idx: val for val, idx in dataset.label_dics[col].items()
                    }
            self.label_cols = dataset.label_cols

        return dataset

    def decode(self, schedules, argmax=True, mask=None):
        """
        Décode les schedules générés en DataFrame d'épisodes d'activités.

        Identique à HouseholdEncoder.decode().

        Args:
            schedules: Tensor de shape [B, M, T] ou [B, M, T, Classes]
            argmax: Si True, applique argmax sur la dernière dimension
            mask: Tensor optionnel de shape [B, M] indiquant les membres réels (1) vs fantômes (0)

        Returns:
            DataFrame avec colonnes [pid, hid, member_idx, act, start, end, duration]
        """
        if isinstance(schedules, torch.Tensor):
            schedules = schedules.cpu()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        idx_to_act = self.idx_to_act if hasattr(self, "idx_to_act") else {}

        if argmax and schedules.dim() == 4:
            schedules = torch.argmax(schedules, dim=-1)

        rows = []
        timestep_min = 1440 / self.config.get('time_steps', 144)

        for i in range(len(schedules)):
            household = schedules[i]
            for member_idx in range(household.shape[0]):
                if mask is not None and mask[i, member_idx] == 0:
                    continue

                acts = household[member_idx].tolist()
                if sum(acts) == 0:
                    continue

                current_act = acts[0]
                start_step = 0
                gen_pid = f"Gen_HH{i}_M{member_idx}"

                for t, act in enumerate(acts):
                    if act != current_act:
                        act_name = idx_to_act.get(current_act, f"act_{current_act}")
                        if act_name != "PAD":
                            rows.append({
                                'pid': gen_pid,
                                'hid': i,
                                'member_idx': member_idx,
                                'act': act_name,
                                'start': start_step * timestep_min,
                                'end': t * timestep_min,
                                'duration': (t - start_step) * timestep_min,
                            })
                        current_act = act
                        start_step = t

                act_name = idx_to_act.get(current_act, f"act_{current_act}")
                if act_name != "PAD":
                    time_steps = self.config.get('time_steps', 144)
                    rows.append({
                        "pid": gen_pid,
                        "hid": i,
                        "member_idx": member_idx,
                        "act": act_name,
                        "start": start_step * timestep_min,
                        "end": 1440,
                        "duration": (time_steps - start_step) * timestep_min,
                    })

        return pd.DataFrame(rows)

    def decode_attributes(self, attributes_tensor, mask=None):
        """
        Décode un tenseur d'attributs encodés vers leurs valeurs textuelles.

        Args:
            attributes_tensor: Tensor de shape [B, M, N_attrs] avec les attributs encodés
            mask: Tensor optionnel de shape [B, M] pour filtrer les membres fantômes

        Returns:
            DataFrame avec les attributs décodés, avec colonnes hid, member_idx, puis les attributs
        """
        if isinstance(attributes_tensor, torch.Tensor):
            attributes_tensor = attributes_tensor.cpu()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        if not hasattr(self, "idx_to_label") or not self.idx_to_label:
            B, M, N = attributes_tensor.shape
            flat = attributes_tensor.reshape(-1, N).numpy()
            df = pd.DataFrame(flat)
            df.insert(0, "member_idx", [j for _ in range(B) for j in range(M)])
            df.insert(0, "hid", [i for i in range(B) for _ in range(M)])
            return df

        rows = []
        B, M, N_attrs = attributes_tensor.shape

        for i in range(B):
            for j in range(M):
                if mask is not None and mask[i, j] == 0:
                    continue

                row = {"hid": i, "member_idx": j}
                for attr_idx, col_name in enumerate(self.label_cols):
                    if attr_idx >= N_attrs:
                        break
                    enc_val = int(attributes_tensor[i, j, attr_idx].item())
                    if col_name in self.idx_to_label and enc_val in self.idx_to_label[col_name]:
                        row[col_name] = self.idx_to_label[col_name][enc_val]
                    else:
                        row[col_name] = enc_val
                rows.append(row)

        return pd.DataFrame(rows)
