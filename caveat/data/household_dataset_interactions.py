

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HouseholdDatasetWithInteractions(Dataset):

    def __init__(self, config, schedules, attributes, interactions=None):


        self.time_steps = int(config.get('time_steps', 144))
        ds_params = config.get('dataset_params', {})
        self.max_members = int(ds_params.get('max_members', 5))
        self.group_col = ds_params.get('group_by', 'hid')
        self.label_cols = list(ds_params.get('label_cols', []))
        # winner ou diversity
        # winner = act la plus longue gagne
        # diversity = garde l'act intermédiaire entre 2 act de même nature
        self.slot_conflict_mode = str(
            ds_params.get("slot_conflict_mode", config.get("slot_conflict_mode", "winner"))
        ).lower()


        # pré-calcul pour conversion temps → slot
        self.step_size = 1440.0 / self.time_steps
        self.slot_edges = np.arange(self.time_steps + 1, dtype=np.float32) * self.step_size

        schedules = schedules.copy()
        attributes = attributes.copy()

        schedules['pid'] = schedules['pid'].astype(str)
        attributes['pid'] = attributes['pid'].astype(str)
        self.has_schedule_id = 'schedule_id' in schedules.columns
        if self.has_schedule_id:
            schedules['schedule_id'] = schedules['schedule_id'].astype(int)

        # encoding des activités
        acts = sorted(schedules['act'].unique())
        self.act_to_idx = {"PAD": 0} | {act: idx for idx, act in enumerate(acts, start=1)}
        self.idx_to_act = {idx: act for act, idx in self.act_to_idx.items()}
        self.activity_encodings = len(self.act_to_idx)

        print(f"  Activités: {self.activity_encodings}")
        print(f"  Slot conflict mode: {self.slot_conflict_mode}")

        # Colonnes de tri
        if 'age' in attributes.columns:
            attributes['_age_sort'] = pd.to_numeric(attributes['age'], errors='coerce').fillna(-1)
        else:
            attributes['_age_sort'] = 0

        if 'sex' in attributes.columns:
            attributes['_sex_sort'] = attributes['sex'].astype(str)
        else:
            attributes['_sex_sort'] = ''

        # encode les attributs
        self.label_dics = {}
        self.label_cardinalities = []

        for col in self.label_cols:
            labels = sorted(attributes[col].astype(str).unique())
            mapping = {"PAD": 0} | {lab: i for i, lab in enumerate(labels, start=1)}
            self.label_dics[col] = mapping
            self.label_cardinalities.append(len(mapping))
            attributes[f'_enc_{col}'] = attributes[col].astype(str).map(mapping).fillna(0).astype(np.int16)

        print(f"  Labels: {len(self.label_cols)} colonnes, cardinalités: {self.label_cardinalities}")

        # préparation des interactions
        self.interactions_present = interactions is not None
        self.interaction_map = {}
        self.interaction_intervals_map = {}
        self.has_interval_interactions = False
        if self.interactions_present:
            if not self.has_schedule_id:
                raise ValueError(
                    "Interactions provided but schedules have no 'schedule_id' column."
                )
            interactions = interactions.copy()
            interactions['pid'] = interactions['pid'].astype(str)
            interactions['schedule_id'] = interactions['schedule_id'].astype(int)
            if 'interaction_bin' not in interactions.columns:
                interactions['interaction_bin'] = 1
            interactions['interaction_bin'] = pd.to_numeric(
                interactions['interaction_bin'], errors='coerce'
            ).fillna(0).astype(np.uint8)

            # schedule_id est unique donc max = min = mean = first etc
            interaction_max = interactions.groupby('schedule_id', sort=False)['interaction_bin'].max()
            self.interaction_map = {
                int(sid): int(val) for sid, val in interaction_max.items() if int(val) > 0
            }

            # nouveau format: intervalles de recouvrement par schedule_id
            has_overlap_cols = {'overlap_start', 'overlap_end'}.issubset(interactions.columns)
            if has_overlap_cols:
                interactions['overlap_start'] = pd.to_numeric(
                    interactions['overlap_start'], errors='coerce'
                )
                interactions['overlap_end'] = pd.to_numeric(
                    interactions['overlap_end'], errors='coerce'
                )
                # seulement les interactions avec des intervalles valides
                intervals_df = interactions.loc[
                    (interactions['interaction_bin'] > 0)
                    & interactions['overlap_start'].notna()
                    & interactions['overlap_end'].notna()
                    & (interactions['overlap_end'] > interactions['overlap_start']),
                    ['schedule_id', 'overlap_start', 'overlap_end'],
                ]
                interval_map = {}
                for sid, group in intervals_df.groupby('schedule_id', sort=False):
                    # pour chaque schedule_id on ajoute la liste des intervalles
                    starts = group['overlap_start'].to_numpy(dtype=np.float32)
                    ends = group['overlap_end'].to_numpy(dtype=np.float32)
                    interval_map[int(sid)] = list(zip(starts, ends))
                self.interaction_intervals_map = interval_map
                self.has_interval_interactions = bool(interval_map)

            n_positive = len(self.interaction_map)
            n_total = int(interactions['schedule_id'].nunique())
            if self.has_interval_interactions:
                n_intervals = sum(len(v) for v in self.interaction_intervals_map.values())
                print(
                    f"  Interactions: {n_total} schedule_ids, {n_positive} positives "
                    f"({100*n_positive/max(n_total,1):.1f}%), {n_intervals} intervalles"
                )
            else:
                print(
                    f"  Interactions: {n_total} schedule_ids, {n_positive} positives "
                    f"({100*n_positive/max(n_total,1):.1f}%)"
                )
        else:
            print("  Interactions: aucune")

        # merge des données
        enc_cols = [f'_enc_{col}' for col in self.label_cols]
        sched_cols = ["pid", "act", "start", "end"]
        if self.has_schedule_id:
            sched_cols.append("schedule_id")
        attr_cols = ["pid", self.group_col, "_age_sort", "_sex_sort"] + enc_cols

        df = schedules[sched_cols].merge(
            attributes[attr_cols],
            on="pid",
            how="left"
        )

        # applique le dictionnaire pour encoder les activités
        df['_act_idx'] = df['act'].astype(str).map(self.act_to_idx).fillna(0).astype(np.uint8)

        # idem pour les interactions
        if self.has_schedule_id and self.interaction_map:
            df['_interaction_bin'] = df['schedule_id'].map(self.interaction_map).fillna(0).astype(np.uint8)
        else:
            df['_interaction_bin'] = 0


        print("Tri des données...")
        df.sort_values(
            by=[self.group_col, '_age_sort', '_sex_sort'],
            ascending=[True, False, True],
            inplace=True
        )

        grouped = df.groupby(self.group_col, sort=False)
        self.num_households = len(grouped)

        print(f"  Ménages: {self.num_households}")

        # création des grilles
        self.x_tensor = torch.zeros(
            (self.num_households, self.max_members, self.time_steps),
            dtype=torch.uint8
        )
        self.mask_tensor = torch.zeros(
            (self.num_households, self.max_members),
            dtype=torch.bool
        )
        self.label_tensor = torch.zeros(
            (self.num_households, self.max_members, len(self.label_cols)),
            dtype=torch.uint8
        )

        # Tenseur d'interactions [B, M, T]
        self.interaction_tensor = torch.zeros(
            (self.num_households, self.max_members, self.time_steps),
            dtype=torch.uint8
        )

        # Paires de membres en interaction simultanée [hh_idx, m1, m2, t]
        self.interaction_pairs_indices = []

        enc_col_names = enc_cols

        print("  Génération des grilles...")

        for i, (hid, hh_group) in enumerate(tqdm(grouped, desc="  Ménages")):

            # groupe par membre sans avoir à filtrer manuellement chaque PID
            pid_groups = hh_group.groupby("pid", sort=False)

            # Mapping pid -> member_index pour ce ménage (essentiel pour _build_interaction_pairs plus tard)
            pid_to_midx = {}

            m_idx = 0
        
            for pid, member_df in pid_groups:
                if m_idx >= self.max_members:
                    break

                # position du membre dans le tenseur
                pid_to_midx[pid] = m_idx
                # marquer le membre comme présent
                self.mask_tensor[i, m_idx] = True

                # attributs (constant par membre, première ligne suffit)
                first_row = member_df.iloc[0]
                for j, enc_col in enumerate(enc_col_names):
                    self.label_tensor[i, m_idx, j] = first_row[enc_col]

                # récupération des données brutes pour ce membre
                # colonne temporarires sans NaN
                # stable permet de garder l'ordre d'origine en cas d'égalité
                member_df = member_df.assign(
                    _start_num=pd.to_numeric(member_df["start"], errors="coerce"),
                    _end_num=pd.to_numeric(member_df["end"], errors="coerce"),
                ).sort_values(by=["_start_num", "_end_num"], kind="stable")

                acts = member_df["_act_idx"].to_numpy(dtype=np.int16)
                starts = member_df["_start_num"].to_numpy(dtype=np.float32)
                ends = member_df["_end_num"].to_numpy(dtype=np.float32)
                # récupération du flag binaire d'interaction
                interaction_bins = member_df["_interaction_bin"].to_numpy(dtype=np.uint8)
                schedule_ids = (
                    member_df['schedule_id'].values
                    if self.has_schedule_id and 'schedule_id' in member_df.columns
                    else np.full(len(member_df), -1, dtype=np.int64)
                )

                # Stocke la durée (en min) de l'activité qui occupe actuellement la case.
                # Initialisé à -1.0 pour être sûr que n'importe quelle vraie durée (>0) l'écrase.
                slot_candidates = [[] for _ in range(self.time_steps)]
                
                # Stocke l'ID de l'activité gagnante (0 = PAD par défaut)
                member_grid = np.zeros(self.time_steps, dtype=np.int16)
                # Stocke le flag d'interaction correspondant à l'activité gagnante
                interaction_grid = np.zeros(self.time_steps, dtype=np.uint8)

                # Pour chaque activité 
                for seq_idx, (act, start, end, ibin, schedule_id) in enumerate(
                    zip(acts, starts, ends, interaction_bins, schedule_ids)
                ):
                    if np.isnan(start) or np.isnan(end) or int(act) == 0:
                        continue
                    # Conversion temps continu -> indices de slots (discrets)
                    s_slot = int(start / self.step_size)
                    e_slot = int((end - 1e-4) / self.step_size)

                    if e_slot < 0 or s_slot >= self.time_steps:
                        continue
                    s_slot = max(0, s_slot)
                    e_slot = min(self.time_steps - 1, e_slot)
                    if s_slot > e_slot:
                        continue

                    # Bords des slots concernés pour calcul précis de l'overlap
                    # +2 car slice Python exclut la fin et on veut le bord droit du dernier slot
                    edges = self.slot_edges[s_slot:e_slot + 2]

                    # Calcul de l'overlap : min(end, slot_end) - max(start, slot_start)
                    overlap = np.minimum(end, edges[1:]) - np.maximum(start, edges[:-1])
                    overlap = np.maximum(overlap, 0.0)
                    # overlap de chaque activité pour chaque slot

                    # obtient les intervalles par sid
                    # puis les convertis par  POUR LES INTERACTIONS
                    sid = int(schedule_id) if schedule_id is not None else -1
                    intervals = self.interaction_intervals_map.get(sid)
                    if intervals:
                        slot_interaction = self._intervals_to_slot_flags(
                            slot_edges=edges,
                            activity_overlap=overlap,
                            intervals=intervals,
                        )
                    else:
                        # sinon 1 partout
                        slot_interaction = np.full(overlap.shape, ibin, dtype=np.uint8)
                    # itère sur chaque slot local touché par cette activité.
                    # local_idx = index du slot touché 
                    for local_idx, ov in enumerate(overlap):
                        if ov <= 0.0:
                            continue
                        t = s_slot + local_idx # time step
                        # quelle ordre d'activité, quel nom d'activité, son overlap avec le slot actuel, et est ce en interaction
                        slot_candidates[t].append(
                            (
                                seq_idx,
                                int(act),
                                float(ov),
                                int(slot_interaction[local_idx]),
                            )
                        )
                # slot_candidates peut être > 1 si on a plusieurs activités
                # il faut choisir le gagnant soit via winner soit via diversity
                # résout les conflits de slots 
                resolved_winners = self._resolve_member_slot_winners(slot_candidates)
                # stocke dans le tenseur du ménage/membre
                for t, winner in enumerate(resolved_winners):
                    if winner is not None:
                        member_grid[t] = winner[1]
                        interaction_grid[t] = winner[3]

                self.x_tensor[i, m_idx] = torch.from_numpy(member_grid.astype(np.uint8))
                self.interaction_tensor[i, m_idx] = torch.from_numpy(interaction_grid)

                m_idx += 1

            # construction des paires d'interaction 
            # utilise pid_to_midx pour savoir quels lignes comparer
            if self.interactions_present:
                self._build_interaction_pairs(i, hh_group, pid_to_midx)

        # Convertir les indices de paires en tenseur sparse
        if self.interactions_present:
            self._finalize_interaction_pairs()
        else:
            self.num_interaction_pairs = 0
            self.interaction_pairs_data = torch.zeros((0, 4), dtype=torch.int32)

        # Poids
        self._calculate_weights()

    def _winner_max_overlap(self, candidates):
        if not candidates:
            return None

        # mode winner: max overlap 
        winner_idx = 0
        winner_overlap = candidates[0][2]
        for i in range(1, len(candidates)):
            ov = candidates[i][2]
            if ov > winner_overlap:
                winner_idx = i
                winner_overlap = ov

        return candidates[winner_idx]

    def _resolve_member_slot_winners(self, slot_candidates):
        """
        Résout les gagnants par slot.
        - mode winner: max overlap indépendant par slot
        - mode diversity:
          si une activité perd en winner ET n'existe pas dans les slots adjacents,
          alors elle gagne. Si c'est ambigu (ex: 2 activités "disparues"), on garde winner.
        """
        winners = [self._winner_max_overlap(candidates) for candidates in slot_candidates]

        if self.slot_conflict_mode != "diversity":
            return winners

        resolved = list(winners)
        n_slots = len(slot_candidates)

        def has_adjacent_same(slot_idx: int, act: int) -> bool:
            if slot_idx > 0 and winners[slot_idx - 1] is not None and int(winners[slot_idx - 1][1]) == act:
                return True
            if slot_idx + 1 < n_slots and winners[slot_idx + 1] is not None and int(winners[slot_idx + 1][1]) == act:
                return True
            return False

        for t, candidates in enumerate(slot_candidates):
            if not candidates or winners[t] is None:
                continue

            acts = sorted({int(c[1]) for c in candidates})
            if len(acts) < 2:
                continue  # pas de conflit

            winner = winners[t]
            winner_act = int(winner[1])
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
                if int(cand[1]) != target_act:
                    continue
                if (
                    chosen is None
                    or float(cand[2]) > float(chosen[2])
                    or (float(cand[2]) == float(chosen[2]) and int(cand[0]) < int(chosen[0]))
                ):
                    chosen = cand
            if chosen is not None:
                resolved[t] = chosen

        return resolved

    def _select_slot_candidate(self, candidates):
        # Conservé pour compatibilité: sélection locale par max overlap.
        return self._winner_max_overlap(candidates)

    @staticmethod
    def _intervals_to_slot_flags(slot_edges, activity_overlap, intervals):
        """
        Convertit une liste d'intervalles continus [start, end) en flags binaires par slot.
        Un slot est positif si son overlap strict (>0) avec au moins un intervalle est positif.
        """
        flags = np.zeros_like(activity_overlap, dtype=np.uint8)
        for int_start, int_end in intervals:

            # on calcule le minimum entre la fin de l'intervalle et les bords et le max entre le début et les bords
            # ça donne l'overlap
            # on rprend > 0
            overlap = np.minimum(float(int_end), slot_edges[1:]) - np.maximum(
                float(int_start), slot_edges[:-1]
            )
            flags = np.maximum(flags, (overlap > 0.0).astype(np.uint8))
        return np.where(activity_overlap > 0.0, flags, 0).astype(np.uint8)


    def _build_interaction_pairs(self, hh_idx, hh_group, pid_to_midx):
        """
        Construit les paires d'interaction pour un ménage.
        """
        # on loop sur les slots pour trouver les membres qui interagissent simultanément
        for t in range(self.time_steps):
            members_with_interaction = []
            
            for pid in pid_to_midx:
                # sa ligne
                m_idx = pid_to_midx[pid]
                # récupère le flag d'interaction pour ce membre à ce slot
                ibin = self.interaction_tensor[hh_idx, m_idx, t].item()
                # on l'ajoute à la liste si c'est positif
                if ibin > 0:
                    members_with_interaction.append(m_idx)

            # si au moins 2 membres interagissent au même slot
            if len(members_with_interaction) >= 2:
                for idx1 in range(len(members_with_interaction)):
                    for idx2 in range(idx1 + 1, len(members_with_interaction)):
                        m1 = members_with_interaction[idx1]
                        m2 = members_with_interaction[idx2]
                        self.interaction_pairs_indices.append(
                            (hh_idx, m1, m2, t)
                        ) # pour la loss

    def _finalize_interaction_pairs(self):
        """
        Convertit les listes de tuples en tenseurs et construit un index
        par ménage pour un lookup rapide dans le collator.
        """
        if self.interaction_pairs_indices:
            pairs = np.array(self.interaction_pairs_indices, dtype=np.int32)
            # trier par hh_idx pour pouvoir indexer par slices
            order = np.argsort(pairs[:, 0], kind='stable')
            pairs = pairs[order]
            self.num_interaction_pairs = len(pairs)

            # tenseur d'indices [N_pairs, 4] : (hh_idx, m1, m2, t)
            self.interaction_pairs_data = torch.from_numpy(pairs)

            # index par ménage : hh_idx -> (start, end) dans le tenseur trié
            self.pair_slices = {}
            hh_ids = pairs[:, 0]
            changes = np.where(np.diff(hh_ids) != 0)[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(pairs)]])
            for s, e in zip(starts, ends):
                self.pair_slices[int(hh_ids[s])] = (int(s), int(e))

            print(f"  Paires d'interaction: {self.num_interaction_pairs}")
        else:
            self.num_interaction_pairs = 0
            self.interaction_pairs_data = torch.zeros((0, 4), dtype=torch.int32)
            self.pair_slices = {}
            print("  Paires d'interaction: 0")

    def _calculate_weights(self):
        # comptage par slot actif (et non par episodes)
        active_mask = self.mask_tensor.unsqueeze(-1).expand_as(self.x_tensor)
        active_slots = self.x_tensor[active_mask].long()
        counts = torch.bincount(active_slots, minlength=self.activity_encodings).float()
        total = float(counts[1:].sum().item())
        num_classes = self.activity_encodings
        K = float(num_classes - 1) if num_classes > 2 else 1.0

        weights = torch.ones(num_classes, dtype=torch.float32)

        print("  Poids des activités:")
        for idx in range(1, num_classes):
            count = float(counts[idx].item())
            if count <= 0.0:
                continue
            w = total / (K * count)
            weights[idx] = w
            act_name = self.idx_to_act.get(idx, f"act_{idx}")
            print(f"    {act_name}: {w:.4f}")

        weights = torch.clamp(weights, min=0.1, max=10.0)
        weights[0] = 0.0
        self.activity_weights = weights

    def shape(self):
        return (self.max_members, self.time_steps)

    def __len__(self):
        return self.num_households

    def __getitem__(self, idx):
        return {
            "x": self.x_tensor[idx].long(),
            "mask": self.mask_tensor[idx].float(),
            "labels": self.label_tensor[idx].long(),
            "interactions": self.interaction_tensor[idx].long(),
            "interactions_present": self.interactions_present,
            "index": idx,
        }

    def get_interaction_pairs_for_batch(self, batch_indices):
        # lookup direct via l'index pré-calculé par ménage
        if self.num_interaction_pairs == 0:
            return torch.zeros((0, 4), dtype=torch.long)

        chunks = []
        for local_idx, global_idx in enumerate(batch_indices):
            global_idx = int(global_idx)
            sl = self.pair_slices.get(global_idx)
            if sl is not None:
                start, end = sl
                chunk = self.interaction_pairs_data[start:end].clone()
                chunk[:, 0] = local_idx  # remap vers l'index batch-local
                chunks.append(chunk)

        if not chunks:
            return torch.zeros((0, 4), dtype=torch.long)
        return torch.cat(chunks, dim=0).long()


class InteractionAwareCollator:
    """
    Collator personnalisé qui ajoute les paires d'interaction au batch.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        # Batch standard
        x = torch.stack([item['x'] for item in batch])
        mask = torch.stack([item['mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        interactions = torch.stack([item['interactions'] for item in batch])
        interactions_present = any(
            [bool(item.get("interactions_present", False)) for item in batch]
        )
        batch_indices = [item['index'] for item in batch] 
        sparse_pairs = self.dataset.get_interaction_pairs_for_batch(batch_indices)
        return {
            'x': x,
            'mask': mask,
            'labels': labels,
            'interactions': interactions, # [B, M, T)]
            'interaction_pairs': sparse_pairs, # [N_pairs, 4]
            'interactions_present': interactions_present, # pour savoir si il y a au moins une interaction dans le batch
        }
