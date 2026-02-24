# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class HouseholdDataset(Dataset):
#     def __init__(self, config, schedules, attributes):

#         # permet d'assurer que le merge marche bien
#         schedules = schedules.copy()
#         attributes = attributes.copy()
#         schedules['pid'] = schedules['pid'].astype(str)
#         attributes['pid'] = attributes['pid'].astype(str)
#         # initialisation des paramètres de la config
#         self.time_steps = config.get('max_length',144)
#         ds_params = config.get('dataset_params', {})
#         self.max_members = ds_params.get('max_members', 5)
#         self.group_col = ds_params.get('group_by', 'hid')
#         self.label_cols = ds_params.get('label_cols', [])

#         # initialisation de l'encoding activité
#         # on crée des index pour chaque activité (0 réservé à PAD)
#         acts = sorted(schedules['act'].unique())
#         self.act_to_idx = {"PAD": 0} | {act: idx for idx, act in enumerate(acts, start=1)}
#         self.activity_encodings = len(self.act_to_idx)
#         self.encoding_weights = None # pas utilisé pour le moment

#         # pour la génération
#         self.idx_to_act = {idx:act for act,idx in self.act_to_idx.items()}

#         # calcul des poids pour la loss
#         counts = schedules['act'].value_counts()

#         # nombre total d'occurrences
#         total_occurrences = float(counts.sum())
#         # nombre de classes (sans PAD)
#         num_classes = max(self.act_to_idx.values()) + 1
#         K = float(num_classes - 2)
#         weights = torch.ones(num_classes, dtype=torch.float)

#         # calcul du poids par activité
#         for act, count in counts.items():
#             if act in self.act_to_idx:
#                 idx = self.act_to_idx[act]
#                 if idx == 0: continue

#                 if act.lower() in ["trip", "travel", "trajet"]:
#                     weights[idx] = 1.0
#                     print(f"  - {act} : Poids FORCÉ à 1.0000")
#                     continue
#                 # formule classique
#                 w = total_occurrences / (K * float(count))
#                 weights[idx] = w
#                 print(f"  - {act} : Poids = {w:.4f}")

#         # clamp weights first
#         weights = torch.clamp(weights, min=0.1, max=10.0)
#         # then set PAD weight to 0 (so it's not affected by clamping)
#         weights[0] = 0.0

#         # enregistre les poids pour la loss
#         self.activity_weights = weights

        
#         # initialisation de l'encoding labels

#         # on a besoin, pour chaque label, de crée un dictionnaire d'indexation
#         # la cardinalité est nécessaire pour l'embedding
#         self.label_cardinalities = []
#         self.label_dics = {}

#         attributes_str = attributes.copy()
#         for col in self.label_cols: # loop sur les labels
#              labels = sorted(attributes[col].unique().astype(str))
#              mapping = {"PAD": 0} | {lab:i for i,lab in enumerate(labels, start=1)} # dictionnaire d'indexation
#              self.label_dics[col] = mapping 
#              self.label_cardinalities.append(len(mapping)) 
#              attributes_str[col] = attributes_str[col].astype(str)  
#         attributes_sorted = attributes.copy()
        
#         # on impose l'ordre dans la grille -> les membres sont triés par âge décroissant puis par sex
#         # passe en numérique au cas où pour que le sort marche dans le bon ordre
#         if 'age' in attributes_sorted.columns:
#             attributes_sorted['age_sort'] = pd.to_numeric(attributes_sorted['age'], errors='coerce').fillna(-1)
#         else:
#             attributes_sorted['age_sort'] = 0

#         # rassemble les schedules et les attributs en mergeant sur pid
#         self.data = schedules[["pid", "act", "start", "end"]].merge(
#             attributes_sorted[["pid", self.group_col, "age_sort"] + self.label_cols], 
#             on="pid", 
#             how="left"
#         )
#         print("Tri des membres par Age")
#         # tri
#         self.data.sort_values(
#             by=[self.group_col, 'age_sort', 'sex'], 
#             ascending=[True, False, True], 
#             inplace=True)
#         # on crée un Dataset niveau household
#         # chaque élément représente l'ensemble des membres du ménage
#         self.households = [group for _, group in self.data.groupby(self.group_col,sort=False)]


#     def shape(self):
#          return (self.max_members, self.time_steps)
    
#     def __getitem__(self, idx):

#         # un élement = un household
#         hh_data = self.households[idx]

#         # nombre de membres du ménage
#         unique_pids = hh_data["pid"].unique()
#         num_present = len(unique_pids)

#         # on limite à max_members -> au dessus on ne garde que les 5 premiers
#         if num_present > self.max_members:
#             unique_pids = unique_pids[:self.max_members]
#             num_present = self.max_members
        
#         # initialisation de la grille [M,T] (long car ce sont des index d'activité)
#         grid = torch.zeros((self.max_members,self.time_steps), dtype=torch.long)

#         # inialisation de la grille du mask -> [M] avec 1.0 pour les membres présents et 0.0 pour les autres
#         mask = torch.zeros(self.max_members, dtype=torch.float)
#         mask[:num_present] = 1.0

#         # pour passer d'un temps réel à un indice
#         step_size = 1440.0 / self.time_steps

#         # initialisation de la grille des labels -> constant dans le temps donc [M,L]
#         label_grid = torch.zeros((self.max_members, len(self.label_cols)), dtype=torch.long)

#         # pour chaque membre du ménage on crée sa ligne dans les grilles
#         for i, pid in enumerate(unique_pids):
#             # schedule du membre considéré
#             member_sched = hh_data[hh_data["pid"] == pid]
#             # la première row contient déjà les labels
#             first_row = member_sched.iloc[0]

#             for j, col in enumerate(self.label_cols): # loop sur les labels
#                  raw_val = str(first_row[col]) # valeur de l'attribut socio démographique
#                  mapping_col = self.label_dics[col] # son index correspondant
#                  label_grid[i,j] = mapping_col.get(raw_val, 0) # intégration dans la grille 
                 
#             slot_max_durations = torch.full((self.time_steps,), -1.0)
#             for _, row in member_sched.iterrows(): # loop sur le schedule du membre
#                 activity_idx = self.act_to_idx[row["act"]]
#                 start_minute = row["start"]
#                 end_minute = row["end"]

#                 # on identifie les slots touchés par cette activité
               
#                 start_slot = int(start_minute / step_size)
#                 # on retire un epsilon pour gérer le cas pile 10:10 qui appartient au slot d'avant
#                 end_slot = int((end_minute - 0.0001) / step_size)

#                 # Sécurité bornes
#                 start_slot = max(0, min(start_slot, self.time_steps - 1))
#                 end_slot = max(0, min(end_slot, self.time_steps - 1))

#                 # on parcourt tous les slots concernés
#                 for t in range(start_slot, end_slot + 1):
#                     # bornes temporelles du slot actuel 
#                     slot_start_time = t * step_size
#                     slot_end_time = (t + 1) * step_size

#                     # calcul du chevauchement (Overlap)
#                     # Max du début, Min de la fin
#                     intersection_start = max(start_minute, slot_start_time)
#                     intersection_end = min(end_minute, slot_end_time)
                    
#                     duration_overlap = intersection_end - intersection_start

#                     # si cette activité dure plus longtemps dans ce slot
#                     # que celle qu'on avait vue avant, elle gagne la place.
#                     if duration_overlap > slot_max_durations[t]:
#                         slot_max_durations[t] = duration_overlap
#                         grid[i, t] = activity_idx
#         return {"x": grid, "mask": mask, "labels" : label_grid} # return les trois grilles
    
#     def __len__(self):
#             return len(self.households)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_AGE_BINS = [5, 11, 15, 18, 30, 40, 50, 60, 70]


def _age_bin_edges_and_labels(bins):
    edges = [-np.inf] + list(bins) + [np.inf]
    labels = [f"0-{bins[0] - 1}"]
    labels += [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    labels.append(f"{bins[-1]}+")
    return edges, labels


def bin_age_series(age_series: pd.Series, bins: list) -> pd.Series:
    ages = pd.to_numeric(age_series, errors="coerce")
    edges, labels = _age_bin_edges_and_labels(bins)
    binned = pd.cut(
        ages,
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return binned.astype("object").fillna("PAD").astype(str)


def bin_age_value(value, bins: list) -> str:
    return bin_age_series(pd.Series([value]), bins).iloc[0]


class HouseholdDataset(Dataset):

    def __init__(self, config, schedules, attributes):

        
        self.time_steps = int(config.get('time_steps', 144))
        ds_params = config.get('dataset_params', {})
        self.max_members = int(ds_params.get('max_members', 5))
        self.group_col = ds_params.get('group_by', 'hid')
        self.label_cols = list(ds_params.get('label_cols', []))
        self.discretize_age = bool(ds_params.get("discretize_age", False))
        self.age_bins = list(ds_params.get("age_bins", DEFAULT_AGE_BINS))
        self.slot_conflict_mode = str(
            ds_params.get("slot_conflict_mode", config.get("slot_conflict_mode", "winner"))
        ).lower()
        if self.slot_conflict_mode not in {"winner", "diversity"}:
            raise ValueError(
                f"slot_conflict_mode inconnu: {self.slot_conflict_mode}. "
                "Valeurs attendues: winner, diversity."
            )
        
        # Pré-calcul pour conversion temps → slot
        self.step_size = 1440.0 / self.time_steps
        # Bords des slots pour calcul overlap exact
        #[0.0, 10.0, 20.0, ..., 1440.0]
        # ce sont les bords en minutes de chaque slot
        self.slot_edges = np.arange(self.time_steps + 1, dtype=np.float32) * self.step_size
        
        # Copies pour éviter de modifier les originaux
        schedules = schedules.copy()
        attributes = attributes.copy()
        
        # Uniformiser les types pour le merge
        schedules['pid'] = schedules['pid'].astype(str)
        attributes['pid'] = attributes['pid'].astype(str)
        
        acts = sorted(schedules['act'].unique())
        self.act_to_idx = {"PAD": 0} | {act: idx for idx, act in enumerate(acts, start=1)}
        self.idx_to_act = {idx: act for act, idx in self.act_to_idx.items()}
        self.activity_encodings = len(self.act_to_idx) # pour l'embedding
  
        # colonnes de tri
        if 'age' in attributes.columns:
            attributes['_age_sort'] = pd.to_numeric(attributes['age'], errors='coerce').fillna(-1)
        else:
            attributes['_age_sort'] = 0
            
        if 'sex' in attributes.columns:
            attributes['_sex_sort'] = attributes['sex'].astype(str)
        else:
            attributes['_sex_sort'] = ''

        if self.discretize_age and 'age' in attributes.columns:
            print(f"  Discrétisation de l'âge activée (bins={self.age_bins})")
            attributes['age'] = bin_age_series(attributes['age'], self.age_bins)
        
        # encode les attributs
        self.label_dics = {}
        self.label_cardinalities = []
        
        for col in self.label_cols:
            labels = sorted(attributes[col].astype(str).unique())
            mapping = {"PAD": 0} | {lab: i for i, lab in enumerate(labels, start=1)}
            self.label_dics[col] = mapping
            self.label_cardinalities.append(len(mapping))
            # encode directement dans le dataframe
            attributes[f'_enc_{col}'] = attributes[col].astype(str).map(mapping).fillna(0).astype(np.int16)
        
        print(f"  Labels: {len(self.label_cols)} colonnes, cardinalités: {self.label_cardinalities}")
        print(f"  Slot conflict mode: {self.slot_conflict_mode}")
        
        enc_cols = [f'_enc_{col}' for col in self.label_cols]
        sched_cols = ["pid", "act", "start", "end"]
        attr_cols = ["pid", self.group_col, "_age_sort", "_sex_sort"] + enc_cols
        
        # dataframe fusionné
        df = schedules[sched_cols].merge(
            attributes[attr_cols],
            on="pid", 
            how="left"
        )
        
        # Encoder les activités
        df['_act_idx'] = df['act'].astype(str).map(self.act_to_idx).fillna(0).astype(np.uint8)
        print("Tri des données...")
        df.sort_values(
            by=[self.group_col, '_age_sort', '_sex_sort'], 
            ascending=[True, False, True], 
            inplace=True
        )
        # on crée un Dataset niveau household
        # chaque élément représente l'ensemble des membres du ménage
        grouped = df.groupby(self.group_col, sort=False)
        self.num_households = len(grouped)
        
        print(f"  Ménages: {self.num_households}")
        print(f"  Allocation: x[{self.num_households}, {self.max_members}, {self.time_steps}]")
        
        # création des grilles vides
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
        
        enc_col_names = enc_cols
        # remplissage des grilles
        print("  Génération des grilles...")
        # grouped = dataframe groupé par ménage
        # i = index du ménage, hid = id du ménage, hh_group = dataframe du ménage
        for i, (hid, hh_group) in enumerate(tqdm(grouped, desc="  Ménages")):
            
            # groupe par membre sans avoir à filtrer
            pid_groups = hh_group.groupby("pid", sort=False)
            # index du membre
            m_idx = 0
            # pid = id du membre
            for pid, member_df in pid_groups:
                if m_idx >= self.max_members:
                    break
                
                # marquer le membre comme présent
                self.mask_tensor[i, m_idx] = True
                
                # attributs (constant par membre, première ligne suffit)
                first_row = member_df.iloc[0]
                for j, enc_col in enumerate(enc_col_names):
                    self.label_tensor[i, m_idx, j] = first_row[enc_col]
                
                member_df = member_df.assign(
                    _start_num=pd.to_numeric(member_df["start"], errors="coerce"),
                    _end_num=pd.to_numeric(member_df["end"], errors="coerce"),
                ).sort_values(by=["_start_num", "_end_num"], kind="stable")

                acts = member_df["_act_idx"].to_numpy(dtype=np.int16)
                starts = member_df["_start_num"].to_numpy(dtype=np.float32)
                ends = member_df["_end_num"].to_numpy(dtype=np.float32)
                
                # Stocke la durée (en min) de l'activité qui occupe actuellement la case.
                # Initialisé à -1.0 pour être sûr que n'importe quelle vraie durée (>0) l'écrase.
                slot_candidates = [[] for _ in range(self.time_steps)]
                # Stocke l'ID de l'activité gagnante 
                member_grid = np.zeros(self.time_steps, dtype=np.int16)  # 0 = PAD
                
                # Pour chaque activité 
                for seq_idx, (act, start, end) in enumerate(zip(acts, starts, ends)):
                    if np.isnan(start) or np.isnan(end) or int(act) == 0:
                        continue
                    # horaires concernés (discrets)
                    s_slot = int(start / self.step_size)
                    e_slot = int((end - 1e-4) / self.step_size)
                    
                
                    if e_slot < 0 or s_slot >= self.time_steps:
                        continue
                    s_slot = max(0, s_slot)
                    e_slot = min(self.time_steps - 1, e_slot)
                    if s_slot > e_slot:
                        continue
                    
                    # Bords des slots concernés
                    # il faut +2 car on compense le -1 de python et on veut le bord supérieur du dernier slot
                    edges = self.slot_edges[s_slot:e_slot + 2]  
                    
                    # Overlap = min(end, slot_end) - max(start, slot_start)
                    overlap = np.minimum(end, edges[1:]) - np.maximum(start, edges[:-1])
                    overlap = np.maximum(overlap, 0.0)  
                    
                    for local_idx, ov in enumerate(overlap):
                        if ov <= 0.0:
                            continue
                        t = s_slot + local_idx
                        slot_candidates[t].append((seq_idx, int(act), float(ov)))
                
                # résout les conflits de slots (winner simple ou règle d'adjacence en mode diversity)
                resolved_winners = self._resolve_member_slot_winners(slot_candidates)
                # stocke dans le tenseur du ménage/membre
                for t, winner in enumerate(resolved_winners):
                    if winner is not None:
                        member_grid[t] = winner[1]
                self.x_tensor[i, m_idx] = torch.from_numpy(member_grid.astype(np.uint8))
                m_idx += 1
        
        # poids
        self._calculate_weights(schedules)
        
        print("=" * 60)
        print("Dataset prêt")
        print("=" * 60)

    def _winner_max_overlap(self, candidates):
        if not candidates:
            return None

        # mode winner: max overlap (comportement historique)
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

            # Règle demandée:
            # - un seul perdant "disparu" => il gagne
            # - si les deux (ou plus) sont "disparus" => on reste à la majorité (winner)
            # - si le winner est lui-même "disparu" => on reste à la majorité
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

    def _calculate_weights(self, schedules):
        # calcul des poids pour la loss
        counts = schedules['act'].value_counts()
        total = float(counts.sum())
        num_classes = self.activity_encodings
        K = float(num_classes - 2) if num_classes > 2 else 1.0
        
        weights = torch.ones(num_classes, dtype=torch.float32)
        
        print("  Poids des activités:")
        for act, count in counts.items():
            if act in self.act_to_idx:
                idx = self.act_to_idx[act]
                if idx == 0:
                    continue
                
                w = total / (K * float(count))
                weights[idx] = w
                print(f"    {act}: {w:.4f}")
        
        weights = torch.clamp(weights, min=0.1, max=10.0)
        weights[0] = 0.0  # poids 0 pour PAD
        self.activity_weights = weights

    def shape(self):
        
        return (self.max_members, self.time_steps)

    def __len__(self):
        return self.num_households

    def __getitem__(self, idx):
        return {
            "x": self.x_tensor[idx].long(),
            "mask": self.mask_tensor[idx].float(), 
            "labels": self.label_tensor[idx].long() 
        }
