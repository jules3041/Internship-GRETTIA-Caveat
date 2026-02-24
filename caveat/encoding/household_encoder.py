import pandas as pd
import torch
from caveat.encoding.base import BaseEncoder
from caveat.data.household_dataset import HouseholdDataset
# test unitaire puis remonter 
class HouseholdEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        self.config = kwargs
        self.idx_to_act = {}
        self.idx_to_label = {}  # Pour décoder les attributs
        self.time_steps = kwargs.get("time_steps", 144)

        # dataset_params.slot_conflict_mode:
        # - "winner" (default): activité avec overlap max dans le slot
        # - "diversity": si motif 1-2-1 dans un slot, l'activité intermédiaire est priorisée
    def encode(self, schedules, labels=None, **kwargs):
       
        # on bypass l'encoding normal en chargeant nous même les attributs
        attr_path = self.config.get("attributes_path")

            
        print(f"Chargement manuel des attributs depuis {attr_path}")
        attributes = pd.read_csv(attr_path)

        # création du Dataset à partir des deux dataframes (l'encoding se fait à l'intérieur de la classe)
        dataset = HouseholdDataset(self.config, schedules, attributes)

        # nécessaire pour reconstituer les activités
        if hasattr(dataset, "idx_to_act"):
            self.idx_to_act = dataset.idx_to_act

        # Créer les mappings inverses pour décoder les attributs
        if hasattr(dataset, "label_dics") and hasattr(dataset, "label_cols"):
            self.idx_to_label = {}
            for col in dataset.label_cols:
                if col in dataset.label_dics:
                    # Inverser le dictionnaire : {idx: valeur}
                    self.idx_to_label[col] = {idx: val for val, idx in dataset.label_dics[col].items()}
            self.label_cols = dataset.label_cols

        return dataset
    
    
    def decode(self, schedules, argmax = True, mask = None):
        '''
        à partir de schedules générés par le modèle on reconstitue les épisodes d'activités type HTS
        via un algorithme de decoding par membre

        Args:
            schedules: Tensor de shape [B, M, T] ou [B, M, T, Classes]
            argmax: Si True, applique argmax sur la dernière dimension
            mask: Tensor optionnel de shape [B, M] indiquant les membres réels (1) vs fantômes (0)
        '''
        # si les schedules sont sur GPU on passe en CPU pour utiliser Pandas
        if isinstance(schedules, torch.Tensor):
            schedules = schedules.cpu()

        # si un masque est fourni, le passer aussi en CPU
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        # vérifie si on a le mapping index to activity
        if hasattr(self, "idx_to_act"):
            idx_to_act = self.idx_to_act

        # les schedules sont sous forme de probabilité de classe
        # on utilise argmax car on génére (la plus grande probabilité)
        if argmax and schedules.dim() == 4:
            schedules = torch.argmax(schedules, dim=-1)
            
        # on va créer une liste des lignes du dataframe sous forme de dict
        # chaque élement = une activité
        rows = []

        # pour passer de index de temps à temps réel
        timestep_min = 1440 / self.config['time_steps']
        
        # len(schedules) = nombre de ménages
        for i in range(len(schedules)):
            household = schedules[i] # [5, 144]
            # pour chaque ménage, pour chaque membre, on récupère ses activités et on transforme en ligne du DF
            for member_idx in range(household.shape[0]):
                # Si un masque est fourni, vérifier si ce membre est actif
                if mask is not None and mask[i, member_idx] == 0:
                    # Membre fantôme, on le saute complètement
                    continue

                acts = household[member_idx].tolist()

                if sum(acts) == 0: # === PAD
                    continue
                
                # initialisation
                current_act = acts[0]
                start_step = 0
                gen_pid = f"Gen_HH{i}_M{member_idx}"
                for t, act in enumerate(acts):
                    if act != current_act:
                        act_name = idx_to_act[current_act]  # on utilise current_act car c'est l'activité qui vient de se terminer (pas la nouvelle)
                        if act_name != "PAD":
                            rows.append({
                                'pid' : gen_pid,
                                'hid' : i,
                                'member_idx' : member_idx,
                                'act': act_name,
                                'start': start_step * timestep_min,
                                'end': t * timestep_min,
                                'duration' : (t-start_step) * timestep_min,
                            })
                        current_act = act
                        start_step = t
                act_name = idx_to_act[current_act]
                if act_name != "PAD":
                        rows.append({
                            "pid": gen_pid,
                            "hid": i,
                            "member_idx": member_idx,
                            "act": act_name,
                            "start": start_step * timestep_min,
                            "end": 1440,
                            "duration": (self.time_steps - start_step) * timestep_min
                        })
        return pd.DataFrame(rows)

    def decode_attributes(self, attributes_tensor, mask=None):
        '''
        Décode un tenseur d'attributs encodés vers leurs vraies valeurs.

        Args:
            attributes_tensor: Tensor de shape [B, M, N_attrs] avec les attributs encodés
            mask: Tensor optionnel de shape [B, M] pour filtrer les membres fantômes

        Returns:
            DataFrame avec les attributs décodés, avec colonnes hid, member_idx, puis les attributs
        '''
        if isinstance(attributes_tensor, torch.Tensor):
            attributes_tensor = attributes_tensor.cpu()

        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        if not hasattr(self, "idx_to_label") or not self.idx_to_label:
            print("⚠ Pas de mapping de décodage disponible. Retour des valeurs brutes.")
            # Retour des valeurs brutes
            B, M, N = attributes_tensor.shape
            flat_attrs = attributes_tensor.reshape(-1, N).numpy()
            df = pd.DataFrame(flat_attrs)

            hids = [i for i in range(B) for _ in range(M)]
            m_ids = [j for _ in range(B) for j in range(M)]
            df.insert(0, "member_idx", m_ids)
            df.insert(0, "hid", hids)
            return df

        rows = []
        B, M, N_attrs = attributes_tensor.shape

        for i in range(B):
            for j in range(M):
                # Filtrer les membres fantômes
                if mask is not None and mask[i, j] == 0:
                    continue

                row = {"hid": i, "member_idx": j}

                # Décoder chaque attribut
                for attr_idx, col_name in enumerate(self.label_cols):
                    if attr_idx >= N_attrs:
                        break

                    encoded_val = int(attributes_tensor[i, j, attr_idx].item())

                    # Décoder si possible, sinon garder la valeur brute
                    if col_name in self.idx_to_label and encoded_val in self.idx_to_label[col_name]:
                        decoded_val = self.idx_to_label[col_name][encoded_val]
                        row[col_name] = decoded_val
                    else:
                        row[col_name] = encoded_val

                rows.append(row)

        return pd.DataFrame(rows)

