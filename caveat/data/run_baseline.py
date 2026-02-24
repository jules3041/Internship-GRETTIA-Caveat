
# %%
import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import random
from typing import List 
from pathlib import Path
# ==============================================================================
# FONCTION UTILITAIRE POUR LES GROUPES D'ÂGE
# ==============================================================================

def create_age_groups_from_bins(age_col: pl.Expr, bins: list) -> pl.Expr:

    # 1. Moins que le premier bin (0-5)
    expr = pl.when(age_col < bins[0]).then(pl.lit(f"0-{bins[0]-1}"))
    
    # 2. Les tranches intermédiaires
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i+1]
        # ex: 6 <= age < 11 -> "6-10"
        expr = expr.when(
            (age_col >= lower) & (age_col < upper)
        ).then(pl.lit(f"{lower}-{upper-1}"))
        
    # 3. La dernière tranche (70+)
    expr = expr.otherwise(pl.lit(f"{bins[-1]}+"))
    
    return expr

# ==============================================================================
# 1. CONFIGURATION & CONSTANTES (PARAMÈTRES)
# ==============================================================================

class Config:
    # class qui gère les chemins et constantes à passer aux autres composantes
    # chemin des fichiers temporaires de EQASIM
    CACHE_DIR = r"C:\Users\hajjar\Desktop\codebase_exp_eqasim\project\cache"
    
    # fichiers d'entrée

    # population synthéthique
    SAMPLED_POP_PREFIX = "synthesis.population.sampled"
    # income de la population synthéthique
    INCOME_PREFIX = "synthesis.population.income.selected"

    # population cible (EGT) -> sous forme de 3 dataframes (individus, households, trips)
    EGT_PREFIX = "data.hts.egt.filtered"
    
    # trips EQASIM matching individuel pour comparaison
    EQASIM_TRIPS_PREFIX = "synthesis.population.trips"

    # Paramètres de l'algo matching
    N_HOUSEHOLDS_TO_MATCH = None  # nombre de ménages à traiter dans la population synthéthique
    TOP_K = 5                     # taille du pool des meilleurs ménages candidats dans lequel on va sampler uniformément
    SEUIL_CANDIDATS = 15         # seuil minimum de candidats par niveau de relaxation
    N_RUNS = 1                 # nombre de runs pour calculer moyenne ± écart type
    
    # dans EQASIM, on réparti le income en tranche pour le HTS
    INCOME_CLASS_BOUNDS_EGT = [800, 1200, 1600, 2000, 2400, 3000, 3500, 4500, 5500, 1e6]

    # si on ne trouve pas de ménages candidats on souhaite relaxer les contraintes
    # intégrer le type de ménage (dans le recensement)
    # peut être remplacer le household_size par ça ?
    # à rajouter en premier
    RELAX_LEVELS = [
        ["household_size", "number_of_vehicles", "departement_id", "income_class", "urban_type"],
        ["household_size", "number_of_vehicles", "income_class", "urban_type"],  
        ["household_size", "number_of_vehicles", "income_class"],               
        ["household_size", "number_of_vehicles"],                               
        ["household_size"],                                                     
    ]

    # params matrice de coût
    # on digitize l'âge (primaire, college, lycée, actifs par décennies)
    AGE_BINS = [5, 11, 15, 18, 30, 40, 50, 60, 70]
    
    # pour normaliser la différence ordinale
    MAX_CAT_DIST = float(len(AGE_BINS))
    
    # poids de pénalisation dans la fonction de coût de l'individu
    LAMBDA = {
        "age": 1.0,
        "sex": 1.0,
        "socioprofessional_class": 1.0,
        "employed": 1.0,
        "studies": 1.0,
        "commute_mode": 1.0,
    }

    # poids dans la fonction de coût du ménage -> contribution de chaque coût individuel
    W_EMP = 0.0 # >0 pour augmenter de x% l'importance des actifs
    W_STU = 0.0 # >0 pour augmenter de x% l'importance des étudiants

    # buffer de temps (départ / fin) et de duration pour matcher les activités/trips joints
    BUFFER_TIME_SEC = 300     # 5 minutes
    BUFFER_DURATION_SEC = 300 # 5 minutes
    @staticmethod
    def find_file(cache_dir, prefix):
        """
        Trouve un fichier dans cache_dir dont le nom commence par prefix
        et se termine par .p
        
        Args:
            cache_dir: répertoire où chercher
            prefix: préfixe du fichier (partie avant le hash)
            
        Returns:
            nom du fichier trouvé
            
        Raises:
            FileNotFoundError: si aucun fichier trouvé
            ValueError: si plusieurs fichiers correspondent
        """
        cache_path = Path(cache_dir)
        
        # Cherche tous les fichiers .p qui commencent par le préfixe
        pattern = f"{prefix}__*.p"
        matching_files = list(cache_path.glob(pattern))
        
        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"Aucun fichier trouvé avec le préfixe '{prefix}' dans {cache_dir}"
            )
        
        if len(matching_files) > 1:
            files_str = '\n  '.join([f.name for f in matching_files])
            raise ValueError(
                f"Plusieurs fichiers trouvés avec le préfixe '{prefix}':\n  {files_str}\n"
                f"Veuillez être plus spécifique ou nettoyer le cache."
            )
        
        return matching_files[0].name

# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

class DataLoader:
    # class pour charger et nettoyer les données
    def __init__(self, cfg):
        # initialise les constantes et chemins
        self.cfg = cfg
        self._resolve_filenames()
    def _resolve_filenames(self):
        """Résout dynamiquement les noms de fichiers complets"""
        print("--- Résolution des noms de fichiers ---")
        
        self.sampled_pop_file = self.cfg.find_file(
            self.cfg.CACHE_DIR, 
            self.cfg.SAMPLED_POP_PREFIX
        )
        print(f"  Population synthétique: {self.sampled_pop_file}")
        
        self.income_file = self.cfg.find_file(
            self.cfg.CACHE_DIR, 
            self.cfg.INCOME_PREFIX
        )
        print(f"  Income: {self.income_file}")
        
        self.egt_file = self.cfg.find_file(
            self.cfg.CACHE_DIR, 
            self.cfg.EGT_PREFIX
        )
        print(f"  EGT: {self.egt_file}")
        
        self.eqasim_trips_file = self.cfg.find_file(
            self.cfg.CACHE_DIR, 
            self.cfg.EQASIM_TRIPS_PREFIX
        )
        print(f"  EQASIM trips: {self.eqasim_trips_file}")
        print()
    def load_synthetic_pop(self):
        # charge la population synthéthique et enrichit les données
        print("--- Chargement Population Synthétique ---")

        sampled_path = os.path.join(self.cfg.CACHE_DIR, self.sampled_pop_file)
        df_sampled = pd.read_pickle(sampled_path)

        # il manque le household_income, on le rajoute
        income_path = os.path.join(self.cfg.CACHE_DIR, self.income_file)
        df_income = pd.read_pickle(income_path)

        # pour avoir le household income dans le dataframe individus
        df_merge = df_sampled.merge(df_income, on=["household_id"], how="left")

        # si on veut matcher dessus on doit transformer le household_income de la population générée en income_class
        # à vérifier la distribution
        df_merge["income_class"] = np.digitize(
            df_merge["household_income"] / df_merge["consumption_units_y"], 
            self.cfg.INCOME_CLASS_BOUNDS_EGT, 
            right=True
        )

        # passe en polars pour la suite
        df_merge = pl.from_pandas(df_merge)
        
        # clean le merge du income
        df_merge = df_merge.with_columns(
            pl.coalesce(
                [pl.col(c) for c in ["consumption_units_y", "consumption_units_x"] if c in df_merge.columns]
            ).alias("consumption_units")
        )

        return df_merge

    def prepare_dataframes(self, df_merge):
        # sépare le dataframe synthéthique en un dataframe niveau ménage et un dataframe niveau individus
        # permet de matcher avec la structure classique des HTS

        # cf variables individuelles dans le le LateX
        INDIV_COLS = ["person_id", "household_id", "age", "sex", "couple", "commute_mode",
                      "employed", "studies", "socioprofessional_class", "census_person_id"]

        # création du dataframe niveau individuel synthéthique
        df_syn_i = (
            df_merge
            .select([c for c in INDIV_COLS if c in df_merge.columns])
            .unique(subset=["person_id"], keep="first")
        )

        # cf LateX
        HH_COLS = ["household_id", "iris_id", "commune_id", "departement_id", "urban_type",
                   "household_size", "number_of_vehicles", "consumption_units", 
                   "census_household_id", "household_income", 'income_class']
        
        # colonnes ménage présentes
        HH_COLS = [c for c in HH_COLS if c in df_merge.columns]

        # on prend la première valeur des colonnes ménages pour créer le dataset households
        df_syn_h = (
            df_merge
            .group_by("household_id")
            .agg([*[pl.first(c).alias(c) for c in HH_COLS if c != "household_id"]])
        )
        return df_syn_h, df_syn_i

    def load_egt(self):
        # charge l'EGT et rajoute le commute_mode dans les variables individuelles
        print("--- Chargement EGT ---")
        egt_path = os.path.join(self.cfg.CACHE_DIR, self.egt_file)
        # read l'EGT niveau individuel (i), household (h) et trips (t)
        data = pd.read_pickle(egt_path)
        df_egt_h, df_egt_i, df_egt_t = pl.DataFrame(data[0]), pl.DataFrame(data[1]), pl.DataFrame(data[2])

        # on a pas commute_mode dans EGT
        # or on peut le calculer en regardant le premier trip vers work ou education
        # ça nous donne le commute_mode
        df_cm = df_egt_t.filter(
            pl.col("preceding_purpose") == "home", 
            pl.col("following_purpose").is_in(["work","education"])
        ).sort("is_first_trip", descending=True).unique(subset="person_id").select(
            "egt_person_id", "egt_household_id", "mode"
        )

        # merge pour l'avoir dans les variables individuelles
        df_egt_i = df_egt_i.join(
            df_cm, on=["egt_person_id", "egt_household_id"], how="left"
        ).rename({"mode": "commute_mode"})

        # On prend urban_type, mais on s'assure d'avoir une seule ligne par ménage
        df_urban = df_egt_i.select("egt_household_id", "urban_type").unique(subset="egt_household_id")

        df_egt_h = df_egt_h.join(
            df_urban, 
            on="egt_household_id", 
            how="left"
        )

        df_egt_h = df_egt_h.unique(subset=["egt_household_id"], keep="first")

        return df_egt_h, df_egt_i, df_egt_t

    def load_eqasim_trips(self):
        # charge les trips EQASIM pour comparaison
        print("--- Chargement EQASIM (Comparaison) ---")
        EQASIM_path = os.path.join(self.cfg.CACHE_DIR, self.eqasim_trips_file)
        df_eqasim_t = pl.DataFrame(pd.read_pickle(EQASIM_path))
        
        # On a besoin de household_id pour la détections des activités jointes
        # il est contenu au niveau individuel
        pop_path = os.path.join(self.cfg.CACHE_DIR, self.sampled_pop_file)
        df_eqasim_p = pl.DataFrame(pd.read_pickle(pop_path))
        
        # on le récupère (renomme pour convenience)
        df_eqasim_t = df_eqasim_t.join(
            df_eqasim_p.select("person_id", "household_id"), on="person_id", how="left"
        ).rename({'household_id' : 'egt_household_id'})
        
        return df_eqasim_t

# ==============================================================================
# 3. ALGORITHME DE MATCHING
# ==============================================================================

class MatchingAlgo:
    # implémente les calculs et l'optimisation
    def __init__(self, cfg, df_syn_h, df_syn_i, df_egt_h, df_egt_i):
        # reçoit les constantes et les dataframes nettoyés
        self.cfg = cfg

        # cast pour comparaison
        self.df_syn_h_k = self.cast_keys_for_matching(df_syn_h)
        self.df_egt_h_k = self.cast_keys_for_matching(df_egt_h)
        self.df_syn_i_k = self.cast_individual_for_matching(df_syn_i)
        self.df_egt_i_k = self.cast_individual_for_matching(df_egt_i)
        
        # indexation des ménages candidats
        # au lieu de chercher pour chaque ménage synthéthique, les candidats lignes par lignes
        # on crée un dictionnaire (clés = valeurs des variables ménages, valeurs = ID des household de l'EGT)
        self.idx_maps = self.build_candidate_index(self.df_egt_h_k, self.cfg.RELAX_LEVELS)
        
        # pour calculer le coût de toute la famille on groupe chaque individu sur une même ligne
        self.PERSON_COLS = ["person_id", "age", "sex", "socioprofessional_class", "employed", "studies", "commute_mode"]
        self.syn_g, self.syn_g_idx = self.group_persons(self.df_syn_i_k.select(["household_id"] + self.PERSON_COLS), self.PERSON_COLS)
        self.egt_g, self.egt_g_idx = self.group_persons(self.df_egt_i_k.select(["household_id"] + self.PERSON_COLS), self.PERSON_COLS)


    

    def cast_keys_for_matching(self, df):
        # variables numériques
        KEY_INT_COLS = ["household_size", "number_of_vehicles", "income_class"]
        # variables catégorielles
        KEY_CAT_COLS = ["departement_id", "urban_type"]
        
        exprs = []
        for c in ["household_id"] + KEY_INT_COLS:
            # cast en bons types pour matching
            if c in df.columns: exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
        for c in KEY_CAT_COLS:
            if c in df.columns: exprs.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        return df.with_columns(exprs)

    def cast_individual_for_matching(self, df):
        # uniformiser les types
        exprs = []
        for c in ["person_id", "household_id", "age", "socioprofessional_class"]:
            if c in df.columns: exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
        for c in ["employed", "studies"]:
            if c in df.columns: exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
        for c in ["sex", "commute_mode"]:
            if c in df.columns: exprs.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        
        df = df.with_columns(exprs)
        # pour comparer une info manquante on préfére MISSING à null car on ne peut pas comparer deux nulls
        if "commute_mode" in df.columns:
            df = df.with_columns(pl.col("commute_mode").fill_null("__MISSING__").alias("commute_mode"))
        return df

    def build_candidate_index(self, df_egt_h_k, relax_levels):
        # safety check sur les variables de match
        all_cols = sorted({c for lvl in relax_levels for c in lvl})
        all_cols = [c for c in all_cols if c in df_egt_h_k.columns]

        # on construit le set cible
        dfh = df_egt_h_k.select(["household_id"] + all_cols)

        # liste de dictionnaires par niveau de relax
        # 0 = on utilise toutes les colonnes
        # 4 = que household size
        maps = [dict() for _ in relax_levels]
        # boucle sur tous les ménages observés
        for r in dfh.iter_rows(named=True):
            # de quel ménage on parle
            hid = int(r["household_id"])
            
            # boucle sur les niveaux de relax
            # et ajout des valeurs associées à chacune des colonnes du relax level
            for li, cols in enumerate(relax_levels):
                cols_ok = [c for c in cols if c in dfh.columns]
                # valeurs associées : key = (3, 1, "75", 4, "dense")
                key = tuple(r.get(c) for c in cols_ok)
                # sélectionne le dictionnaire lié au niveau
                # crée une liste vide si la clé n'existe pas encore
                # ajoute le household_id qui a la clé
                # dict -> values des variables : ids des households concernés 
                maps[li].setdefault(key, []).append(hid)
        # pour chaque combinaison de valeurs observées, pour chaque niveau de relax, on store les IDs des ménages observés qui correspondent
        return maps

    def get_candidates_for_household(self, h_row: dict, idx_maps, relax_levels):
            # pour un ménage cible synthéthique on récupére les candidats associés
            accumulated_cands = set()
            
            # idx map contient une liste de dictionnaires, un par niveau de relaxation
            # relax_levels contient les colonnes associés à ce niveau de relaxation
            # m = le dictionnaire de niveau courant, cols = la liste de colonnes à utiliser
            # ex : level = 0, m = map0, cols = ["household_size","vehicles","dept","income","urb"]
            for level, (m, cols) in enumerate(zip(idx_maps, relax_levels)):
                
                # récupère les clés du ménage cible (valeurs des variables ménages)
                key = tuple(h_row.get(c) for c in cols)

                # on check s'il existe des IDs candidats associés
                new_cands = m.get(key)
                
                if new_cands:
                    # ajout des nouveaux candidats au set (pour éviter les doublons automatiquement)
                    accumulated_cands.update(new_cands)
                
                # si on a assez de candidats cumulés, on arrête et on retourne le tout
                if len(accumulated_cands) >= self.cfg.SEUIL_CANDIDATS:
                    return list(accumulated_cands), cols, level
            # il faut enlever ceux d'avant car ceux d'après contiennent forcemment ceux d'avant
            # Si on a fini tous les niveaux et qu'on a trouvé quelques candidats (même < seuil)
            # on retourne ce qu'on a. Sinon rien.
            if accumulated_cands:
                return list(accumulated_cands), None, len(relax_levels)
                
            return [], None, None

    def group_persons(self, df_indiv, cols):
        # passer de une ligne par individu à une ligne ménage -> liste de personnes
        df_sorted = df_indiv.sort(["household_id", "person_id"])
        # groupe par ménage et concatene leurs caractéristiques
        df_g = df_sorted.group_by("household_id", maintain_order=True).agg([pl.col(c).alias(c) for c in cols])

        # on construit un dictionnaire household_id : index de la ligne dans df_g afin d'y accéder très rapidement
        hid_list = df_g["household_id"].to_list()
        hid_to_idx = {int(hid): i for i, hid in enumerate(hid_list)}
        return df_g, hid_to_idx

    def get_group_row(self, df_g, hid_to_idx, hid):
        # renvoi le dictionnaire du ménage -> {"household_id": 10,, "person_id": [1,2,3]...}
        i = hid_to_idx.get(int(hid))
        if i is None: return None
        return df_g.row(i, named=True)

    def compute_w_i(self, employed: np.ndarray, studies: np.ndarray) -> np.ndarray:
        # calcul des poids

        # 1 pour tous les membres du ménage
        w = np.ones_like(employed, dtype=float)

        # ajuste le poids pour les étudiants et actifs
        w += self.cfg.W_EMP * (employed == 1).astype(float)
        w += self.cfg.W_STU * (studies == 1).astype(float)
        return w

    def cost_matrix(self, syn: dict, don: dict) -> np.ndarray:
        # calcul de la matrice de coût pour deux ménages (cf LateX) 

        # vectorisation des listes
        a_age = np.asarray(syn["age"], dtype=float)
        b_age = np.asarray(don["age"], dtype=float)
        a_sex = np.asarray(syn["sex"])
        b_sex = np.asarray(don["sex"])
        a_csp = np.asarray(syn["socioprofessional_class"])
        b_csp = np.asarray(don["socioprofessional_class"])
        a_emp = np.asarray(syn["employed"], dtype=int)
        b_emp = np.asarray(don["employed"], dtype=int)
        a_stu = np.asarray(syn["studies"], dtype=int)
        b_stu = np.asarray(don["studies"], dtype=int)
        # commute_mode est string 
        a_mode = np.asarray(syn["commute_mode"])
        b_mode = np.asarray(don["commute_mode"])

        # vérifie que les sizes des ménages matchent (important pour optimisation)
        n = a_age.shape[0]
        if b_age.shape[0] != n:
            raise ValueError(f"Household sizes differ: n_syn={n}, n_don={b_age.shape[0]}")

        # passe l'age en catégories 
        a_age_cat = np.digitize(a_age, self.cfg.AGE_BINS)
        b_age_cat = np.digitize(b_age, self.cfg.AGE_BINS)

        # Calcul de la distance ordinale 
        # Ex: Lycéen (3) vs Primaire (1) = distance de 2
        dist_ordinal = np.abs(a_age_cat[:, None] - b_age_cat[None, :])

        # Normalisation entre 0 et 1
        # Une distance de 1 (catégorie adjacente) vaut 1/9 ~= 0.11
        # Une distance max 1
        age_diff = dist_ordinal / self.cfg.MAX_CAT_DIST
        
        # 0 pénalité si =, 1 si !=
        sex_diff  = (a_sex[:, None]  != b_sex[None, :]).astype(float)
        csp_diff  = (a_csp[:, None]  != b_csp[None, :]).astype(float)
        emp_diff  = (a_emp[:, None]  != b_emp[None, :]).astype(float)
        stu_diff  = (a_stu[:, None]  != b_stu[None, :]).astype(float)
        mode_diff = (a_mode[:, None] != b_mode[None, :]).astype(float)

        # calcul de la distance via les lambdas
        # on a alors la distance pour tous les membres du ménage en fonction de leur affectation
        dist = (
            self.cfg.LAMBDA["age"] * age_diff
            + self.cfg.LAMBDA["sex"] * sex_diff
            + self.cfg.LAMBDA["socioprofessional_class"] * csp_diff
            + self.cfg.LAMBDA["employed"] * emp_diff
            + self.cfg.LAMBDA["studies"] * stu_diff
            + self.cfg.LAMBDA["commute_mode"] * mode_diff
        )
        # calcul des poids
        w = self.compute_w_i(a_emp, a_stu)

        # calcul de la matrice de coût pour tous les membres du ménage en fonction de leur affectation
        C = w[:, None] * dist
        return C

    def assignment_cost_and_mapping(self, C: np.ndarray, syn_person_ids: list[int], don_person_ids: list[int]):
        # à partir de la matrice de coût 
        # on peut résoudre le problème d'apparaiment bijectif

        # r : les i, c : les j
        r, c = linear_sum_assignment(C)
        # calcul le coût de matching de ce ménage
        L = float(C[r, c].sum())

        # matching en IDs pour pouvoir copier les trajets
        mapping = {int(syn_person_ids[i]): int(don_person_ids[j]) for i, j in zip(r, c)}
        return L, mapping

    def match_one_household_topK(self, hid: int):
        # lance la pipeline pour un ménage synthéthique

        # on récupère le ménage cible sous forme de dictionnaire pour match sur les valeurs
        try:
            h_row = self.df_syn_h_k.filter(pl.col("household_id") == hid).row(0, named=True)
        except:
            return None

        # récupère les candidats pour le niveau de relaxation le plus faible
        cands, cols_used, level = self.get_candidates_for_household(h_row, self.idx_maps, self.cfg.RELAX_LEVELS)

        # aucun trouvé : à réfléchir
        if not cands:
            return None

        # on regarde la liste des membres du ménage cible et leurs caractéristiques
        syn_people = self.get_group_row(self.syn_g, self.syn_g_idx, hid)
        if syn_people is None:
            return None

        syn_person_ids = syn_people["person_id"]
        # taille du ménage cible
        n_syn = len(syn_person_ids)

        # prépare les variables pour trouver le candidat avec le meilleur score de dissimilarité
        valid_matches = []
        
        # loop sur chaque candidat et résoud le problème d'optimisation
        for donor_hid in cands:  
            # même format de dictionnaire
            don_people = self.get_group_row(self.egt_g, self.egt_g_idx, int(donor_hid))
            if don_people is None:
                continue
            # match toujours la taille pour l'algo
            if len(don_people["person_id"]) != n_syn:
                continue
            # matrice de coût
            C = self.cost_matrix(syn_people, don_people)

            # résolution du problème avec la meilleure permutation
            L, pmap = self.assignment_cost_and_mapping(C, syn_person_ids, don_people["person_id"])

            # est ce que ce ménage bat ceux d'avant?
            valid_matches.append({
                "donor_hid": int(donor_hid),
                "L": L,
                "map": pmap
            })
            
        if not valid_matches:
            return None

        valid_matches.sort(key=lambda x: x["L"])
        
        selection_pool = valid_matches[:self.cfg.TOP_K]
        # sample uniform
        chosen = random.choice(selection_pool)

        return {
            "syn_household_id": int(hid),
            "donor_household_id": chosen["donor_hid"],
            "relax_level": int(level) if level is not None else -1,
            "L_household": float(chosen["L"]),        
            "person_map": chosen["map"],               
            "nb_candidates": len(cands),
            "household_size": n_syn,
            "pool_size": len(valid_matches)         
        }

    def run_matching_topK(self):
        import time
        # lance la pipeline pour n ménages synthéthiques
        print("--- Démarrage du matching topK ---")
        start_time = time.time()
        # sélectionne tous les ménages cibles
        hids = self.df_syn_h_k.select("household_id").to_series().to_list()
        
        # limite
        if self.cfg.N_HOUSEHOLDS_TO_MATCH:
            hids = hids[:self.cfg.N_HOUSEHOLDS_TO_MATCH]
        # statistiques
        stats_candidates = []       
        stats_relax_levels = []    
        stats_cands_by_level = {} 
        unmatched_households = 0
        rows = []
        # pour chaque ménage on trouve les n candidats et le meilleur parmi les n
        for k, hid in enumerate(hids, start=1):
            out = self.match_one_household_topK(int(hid))
            if out is None:
                unmatched_households += 1
                continue
            nb_cands = out["nb_candidates"]
            lvl = out["relax_level"]
            
            stats_candidates.append(nb_cands)
            stats_relax_levels.append(lvl)
            
            if lvl not in stats_cands_by_level:
                stats_cands_by_level[lvl] = []

            stats_cands_by_level[lvl].append(nb_cands)
            for syn_pid, don_pid in out["person_map"].items():
                rows.append({
                    "syn_household_id": out["syn_household_id"],
                    "donor_household_id": out["donor_household_id"],
                    "relax_level": out["relax_level"],
                    "L_household": out["L_household"],
                    "syn_person_id": int(syn_pid),
                    "donor_person_id": int(don_pid),
                    "nb_candidates": out["nb_candidates"],  
                    "household_size": out["household_size"]
                })

            if k % 100 == 0:
                print(f"processed {k}/{len(hids)} households, mappings={len(rows)}")
        end_time = time.time()
        duration = end_time - start_time
        total_persons_matched = len(rows)
        print(f"Temps d'exécution       : {duration // 60:.0f} min {duration % 60:.0f} sec")
        print(f"Vitesse moyenne         : {len(hids) / duration:.1f} ménages/sec")
        print(f"Ménages traités         : {len(hids)}")
        print(f"Ménages MATCHÉS         : {len(hids) - unmatched_households} ({((len(hids) - unmatched_households)/len(hids))*100:.1f}%)")
        print(f"Ménages NON MATCHÉS     : {unmatched_households} ({unmatched_households/len(hids)*100:.1f}%)")
        print(f"Personnes matchées      : {total_persons_matched}")
        

        cands_arr = np.array(stats_candidates)
        print("STATISTIQUES CANDIDATS :")
        print(f"  Moyenne : {np.mean(cands_arr):.2f}")
        print(f"  Médiane : {np.median(cands_arr):.2f}")
        print(f"  Q1 (25%) : {np.percentile(cands_arr, 25):.2f}")
        print(f"  Q3 (75%) : {np.percentile(cands_arr, 75):.2f}")
        print(f"  Min / Max : {np.min(cands_arr)} / {np.max(cands_arr)}")

        sorted_levels = sorted(stats_cands_by_level.keys())
        from collections import Counter
        level_counts = Counter(stats_relax_levels)
        for lvl in sorted_levels:
            cands_in_lvl = np.array(stats_cands_by_level[lvl])
            count = level_counts[lvl]
            pct = (count / (len(hids) - unmatched_households)) * 100
            
            cols_str = str(self.cfg.RELAX_LEVELS[lvl]) if lvl < len(self.cfg.RELAX_LEVELS) else "Fallback"
            
            print(f"\n[Niveau {lvl}] - {count} ménages ({pct:.1f}%)")
            print(f"  Vars: {cols_str}")
            print(f"  moyenne de candidats : {np.mean(cands_in_lvl):.1f}")
        # dataframe de matching
        return pl.DataFrame(rows)

    def run_matching_algo(self):
        matching_hp = ["TOP_K", "SEUIL_CANDIDATS",
                       "RELAX_LEVELS", "AGE_BINS", "LAMBDA", "W_EMP", "W_STU"]


        params = {hp: getattr(self.cfg, hp) for hp in matching_hp}
        param_str = repr(sorted(params.items()))
        import hashlib
        hash_params = hashlib.md5(param_str.encode('utf-8')).hexdigest()
        filename = f"matching_cache_{hash_params}.parquet"
        cache_path = os.path.join(self.cfg.CACHE_DIR, filename)
        if os.path.exists(cache_path):
            return pl.read_parquet(cache_path)
        else:
            mapping_df = self.run_matching_topK()
        mapping_df.write_parquet(cache_path)
        return mapping_df

    def run_single_matching(self, seed=None):
        if seed is not None:
            random.seed(seed)
        return self.run_matching_topK()

class Validator:
    # run les analyses collectives
    def __init__(self, cfg):
        # import la config pour les dfs
        self.cfg = cfg

    def create_synthetic_trips(self, df_egt_t, mapping_df):
        print("--- Création table trips synthétiques ---")

        # on donne à chaque individu cible sa chaine de déplacements en la copiant de son donneur samplé
        df_syn_t = (
            df_egt_t
            .join(mapping_df, left_on="person_id", right_on="donor_person_id", how="inner")
            .with_columns([
                pl.col("syn_person_id").alias("person_id"),
                pl.col("syn_household_id").alias("household_id"),
            ])
            .drop(["syn_person_id", "syn_household_id", "donor_household_id", "L_household", "relax_level"])
        )
        # garder colonnes pertinentes
        df_syn_t = df_syn_t.select(['person_id', 'household_id',
            'departure_time', 'arrival_time', 'trip_duration', 'activity_duration',
            'following_purpose', 'preceding_purpose', 'is_last_trip', 'is_first_trip', 'mode',  'origin_commune_large', 'destination_commune_large'
        ])
        return df_syn_t.rename({'household_id' : 'egt_household_id'})

    def detect_joint_behaviours(self, df):
        # le but est de détecter les activités jointes, les trips joints, et les escorts trips
        # pour reconstruire
        df_indexed = df.clone().with_row_index("row_id")
        
        # pour les trips joints on doit être dans la même famille + même mode de déplacement
        trips_pairs = df_indexed.join(df_indexed, on=["egt_household_id", "mode"], suffix="_other")
        
        # Filtre de base commun : pas la même personne + départ similaire 
        potential_shared_base = trips_pairs.filter(
            (pl.col("person_id") != pl.col("person_id_other")),
            ((pl.col("departure_time") - pl.col("departure_time_other")).abs() <= self.cfg.BUFFER_TIME_SEC)
        )
        
        # Cas 1 : Si on a le même but = trip joint
        # ICI on garde la contrainte de durée (on voyage ensemble du début à la fin)
        # NEW: origin_commune_large et destination_commune_large doivent être les mêmes
        joint_trips_ids = potential_shared_base.filter(
            (pl.col("following_purpose") == pl.col("following_purpose_other")),
            ((pl.col("trip_duration") - pl.col("trip_duration_other")).abs() <= self.cfg.BUFFER_DURATION_SEC),
            (pl.col("origin_commune_large") == pl.col("origin_commune_large_other")),
            (pl.col("destination_commune_large") == pl.col("destination_commune_large_other"))
        ).select("row_id").unique().to_series().to_list() 
        
        # Cas 2 : Sinon = escort trip (ex: dépose minute)
        # NEW: au moins un des membres doit avoir following_purpose == "escort"
        escort_trips_ids = potential_shared_base.filter(
            (pl.col("following_purpose") != pl.col("following_purpose_other")),
            (
                (pl.col("following_purpose") == "escort") | 
                (pl.col("following_purpose_other") == "escort")
            )
        ).select("row_id").unique().to_series().to_list() 
        
        # activités conjointes -> même but et même famille
        activities_pairs = df_indexed.join(df_indexed, on=["egt_household_id", "following_purpose"], suffix="_other")
        
        # différente personne + arrivée similaire + durée similaire = activité jointe
        # NEW: destination_commune_large doit être la même
        joint_activities_ids = activities_pairs.filter(
            (pl.col("person_id") != pl.col("person_id_other")),
            ((pl.col("arrival_time") - pl.col("arrival_time_other").abs() <= self.cfg.BUFFER_TIME_SEC)),
            ((pl.col("activity_duration") - pl.col("activity_duration_other")).abs() <= self.cfg.BUFFER_DURATION_SEC),
            (pl.col("destination_commune_large") == pl.col("destination_commune_large_other"))
        ).select("row_id").unique().to_series().to_list() 
        
        # binaire sur les trips concernés
        df_result = df_indexed.with_columns([
            pl.col("row_id").is_in(joint_trips_ids).cast(pl.Int8).alias("is_joint_trip"),
            pl.col("row_id").is_in(escort_trips_ids).cast(pl.Int8).alias("is_escort_trip"),
            pl.col("row_id").is_in(joint_activities_ids).cast(pl.Int8).alias("is_joint_activity")
        ]).drop("row_id") 
        
        print(f"--- Detection Complete ---")
        print(f"Joint Trips: {df_result['is_joint_trip'].sum()}")
        print(f"Escort Trips: {df_result['is_escort_trip'].sum()}")
        print(f"Joint Activities: {df_result['is_joint_activity'].sum()}")
        
        return df_result
    def transform_trips_to_continuous_activities_vectorized(self, df_trips: pl.DataFrame):

        df = df_trips.sort(["person_id", "departure_time"])


        
        main_activities = df.select([
            pl.col("person_id"),
            pl.col("following_purpose").alias("Activity"),
            
    
            pl.col("arrival_time").alias("start_time"),
            

            pl.col("arrival_time").shift(-1).over("person_id")
            .fill_null(24 * 3600)
            .alias("end_time")
        ])


        
        initial_activities = (
            df.unique("person_id", keep="first")
            .select([
                pl.col("person_id"),
                pl.col("preceding_purpose").alias("Activity"),
                pl.lit(0.0).alias("start_time"), 
                pl.col("arrival_time").alias("end_time") 
            ])
        )


        df_continuous = (
            pl.concat([initial_activities, main_activities])
            .sort(["person_id", "start_time"]) # On remet dans l'ordre
            .with_columns(
                (pl.col("end_time") - pl.col("start_time")).alias("activity_duration")
            )
        )
        
        return df_continuous

    def compute_multi_model_emd(self, df_ref, df_model_a, df_model_b, label_a="Model A", label_b="Model B"):
        # comparaison EMD de BASELINE et EQASIM vs EGT
        results = []
        total_ref = df_ref.height
        total_a = df_model_a.height
        total_b = df_model_b.height
        def safe_emd(s_ref, s_model):
            # EMD = wassertein
            # ce sont des durées donc on choisit des minutes
            if s_ref.len() == 0 or s_model.len() == 0: return None
            return wasserstein_distance(s_ref.to_list(), s_model.to_list()) / 60.0

        def process_metric(metric_name, filter_col, target_col, unit):
            # pour chaque type de comportement joint (filter_col) on regarde une colonne (durée, start_time) et on calcule l'EMD vs EGT
            d_ref = df_ref.filter(pl.col(filter_col) == 1)
            d_a   = df_model_a.filter(pl.col(filter_col) == 1)
            d_b   = df_model_b.filter(pl.col(filter_col) == 1)
            dist_a = safe_emd(d_ref[target_col], d_a[target_col])
            dist_b = safe_emd(d_ref[target_col], d_b[target_col])
            prop_ref = d_ref.height / total_ref if total_ref > 0 else 0
            prop_a = d_a.height / total_a if total_a > 0 else 0
            prop_b = d_b.height / total_b if total_b > 0 else 0
            # sans trier sur le type d'activité ça nous donne le "combined"
            results.append({
                "Metric": metric_name, "Purpose": "Combined", "Unit": unit,
                f"EMD_{label_a}": dist_a, f"EMD_{label_b}": dist_b,
                "N_Ref": d_ref.height, f"N_{label_a}": d_a.height, f"N_{label_b}": d_b.height,
                "Prop_Ref": prop_ref, f"Prop_{label_a}": prop_a, f"Prop_{label_b}": prop_b
            })
            # pour chaque type d'activité on calcul l'EMD
            purposes = set(d_ref["following_purpose"].unique().to_list()) | set(d_a["following_purpose"].unique().to_list()) | set(d_b["following_purpose"].unique().to_list())
            for p in purposes:
                sub_ref = d_ref.filter(pl.col("following_purpose") == p)[target_col]
                sub_a   = d_a.filter(pl.col("following_purpose") == p)[target_col]
                sub_b   = d_b.filter(pl.col("following_purpose") == p)[target_col]
                dist_p_a = safe_emd(sub_ref, sub_a)
                dist_p_b = safe_emd(sub_ref, sub_b)
                p_p_ref = sub_ref.len() / total_ref if total_ref > 0 else 0
                p_p_a = sub_a.len() / total_a if total_a > 0 else 0
                p_p_b = sub_b.len() / total_b if total_b > 0 else 0
                results.append({
                    "Metric": metric_name, "Purpose": p, "Unit": unit,
                    f"EMD_{label_a}": dist_p_a, f"EMD_{label_b}": dist_p_b,
                    "N_Ref": sub_ref.len(), f"N_{label_a}": sub_a.len(), f"N_{label_b}": sub_b.len(),
                    "Prop_Ref": p_p_ref, f"Prop_{label_a}": p_p_a, f"Prop_{label_b}": p_p_b
                })

        process_metric("Joint Activity Duration", "is_joint_activity", "activity_duration", "Minutes")
        process_metric("Joint Activity Start Time", "is_joint_activity", "arrival_time", "Minutes")
        process_metric("Joint Trip Start Time", "is_joint_trip", "departure_time", "Minutes")
        process_metric("Escort Trip Start Time", "is_escort_trip", "departure_time", "Minutes")

        cols = ["Metric", "Purpose", "Unit", f"EMD_{label_a}", f"EMD_{label_b}", "N_Ref", f"N_{label_a}", f"N_{label_b}", "Prop_Ref", f"Prop_{label_a}", f"Prop_{label_b}"]
        return pd.DataFrame(results)[cols]

    def plot_joint_distributions(self, df_ref, df_model_a, df_model_b, emd_results, label_a="EQASIM", label_b="BASELINE"):
        """
        Affiche les distributions des comportements joints en grille (2x2)
        avec le même style que les marginales (KDE + EMD annotation).
        """
        sns.set_theme(style="ticks", context="talk", font_scale=0.8)
        
        # Configuration des métriques à tracer
        # Clé : Nom dans emd_results -> Valeur : (Colonne Filtre, Colonne Data, Label X, Limite X)
        metrics_map = {
            "Joint Activity Duration": ("is_joint_activity", "activity_duration", "Durée (h)", (0, 6)),
            "Joint Activity Start Time": ("is_joint_activity", "arrival_time", "Heure début (h)", (0, 24)),
            "Joint Trip Start Time": ("is_joint_trip", "departure_time", "Heure départ (h)", (0, 24)),
            "Escort Trip Start Time": ("is_escort_trip", "departure_time", "Heure départ (h)", (0, 24))
        }

        # On filtre pour ne garder que les lignes "Combined" (tous motifs confondus)
        # sinon on aurait trop de graphiques
        emd_subset = emd_results[emd_results["Purpose"] == "Combined"]
        
        # Création de la figure (2 lignes, 2 colonnes)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Couleurs cohérentes avec vos autres graphiques
        c_ref = "black"
        c_a = "tab:blue"      # EQASIM
        c_b = "#D95F02"     # BASELINE (Orange/Rouge)

        for i, (metric_name, config) in enumerate(metrics_map.items()):
            if i >= len(axes): break
            ax = axes[i]
            
            filter_col, target_col, xlabel, xlim = config
            
            # --- 1. Extraction des données (Conversion en Heures / 3600) ---
            def get_vals(df):
                # Filtre : doit être un comportement joint (=1)
                vals = df.filter(pl.col(filter_col) == 1)[target_col].to_numpy()
                return vals / 3600.0 # Conversion en heures pour l'affichage

            d_ref = get_vals(df_ref)
            d_a = get_vals(df_model_a)
            d_b = get_vals(df_model_b)

            # Safety check : pas assez de données
            if len(d_ref) < 10:
                ax.text(0.5, 0.5, "Pas assez de données", ha='center')
                continue

            # --- 2. Plotting (Style Marginales) ---
            
            # A. Référence (EGT) - Aire Grisée
            sns.kdeplot(d_ref, ax=ax, color=c_ref, fill=True, alpha=0.1, linewidth=2, label=f"EGT (N={len(d_ref)})")
            
            # # B. Model A (EQASIM)
            # sns.kdeplot(d_a, ax=ax, color=c_a, linewidth=2.5, linestyle="-", label=label_a)
            
            # C. Model B (BASELINE)
            sns.kdeplot(d_b, ax=ax, color=c_b, linewidth=2.5, linestyle="--", label=label_b)

            # --- 3. Annotations EMD ---
            # Récupération des scores depuis le dataframe de résultats
            row_metrics = emd_subset[emd_subset["Metric"] == metric_name]
            
            if not row_metrics.empty:
                val_a = row_metrics.iloc[0][f'EMD_{label_a}']
                val_b = row_metrics.iloc[0][f'EMD_{label_b}']
                
                # Affichage en haut à droite
                # # EQASIM
                # if pd.notnull(val_a):
                #     ax.text(0.95, 0.90, f"{label_a}: {val_a:.1f} min", 
                #             transform=ax.transAxes, ha='right', color=c_a, fontweight='bold')
                # BASELINE
                if pd.notnull(val_b):
                    ax.text(0.95, 0.83, f"{label_b}: {val_b:.1f} min", 
                            transform=ax.transAxes, ha='right', color=c_b, fontweight='bold')

            # --- 4. Formatting ---
            ax.set_title(metric_name, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Densité")
            ax.set_xlim(xlim)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Formatage axe X (pour les heures 0h, 6h, 12h...)
            if "Time" in metric_name:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%dh'))
            
            # Légende uniquement sur le premier graphe pour ne pas surcharger
            if i == 0:
                ax.legend(loc='upper left', frameon=False, fontsize=10)

        plt.tight_layout()
        plt.show()

class ThemeEvaluator:
    # métriques non collectives des papiers de Tim
    def __init__(self, cfg):
        self.cfg = cfg

    def _safe_emd(self, d1, d2):
        if len(d1) == 0 or len(d2) == 0: return 999.0
        return wasserstein_distance(d1, d2)

    def _get_sequences(self, df):
        
        if "Activity" in df.columns:
             return (
                df.sort(["person_id", "start_time"])
                .with_columns(pl.col("Activity").cast(pl.Utf8)) 
                .group_by("person_id")
                .agg([pl.col("Activity").alias("sequence")])
            )
        else:
            return (
                df.sort(["person_id", "departure_time"])
                .with_columns(pl.col("following_purpose").cast(pl.Utf8)) 
                .group_by("person_id")
                .agg([pl.col("following_purpose").alias("sequence")])
            )

    def compute_participation(self, df_ref, df_model, label):
        # issu de "Participations" du papier sur VAEs

        col_name = "Activity" if "Activity" in df_ref.columns else "following_purpose"
        
        cols_ref = df_ref[col_name].cast(pl.Utf8).unique().to_list()
        cols_mod = df_model[col_name].cast(pl.Utf8).unique().to_list()
        purposes = list(set(cols_ref) | set(cols_mod))
        
        scores = []
        for p in purposes:
            def get_counts(dframe):
                # par type d'activité on calcule le nombre de fois que chaque individu l'effectue 
                counts = dframe.filter(pl.col(col_name).cast(pl.Utf8) == p).group_by("person_id").len()
                all_ids = dframe.select("person_id").unique()
                final = all_ids.join(counts, on="person_id", how="left").fill_null(0)
                return final["len"].to_list() # le nombre de fois 

            c_ref = get_counts(df_ref)
            c_mod = get_counts(df_model)
            scores.append(self._safe_emd(c_ref, c_mod))
        # seulement la moyenne (des EMDs des activités)
        return {"Theme": "Participation", "Metric": "Avg Activity Count EMD", "Label": label, "Score": np.mean(scores)}

    def compute_ngram_transition(self, df_ref, df_model, label, n=2):
        col_name = "Activity" if "Activity" in df_ref.columns else "following_purpose"
        sort_col = "start_time" if "start_time" in df_ref.columns else "departure_time"

        def get_ngrams_dist(df, n_gram):
            df_sorted = df.sort(["person_id", sort_col])
            # chaque activité est un point de départ pour la transition
            cols_to_concat = [pl.col(col_name).cast(pl.Utf8).alias("t0")]

            # on récupère les "n" activités suivantes
            for i in range(1, n_gram):
                cols_to_concat.append(pl.col(col_name).cast(pl.Utf8).shift(-i).over("person_id").alias(f"t{i}"))
            
            # ex : t0,t1 : Travail,Courses PUIS Courses,Maison
            transitions = df_sorted.select(cols_to_concat).drop_nulls()
            # "Travail->Courses" "Courses->Maison"
            ngram_col = pl.concat_str(transitions.columns, separator="->").alias("gram")

            # fréquence dans le jeu de données
            return transitions.select(ngram_col)["gram"].value_counts(normalize=True)

        # on obtient les distributions pour chaque dataframe
        dist_ref = get_ngrams_dist(df_ref, n)
        dist_mod = get_ngrams_dist(df_model, n)
        
        # même dataframe et on compute la distance L1
        comp = dist_ref.join(dist_mod, on="gram", how="full", suffix="_mod").fill_null(0)
        l1_diff = (comp["proportion"] - comp["proportion_mod"]).abs().sum()
        # return la somme des distances L1
        return {"Theme": "Transition", "Metric": f"{n}-gram Distribution L1", "Label": label, "Score": l1_diff}

    def compute_timing(self, df_ref, df_model, label):
            # issu de "Timings" du papier sur VAEs
            
            # colonnes selon le type de DF
            if "start_time" in df_ref.columns:
                start_col, dur_col = "start_time", "activity_duration"
            else:
                start_col, dur_col = "departure_time", "activity_duration"

            # Nettoyage des NaN pour éviter le crash du score
            start_ref = df_ref[start_col].cast(pl.Float64).drop_nulls().to_list()
            start_mod = df_model[start_col].cast(pl.Float64).drop_nulls().to_list()
            dur_ref = df_ref[dur_col].cast(pl.Float64).drop_nulls().to_list()
            dur_mod = df_model[dur_col].cast(pl.Float64).drop_nulls().to_list()

            # en minutes
            emd_start = self._safe_emd(start_ref, start_mod) / 60.0
            emd_dur = self._safe_emd(dur_ref, dur_mod) / 60.0

            return [
                {"Theme": "Timing", "Metric": "Start Time EMD (min)", "Label": label, "Score": emd_start},
                {"Theme": "Timing", "Metric": "Duration EMD (min)", "Label": label, "Score": emd_dur}
            ]

    def compute_diversity(self, df_ref, df_model, label):
        # une séquence de journée = une diversité
        # on regarde combien le modèle en couvre 

        s_ref = self._get_sequences(df_ref)["sequence"]
        s_mod = self._get_sequences(df_model)["sequence"]
        
        # convertit les listes polars en liste python, puis les sous-listes en tuples
        seq_ref = set(tuple(x) for x in s_ref.to_list() if x is not None)
        seq_mod = set(tuple(x) for x in s_mod.to_list() if x is not None)
        
        # calcule les élements en communs dans les deux ensembles
        intersection = seq_ref.intersection(seq_mod)

        # % couvert par le modèle
        coverage = len(intersection) / len(seq_ref) if len(seq_ref) > 0 else 0
        
        return {"Theme": "Diversity", "Metric": "Real Seq Coverage %", "Label": label, "Score": coverage * 100}

    def compute_validity(self, df_model, label):
        # validité
        seq = self._get_sequences(df_model)

        # % finir à la maison 
        score = seq.with_columns(
            (pl.col("sequence").list.last().cast(pl.Utf8).str.to_lowercase() == "home").alias("ends_home")
        )["ends_home"].mean()
        return {"Theme": "Validity", "Metric": "% Ends at Home", "Label": label, "Score": score * 100 if score is not None else 0.0}

    def evaluate_all(self, df_ref, df_model, label_model="Model"):
        # sous forme de dataframe
        print(f"--- Évaluation de {label_model} ---")
        results = []
        results.append(self.compute_participation(df_ref, df_model, label_model))
        results.append(self.compute_ngram_transition(df_ref, df_model, label_model, n=2))
        results.append(self.compute_ngram_transition(df_ref, df_model, label_model, n=3))
        results.extend(self.compute_timing(df_ref, df_model, label_model))
        results.append(self.compute_diversity(df_ref, df_model, label_model))
        results.append(self.compute_validity(df_model, label_model))
        # results.append(self.compute_validity(df_ref, "Reference (Target)")) # Redondant si on le fait à chaque fois
        return pd.DataFrame(results)

class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        sns.set_theme(style="ticks", context="talk", font_scale=0.9)

    def _prepare_single_df(self, df_pl):

        df = df_pl.select([
            pl.col("Activity"), # Doit être présent (format continu)
            (pl.col("start_time") / 3600).alias("Start Time (h)"),
            (pl.col("end_time") / 3600).alias("End Time (h)"),
            (pl.col("activity_duration") / 60).alias("Duration (min)"),
        ]).to_pandas()
        return df

    def get_consistent_activity_colormap(self, list_of_dfs):
        # pour avoir une color map cohérente
        all_activities = set()
        for df in list_of_dfs:
            col = "Activity" if "Activity" in df.columns else "following_purpose"
            acts = df[col].unique().to_list()
            all_activities.update(acts)
        

        all_activities = {a for a in all_activities if a.lower() != 'home'}    
        sorted_acts = sorted(list(all_activities))
        palette = sns.color_palette("bright", len(sorted_acts))
        color_map = dict(zip(sorted_acts, palette))
        
        for h in ["Home", "home", "HOME"]:
            color_map[h] = "#F0F0F0" 
            
        return color_map, sorted_acts

    def plot_single_tempogram(self, ax, df_pl, title, color_map, sorted_activities):

        # granularité
        resolution_min = 1
        n_bins = int((24 * 3600 + 60) / (resolution_min * 60))
        time_axis = np.linspace(0, 24, n_bins)
        
        # ordre d'empilement
        acts_to_plot = sorted_activities.copy()
        curr_acts = df_pl["Activity"].unique().to_list()
        
        has_home = any(a.lower() == 'home' for a in curr_acts)
        if has_home:
            acts_to_plot.append("home")  # home en dernier

        curves = {}
        total_pop = np.zeros(n_bins)
        
        # on passe en temps discret et on calcule le taux de participation de la population à chaque pas de temps pour chaque activité
        for act in acts_to_plot:
            try:
                # passage en discret
                subset = df_pl.filter(pl.col("Activity") == act).select([
                    (pl.col("start_time") / (resolution_min * 60)).round().cast(pl.Int32).alias("start_idx"),
                    (pl.col("end_time") / (resolution_min * 60)).round().cast(pl.Int32).alias("end_idx")
                ])
                
                # safety check : empêche de sortir des bornes
                starts = np.clip(subset["start_idx"].to_numpy(), 0, n_bins - 1)
                ends = np.clip(subset["end_idx"].to_numpy(), 0, n_bins - 1)
                
                # on fait une somme cumulative à chaque début d'activité
                # les deltas sont les changements : les évenements (quelqu'un arrive/part) 
                deltas = np.zeros(n_bins)
                np.add.at(deltas, starts, 1)
                np.add.at(deltas, ends, -1)
                
                # en sommant l'historique on a le nombre de personnes
                counts = np.cumsum(deltas)
                curves[act] = counts
                total_pop += counts
            except:
                curves[act] = np.zeros(n_bins)

        # ratio
        max_pop = np.max(total_pop) if np.max(total_pop) > 0 else 1
        
        stack_values_pct = []
        labels_stack = []
        stack_colors = []
        
        for act in acts_to_plot:
            if act in curves:
                stack_values_pct.append(curves[act] / max_pop)
                labels_stack.append(act.title())
                stack_colors.append(color_map.get(act, "#333"))

        # plot
        ax.stackplot(time_axis, *stack_values_pct, labels=labels_stack, colors=stack_colors, alpha=0.9)
        
       
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.axvline(12, color='black', linestyle=':', alpha=0.3)
        ax.grid(False) 

    def plot_single_distribution_curve(self, ax, df_pl, metric_col, activity, label, color, linestyle, linewidth=2.0):
  
        df = self._prepare_single_df(df_pl)
        
        # filtrage de l'activité
        subset = df[df["Activity"] == activity]
        data = subset[metric_col].dropna()
        data = data[np.isfinite(data)] 
        # pour le KDE on doit avoir un minimum de data et de variation
        if len(data) > 10 and data.std() > 0.01:
            try:
                sns.kdeplot(
                    data=data, 
                    ax=ax, 
                    color=color, 
                    linestyle=linestyle, 
                    linewidth=linewidth,
                    label=label,
                    warn_singular=False, 
                    fill=False
                )
            except Exception as e:
                pass

    def plot_joint_distribution_2d(self, ax, df_pl, activity, cmap="Blues", x_bins=48, y_bins=48):

            df = self._prepare_single_df(df_pl)
            
            # filtrage de l'activité
            subset = df[df["Activity"] == activity]
            

            # données : Start Time (h) vs Duration (h)
            x = subset["Start Time (h)"].values
            y = subset["Duration (min)"].values / 60.0 # en heures pour la lisibilité
            
            # histogramme 2D (densité start time vs durée)
            # range: [[xmin, xmax], [ymin, ymax]]
            # densité dans chaque case
            h = ax.hist2d(x, y, bins=[x_bins, y_bins], range=[[0, 24], [0, 16]], 
                        cmap=cmap, density=True, cmin=0.0001)
            
            ax.set_title(activity.title(), fontsize=12, fontweight='bold')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Durée (h)")
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Heure début (h)")
                
            ax.grid(False)

# ==============================================================================
# 5. CLASSES POUR MÉTRIQUES CONDITIONNELLES
# ==============================================================================

class ConditionalMetrics:

    def __init__(self, config: dict = None, evaluator=None):
        self.config = config or {}
        # config correspond aux constantes + chemnins
        # evaluator permet de réuiliser les fonctions d'évaluation thématiques
        self.evaluator = evaluator if evaluator else ThemeEvaluator(self.config)

    def add_attributes_to_schedules(self, schedules: pl.DataFrame, attributes: pd.DataFrame, join_col: str = "person_id") -> pl.DataFrame:
        # merge les attributs aux schedules
        # Conversion pandas -> polars si besoin
        if isinstance(attributes, pd.DataFrame):
            attrs_pl = pl.from_pandas(attributes)
        else:
            attrs_pl = attributes

        # harmonisation des noms et types (surtout dans caveat au cas où)
        if "pid" in attrs_pl.columns and join_col == "person_id":
            attrs_pl = attrs_pl.rename({"pid": "person_id"})

        if join_col in schedules.columns:
            schedules = schedules.with_columns(pl.col(join_col).cast(pl.Int64))
            attrs_pl = attrs_pl.with_columns(pl.col(join_col).cast(pl.Int64))

        # jointure
        cols_to_keep = [join_col] + [c for c in attrs_pl.columns if c not in schedules.columns and c != join_col]
        return schedules.join(attrs_pl.select(cols_to_keep), on=join_col, how="left")
class ConditionalMetrics:
    """
    """

    def __init__(self, config: dict = None, evaluator=None):
        self.config = config or {}
        self.evaluator = evaluator if evaluator else ThemeEvaluator(self.config)

    def add_attributes_to_schedules(self, schedules: pl.DataFrame, attributes: pd.DataFrame, join_col: str = "person_id") -> pl.DataFrame:
        """Fusionne les attributs (Age, Sexe) aux emplois du temps."""
        if isinstance(attributes, pd.DataFrame):
            attrs_pl = pl.from_pandas(attributes)
        else:
            attrs_pl = attributes

        if "pid" in attrs_pl.columns and join_col == "person_id":
            attrs_pl = attrs_pl.rename({"pid": "person_id"})

        if join_col in schedules.columns:
            schedules = schedules.with_columns(pl.col(join_col).cast(pl.Int64))
            attrs_pl = attrs_pl.with_columns(pl.col(join_col).cast(pl.Int64))

        cols_to_keep = [join_col] + [c for c in attrs_pl.columns if c not in schedules.columns and c != join_col]
        return schedules.join(attrs_pl.select(cols_to_keep), on=join_col, how="left")

    def compute_conditional_metrics(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        ref_attributes: pd.DataFrame,
        model_attributes: pd.DataFrame,
        condition_on: List[str],
        label_model: str = "Model",
        age_bins: list = None
    ) -> pd.DataFrame:
        """
        Calcule les métriques pour chaque attribut dans condition_on.
        """
        df_ref_full = self.add_attributes_to_schedules(df_ref, ref_attributes)
        df_model_full = self.add_attributes_to_schedules(df_model, model_attributes)

        # Gestion dynamique de l'âge
        if "age" in condition_on or "age_group" in condition_on:
            
            if "age_group" not in df_ref_full.columns:
                if "age" in df_ref_full.columns and age_bins is not None:
                    df_ref_full = df_ref_full.with_columns(
                        create_age_groups_from_bins(pl.col("age"), age_bins).alias("age_group")
                    )
        
            if "age_group" not in df_model_full.columns:
                if "age" in df_model_full.columns and age_bins is not None:
                    df_model_full = df_model_full.with_columns(
                        create_age_groups_from_bins(pl.col("age"), age_bins).alias("age_group")
                    )

            if "age" in condition_on:
                condition_on.remove("age")
                if "age_group" not in condition_on:
                    condition_on.append("age_group")

        all_results = []
        # loop sur les attributs et calcul les métriques
        for attr in condition_on:
            if attr not in df_ref_full.columns or attr not in df_model_full.columns:
                continue
            
            # ce sont les mêmes catégories dans les deux datasets normalement
            cats_ref = df_ref_full[attr].unique().to_list()
            cats_mod = df_model_full[attr].unique().to_list()
            all_cats = sorted([c for c in list(set(cats_ref) | set(cats_mod)) if c is not None])
            # on loop sur les valeurs possibles
            for cat in all_cats:
                sub_ref = df_ref_full.filter(pl.col(attr) == cat)
                sub_mod = df_model_full.filter(pl.col(attr) == cat)

                n_ref = sub_ref.select("person_id").n_unique()
                n_model = sub_mod.select("person_id").n_unique()
                # éviter les petits échantillons
                if n_ref < 10 or n_model < 10:
                    continue
                
                # appel des métriques
                # et constituation du tableau final
                p_res = self.evaluator.compute_participation(sub_ref, sub_mod, label_model)
                p_res.update({"Attribute": attr, "Category": str(cat), "N_ref": n_ref})
                all_results.append(p_res)

                t_res = self.evaluator.compute_timing(sub_ref, sub_mod, label_model)
                for t in t_res:
                    t.update({"Attribute": attr, "Category": str(cat), "N_ref": n_ref})
                    all_results.append(t)
                
                v_res = self.evaluator.compute_validity(sub_mod, label_model)
                v_res.update({"Attribute": attr, "Category": str(cat), "N_ref": n_ref})
                all_results.append(v_res)

        return pd.DataFrame(all_results)

class ConditionalVisualizer:
    # pour visualiser les métriques conditionnelles
    def __init__(self, config: dict = None):
        self.config = config or {}
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)

    def plot_heatmap(self, df_results, attribute, metric_name, model_name="Model"):
        # Heatmap des scores par catégorie pour un attribut et une métrique donnés.
        # Filtrer
        subset = df_results[
            (df_results["Attribute"] == attribute) & 
            (df_results["Metric"].str.contains(metric_name, case=False))
        ]
        
        if subset.empty:
            print(f"Rien à afficher pour {attribute} / {metric_name}")
            return

        # Lignes=Metric, Colonnes=Category
        pivot = subset.pivot_table(index="Metric", columns="Category", values="Score")
        
        # tri des colonnes
    
        cols = sorted(pivot.columns)
        pivot = pivot[cols]

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={'label': 'Erreur (EMD/L1)'})
        plt.title(f"Performance par {attribute} - {metric_name} ({model_name})")
        plt.tight_layout()
        plt.show()

    def plot_distributions_by_condition(self, df_ref, df_model, ref_attrs, mod_attrs, condition_attr, age_bins, activity="work"):
        cm = ConditionalMetrics()
        df_r = cm.add_attributes_to_schedules(df_ref, ref_attrs)
        df_m = cm.add_attributes_to_schedules(df_model, mod_attrs)

        if condition_attr == "age_group":
            df_r = df_r.with_columns(create_age_groups_from_bins(pl.col("age"), age_bins).alias("age_group"))
            df_m = df_m.with_columns(create_age_groups_from_bins(pl.col("age"), age_bins).alias("age_group"))

        cats = sorted([c for c in df_r[condition_attr].unique().to_list() if c is not None])
        
        # Création des sous-plots
        n_cols = 4
        n_rows = (len(cats) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        col_act = "Activity" if "Activity" in df_r.columns else "following_purpose"
        # on loop sur les catégories uniques de l'attribut et on regarde la distribution du start_time
        for i, cat in enumerate(cats):
            ax = axes[i]
            
            sr = df_r.filter((pl.col(condition_attr) == cat) & (pl.col(col_act) == activity))["start_time"].to_numpy() / 3600
            sm = df_m.filter((pl.col(condition_attr) == cat) & (pl.col(col_act) == activity))["start_time"].to_numpy() / 3600
            
            if len(sr) > 10:
                sns.kdeplot(sr, ax=ax, fill=True, color="black", alpha=0.1, label="Ref")
            if len(sm) > 10:
                sns.kdeplot(sm, ax=ax, color="tab:orange", linestyle="--", label="Model")
            
            ax.set_title(f"{cat} (N={len(sr)})")
            ax.set_xlim(0, 24)
            if i == 0: ax.legend()

        plt.suptitle(f"Heure de début : {activity.upper()} par {condition_attr}")
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 6. FONCTIONS POUR ÉVALUATION MULTI-RUNS
# ==============================================================================

def aggregate_scores_multi_runs(list_of_score_dfs):
    all_scores = pd.concat(list_of_score_dfs, ignore_index=True)
    # le groupby permet d'obtenir les lignes correspondant à une métrique pour les df de chaque run (concat)
    # le agg faire mean et std
    agg = all_scores.groupby(["Theme", "Metric", "Label"])["Score"].agg(["mean", "std"]).reset_index() 
    agg["Score (mean ± std)"] = agg.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}" if pd.notnull(r['std']) else f"{r['mean']:.2f}",
        axis=1
    )
    return agg


def aggregate_emd_multi_runs(list_of_emd_dfs, label_a="EQASIM", label_b="BASELINE"):
    all_emd = pd.concat(list_of_emd_dfs, ignore_index=True)
    group_cols = ["Metric", "Purpose", "Unit"]
    # le groupby permet d'obtenir les lignes correspondant à une métrique pour les df de chaque run (concat)
    # ici on agg différements (comptage et EMD)
    agg = all_emd.groupby(group_cols).agg({
        f"EMD_{label_a}": ["mean", "std"],
        f"EMD_{label_b}": ["mean", "std"],
        "N_Ref": "mean",
        f"N_{label_a}": "mean",
        f"N_{label_b}": "mean",
    }).reset_index()
    # pandas crée des tuples comme ("EMD_EQASIM", "mean")
    # on les transforme en "EMD_EQASIM_mean"
    agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg.columns]

    # formattage
    agg[f"EMD_{label_a} (mean ± std)"] = agg.apply(
        lambda r: f"{r[f'EMD_{label_a}_mean']:.2f} ± {r[f'EMD_{label_a}_std']:.2f}"
        if pd.notnull(r[f'EMD_{label_a}_std']) else f"{r[f'EMD_{label_a}_mean']:.2f}",
        axis=1
    )
    agg[f"EMD_{label_b} (mean ± std)"] = agg.apply(
        lambda r: f"{r[f'EMD_{label_b}_mean']:.2f} ± {r[f'EMD_{label_b}_std']:.2f}"
        if pd.notnull(r[f'EMD_{label_b}_std']) else f"{r[f'EMD_{label_b}_mean']:.2f}",
        axis=1
    )
    return agg


def run_multiple_evaluations(cfg, loader, df_syn_h, df_syn_i, df_egt_h, df_egt_i, df_egt_t, n_runs=10):
    print(f"\n{'='*60}")
    print(f"   LANCEMENT DE {n_runs} RUNS POUR STATISTIQUES")
    print(f"{'='*60}\n")

    all_baseline_scores = []
    all_emd_results = []

    val = Validator(cfg)
    paper_eval = ThemeEvaluator(cfg)
    df_eqasim_t = loader.load_eqasim_trips()

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        engine = MatchingAlgo(cfg, df_syn_h, df_syn_i, df_egt_h, df_egt_i)
        # seed différent à chaque run pour varier le matching
        mapping_df = engine.run_single_matching(seed=run_idx * 42)

        df_syn_t = val.create_synthetic_trips(df_egt_t, mapping_df)
        df_egt_continuous = val.transform_trips_to_continuous_activities_vectorized(df_egt_t)
        df_syn_continuous = val.transform_trips_to_continuous_activities_vectorized(df_syn_t)
        df_eqasim_continuous = val.transform_trips_to_continuous_activities_vectorized(df_eqasim_t)

        cols_comp = ['person_id', 'egt_household_id', 'departure_time', 'arrival_time', 'trip_duration',
                     'activity_duration', 'following_purpose', 'preceding_purpose', 'is_last_trip',
                     'is_first_trip', 'mode', 'origin_commune_large', 'destination_commune_large']
        df_egt_to_compare = val.detect_joint_behaviours(df_egt_t.select(cols_comp))
        df_syn_to_compare = val.detect_joint_behaviours(df_syn_t)
        df_eqasim_to_compare = val.detect_joint_behaviours(df_eqasim_t)

        emd_results = val.compute_multi_model_emd(
            df_ref=df_egt_to_compare,
            df_model_a=df_eqasim_to_compare,
            df_model_b=df_syn_to_compare,
            label_a="EQASIM",
            label_b="BASELINE"
        )
        all_emd_results.append(emd_results)

        df_scores_baseline = paper_eval.evaluate_all(
            df_ref=df_egt_continuous,
            df_model=df_syn_continuous,
            label_model="BASELINE"
        )
        all_baseline_scores.append(df_scores_baseline)

        print(f"  Run {run_idx + 1}: OK ({mapping_df.height} personnes matchées)")

    print(f"\n{'='*60}")
    print(f"   RÉSULTATS AGRÉGÉS ({n_runs} runs)")
    print(f"{'='*60}\n")

    if all_baseline_scores:
        agg_scores = aggregate_scores_multi_runs(all_baseline_scores)
        print("\n--- Scores BASELINE (moyenne ± écart type) ---")
        print(agg_scores[["Theme", "Metric", "Label", "Score (mean ± std)"]].to_string(index=False))

    if all_emd_results:
        agg_emd = aggregate_emd_multi_runs(all_emd_results, label_a="EQASIM", label_b="BASELINE")
        print("\n--- EMD Comportements Joints (moyenne ± écart type) ---")
        cols_display = ["Metric", "Purpose", "EMD_EQASIM (mean ± std)", "EMD_BASELINE (mean ± std)"]
        print(agg_emd[cols_display].to_string(index=False))

    return agg_scores if all_baseline_scores else None, agg_emd if all_emd_results else None


if __name__ == "__main__":
    cfg = Config()
    loader = DataLoader(cfg)

    # chargement
    df_syn_pop_merged = loader.load_synthetic_pop()
    df_syn_h, df_syn_i = loader.prepare_dataframes(df_syn_pop_merged)
    df_egt_h, df_egt_i, df_egt_t = loader.load_egt()

    # si N_RUNS > 1 : évaluation multi-runs avec moyenne ± écart type
    if cfg.N_RUNS > 1:
        agg_scores, agg_emd = run_multiple_evaluations(
            cfg, loader, df_syn_h, df_syn_i, df_egt_h, df_egt_i, df_egt_t,
            n_runs=cfg.N_RUNS
        )
        print("\n--- Évaluation multi-runs terminée ---")
    else:
        # matching simple (1 run)
        engine = MatchingAlgo(cfg, df_syn_h, df_syn_i, df_egt_h, df_egt_i)
        mapping_df = engine.run_matching_algo()

        print(f"Nb personnes mappées: {mapping_df.height}")

    if cfg.N_RUNS == 1 and not mapping_df.is_empty():
        # copie les trips
        val = Validator(cfg)
        df_syn_t = val.create_synthetic_trips(df_egt_t, mapping_df)
        df_eqasim_t = loader.load_eqasim_trips()
        

        print("--- Transformation en activités continues ---")
        df_egt_continuous = val.transform_trips_to_continuous_activities_vectorized(df_egt_t)
        df_syn_continuous = val.transform_trips_to_continuous_activities_vectorized(df_syn_t)
        df_eqasim_continuous = val.transform_trips_to_continuous_activities_vectorized(df_eqasim_t)

        cols_comp = ['person_id', 'egt_household_id', 'departure_time', 'arrival_time', 'trip_duration', 'activity_duration', 'following_purpose', 'preceding_purpose', 'is_last_trip', 'is_first_trip', 'mode', 'origin_commune_large', 'destination_commune_large']
        
        df_egt_to_compare = val.detect_joint_behaviours(df_egt_t.select(cols_comp))
        df_syn_to_compare = val.detect_joint_behaviours(df_syn_t)
        df_eqasim_to_compare = val.detect_joint_behaviours(df_eqasim_t)
        emd_results = val.compute_multi_model_emd(
            df_ref=df_egt_to_compare, 
            df_model_a=df_eqasim_to_compare, 
            df_model_b=df_syn_to_compare,
            label_a="EQASIM",
            label_b="BASELINE"
        )
        print(emd_results)
        val.plot_joint_distributions(df_egt_to_compare, df_eqasim_to_compare, df_syn_to_compare, emd_results, label_a="EQASIM", label_b="BASELINE")
        print("\n" + "="*50)
        print("   ÉVALUATION (BASELINE vs EGT)")

        paper_eval = ThemeEvaluator(cfg)
        
        
        df_scores_paper = paper_eval.evaluate_all(
            df_ref=df_egt_continuous,      
            df_model=df_syn_continuous,    
            label_model="BASELINE"
        )

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n--- Tableau des Scores (BASELINE) ---")
        print(df_scores_paper)
        
        df_scores_paper2 = paper_eval.evaluate_all(
            df_ref=df_egt_continuous,      
            df_model=df_eqasim_continuous,    
            label_model="EQASIM"
        )
        print("\n--- Tableau des Scores (EQASIM) ---")
        print(df_scores_paper2)


        vis = Visualizer(cfg)
        

        models_to_plot = [
            {"name": "EGT",       "df": df_egt_continuous,     "color": "black",   "style": "-",  "lw": 2.5},
            {"name": "BASELINE", "df": df_syn_continuous,     "color": "#D95F02", "style": "--", "lw": 2.5},
        ]


        print("--- Visualisation : participation ---")
        
        #  map de couleur cohérente pour tous les DF
        all_dfs = [m["df"] for m in models_to_plot]
        color_map, sorted_activities = vis.get_consistent_activity_colormap(all_dfs)

        # création de la figure 
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)

        # boucle d'appel pour chaque modèle
        for i, model in enumerate(models_to_plot):
            vis.plot_single_tempogram(
                ax=axes[i], 
                df_pl=model["df"], 
                title=model["name"], 
                color_map=color_map, 
                sorted_activities=sorted_activities
            )

            if i == 0: axes[i].set_ylabel("Proportion Population")
            else: axes[i].set_ylabel("")
            axes[i].set_xlabel("Heure (h)")

        # légende commune
        handles, labels = axes[-1].get_legend_handles_labels()
        # inverser pour avoir home en haut 
        fig.legend(handles[::-1], labels[::-1], loc='center right', bbox_to_anchor=(0.99, 0.5), title="Activités")
        plt.subplots_adjust(right=0.85, wspace=0.1)
        plt.show()


        print("--- Visualisation : Distributions ---")

        metrics_cfg = [
            ("Start Time (h)", (0, 24)), 
            ("End Time (h)", (0, 30)), 
            ("Duration (min)", (0, 600))
        ]
        
        activities = sorted_activities # calculé précédemment
        
        # création de la grille (lignes = Métriques, colonnes = activités)
        n_rows = len(metrics_cfg)
        n_cols = len(activities)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey="row")
        
        # boucle sur la grille
        for col_idx, act in enumerate(activities):
            for row_idx, (metric_name, xlim) in enumerate(metrics_cfg):
                
                ax = axes[row_idx, col_idx]
                
                # boucle sur chaque modèle pour superposer les courbes sur cet axe
                for model in models_to_plot:
                    # label que sur le tout premier plot
                    label = model["name"] if (row_idx == 0 and col_idx == 0) else None
                    if model["name"] != "EGT":
                        try:
                            # Calcul EMD pour chaque activité et chaque métrique
                            raw_col = None
                            if "Start Time" in metric_name:
                                raw_col = "start_time"
                            elif "End Time" in metric_name:
                                raw_col = "end_time"
                            elif "Duration" in metric_name:
                                raw_col = "activity_duration"

                            if raw_col:
                                
                                ref_data = df_egt_continuous.filter(
                                    pl.col("Activity") == act
                                ).select(pl.col(raw_col)).drop_nulls().to_series().to_numpy() / 60.0

                                
                                mod_data = model["df"].filter(
                                    pl.col("Activity") == act
                                ).select(pl.col(raw_col)).drop_nulls().to_series().to_numpy() / 60.0

                                
                                if len(ref_data) > 0 and len(mod_data) > 0:
                                    dist = wasserstein_distance(ref_data, mod_data)
                                    color_text = model["color"]
                                    
                                    y_pos = 0.95 - (0.10 * models_to_plot.index(model))
                                    
                                   
                                    ax.text(0.96, y_pos, f"EMD: {dist:.1f} min", 
                                            transform=ax.transAxes, 
                                            ha='right', va='top', 
                                            fontsize=12,        
                                            fontweight='bold', 
                                            )

                        except Exception as e:
                            pass
                    vis.plot_single_distribution_curve(
                        ax=ax,
                        df_pl=model["df"],
                        metric_col=metric_name, 
                        activity=act,
                        label=label,
                        color=model["color"],
                        linestyle=model["style"],
                        linewidth=model["lw"]
                    )
                

                ax.set_xlim(xlim)
                ax.grid(True, linestyle=':', alpha=0.5)
                
    
                if row_idx == 0:
                    ax.set_title(act.upper(), fontsize=12, fontweight="bold")
                
        
                if col_idx == 0:
                    ax.set_ylabel(metric_name.split(" ")[0], fontweight="bold")
                else:
                    ax.set_ylabel("")
                    

                if row_idx == n_rows - 1:
                    ax.set_xlabel(metric_name)
                else:
                    ax.set_xlabel("")


        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(models_to_plot), frameon=False, fontsize=14)
        
        plt.tight_layout()
        plt.show()

        print("--- Visualisation : Distributions Jointes (Start vs Duration) ---")
        
        # quelques activités clés à analyser
        target_acts = ["work", "shop", "education", "home"]
        
        n_models = len(models_to_plot)
        n_acts = len(target_acts)
        
        # Grille : lignes = modèles, colonnes = activités
        fig, axes = plt.subplots(n_models, n_acts, figsize=(4 * n_acts, 3.5 * n_models), sharex=True, sharey=True)

        # loop pour plot individuel
        for row_idx, model in enumerate(models_to_plot):
            for col_idx, act in enumerate(target_acts):
                ax = axes[row_idx, col_idx]
                

                cmaps = ["Blues", "Oranges", "Greens", "Purples"]
                cmap_name = cmaps[row_idx % len(cmaps)]
                
                vis.plot_joint_distribution_2d(
                    ax=ax, 
                    df_pl=model["df"], 
                    activity=act, 
                    cmap=cmap_name
                )
                

                if col_idx == 0:
                    ax.text(-0.3, 0.5, model["name"], transform=ax.transAxes, 
                            va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)
                

                if row_idx == 0:
                    ax.set_title(act.upper(), fontsize=14, fontweight='bold')
                else:
                    ax.set_title("") 

        plt.tight_layout()
        plt.show()

        print("\n" + "="*50)
        print("   ÉVALUATION CONDITIONNELLE")

        df_ref_enriched = df_egt_i.join(
            df_egt_h, on="household_id", how="left"
        )
        ref_attrs = df_ref_enriched.to_pandas()

        df_mod_enriched = df_syn_i.join(
            df_syn_h, on="household_id", how="left"
        )
        model_attrs = df_mod_enriched.to_pandas()
        cond_eval = ConditionalMetrics(cfg, evaluator=paper_eval)
        cond_vis = ConditionalVisualizer(cfg)

        # Calcul des métriques conditionnelles
        df_cond = cond_eval.compute_conditional_metrics(
            df_ref=df_egt_continuous,
            df_model=df_syn_continuous,
            ref_attributes=ref_attrs,
            model_attributes=model_attrs,
            condition_on=["age", "sex", "employed", "socioprofessional_class", "number_of_vehicles"], 
            age_bins=cfg.AGE_BINS,                   
            label_model="BASELINE"
        )
        print("\n--- Tableau Synthétique Conditionnel ---")
            
        # 1. Filtrer pour ne garder que les 3 métriques d'intérêt
        # On utilise str.contains pour matcher "Avg Activity Count", "Start Time", "Duration"
        df_subset = df_cond[
            df_cond["Metric"].str.contains("Avg Activity Count|Start Time|Duration", case=False, regex=True)
        ].copy()
        # 2. Nettoyer les noms des métriques pour qu'ils soient courts en colonne
        metric_map = {
            "Avg Activity Count EMD": "Participation (%)",
            "Start Time EMD (min)": "Start Time (min)",
            "Duration EMD (min)": "Duration (min)"
        }
        # On applique le mapping si le nom exact est trouvé
        df_subset["Metric_Short"] = df_subset["Metric"].map(metric_map).fillna(df_subset["Metric"])
        # 3. Pivot : 
        # Index = [Attribute, Category, N_ref]
        # Colonnes = Metric_Short
        # Valeurs = Score
        pivot_table = df_subset.pivot_table(
            index=["Attribute", "Category", "N_ref"], 
            columns="Metric_Short", 
            values="Score"
        ).reset_index()
        # 4. Formattage et Affichage
        # On trie pour que ce soit joli (par attribut puis catégorie)
        pivot_table = pivot_table.sort_values(by=["Attribute", "Category"])
        display(pivot_table)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', '{:.2f}'.format) # 2 décimales    

        if not df_cond.empty:
            cond_vis.plot_heatmap(df_cond, attribute="socioprofessional_class", metric_name="Count", model_name="BASELINE")
            
            cond_vis.plot_distributions_by_condition(
                df_ref=df_egt_continuous,
                df_model=df_syn_continuous,
                ref_attrs=ref_attrs,
                mod_attrs=model_attrs,
                condition_attr="socioprofessional_class",
                age_bins=cfg.AGE_BINS,
                activity="work"
            )
            
            cond_vis.plot_distributions_by_condition(
                df_ref=df_egt_continuous,
                df_model=df_syn_continuous,
                ref_attrs=ref_attrs,
                mod_attrs=model_attrs,
                condition_attr="socioprofessional_class",
                age_bins=cfg.AGE_BINS,
                activity="education"
            )
        # table avec les N et les EMD par activité. peut être N en proportion qu'en absolu. proportion de fois où c'est observé
        # corriger le bug puis mettre les graphiques intéressants dans les slides


# %%
pivot_table[pivot_table["Attribute"].isin(["age_group","socioprofessional_class", "number_of_vehicles"])]

# %%
import polars as pl
import os
def export_attributes_to_caveat_csv(
    df_persons: pl.DataFrame, 
    df_households: pl.DataFrame, 
    output_path: str
):
    if "egt_household_id" in df_persons.columns:
        p = df_persons.rename({
            "person_id": "pid", 
            "egt_household_id": "hid"
        })
        
        h = df_households.rename({
            "egt_household_id": "hid"
        })
    if "household_id" in df_persons.columns:
        p = df_persons.rename({
            "person_id": "pid", 
            "household_id": "hid"
        })
        
        h = df_households.rename({
            "household_id": "hid"
        })

    cols_menage = [
        "hid", 
        "household_size", 
        "number_of_vehicles", 
        "income_class", 
        "urban_type", 
        "departement_id"
    ]
    

    existing_h_cols = [c for c in cols_menage if c in h.columns]
    h_clean = h.select(existing_h_cols)

    df_merged = p.join(h_clean, on="hid", how="left")

    cols_indiv = [
        "pid", "hid", 
        "age", "sex", "employed", "studies", 
        "socioprofessional_class", "commute_mode"
    ]
    
    final_cols = cols_indiv + [c for c in existing_h_cols if c != "hid"]
    
    final_cols_safe = [c for c in final_cols if c in df_merged.columns]

    df_caveat_attr = df_merged.select(final_cols_safe)

    print(f"Lignes générées : {df_caveat_attr.height}")
    print(df_caveat_attr.head())
    
    df_caveat_attr.write_csv(output_path)
    print("Export attributs terminé.")

    return df_caveat_attr
import polars as pl
import os

import polars as pl
import os

def export_schedules_for_caveat(
    df_trips: pl.DataFrame, 
    df_attributes: pl.DataFrame, 
    output_path: str
):
    print(f"--- Préparation CAVEAT Schedules : {output_path} ---")
    
    # Indexation des trips si nécessaire
    if "trip_id" not in df_trips.columns:
        df = df_trips.with_row_index("trip_id")
    else:
        df = df_trips

    # 1. Lignes de TRIPS
    # act = "trip"
    # trip_id = le trip_id réel
    trips_rows = df.select([
        pl.col("person_id").alias("pid"),
        pl.lit("trip").alias("act"),
        (pl.col("departure_time") / 60).cast(pl.Int64).alias("start"),
        (pl.col("arrival_time") / 60).cast(pl.Int64).alias("end"),
        pl.col("trip_id").cast(pl.Int64)
    ])

    # 2. Lignes d'ACTIVITÉS
    # act = following_purpose
    # trip_id = le trip_id qui a mené à cette activité (pour faire le lien plus tard)
    stationary_rows = df.select([
        pl.col("person_id").alias("pid"),
        pl.col("following_purpose").cast(pl.String).str.to_lowercase().alias("act"),
        (pl.col("arrival_time") / 60).cast(pl.Int64).alias("start"),
        (pl.col("departure_time").shift(-1).over("person_id") / 60).fill_null(1440).cast(pl.Int64).alias("end"),
        pl.col("trip_id").cast(pl.Int64) 
    ])

    # 3. Activité INITIALE (avant le 1er trip)
    # trip_id = -1 (pas de trip précédent)
    initial_rows = df.unique("person_id", keep="first").select([
        pl.col("person_id").alias("pid"),
        pl.col("preceding_purpose").cast(pl.String).str.to_lowercase().alias("act"),
        pl.lit(0).cast(pl.Int64).alias("start"),
        (pl.col("departure_time") / 60).cast(pl.Int64).alias("end"),
        pl.lit(-1).cast(pl.Int64).alias("trip_id")
    ])

    # 4. Concaténation
    df_schedules = (
        pl.concat([initial_rows, trips_rows, stationary_rows])
        .sort(["pid", "start"])
        .filter(pl.col("start") < 1440)
        .with_columns(pl.col("end").clip(0, 1440))
        .filter(pl.col("end") > pl.col("start"))
    )

    # 5. Gestion des IMMOBILES
    all_pids = df_attributes.select("pid").unique()
    mobile_pids = df_schedules.select("pid").unique()
    immobile_pids = all_pids.join(mobile_pids, on="pid", how="anti")

    if immobile_pids.height > 0:
        print(f"Ajout de {immobile_pids.height} personnes immobiles.")
        df_immobiles = immobile_pids.select([
            pl.col("pid"),
            pl.lit("home").alias("act"),
            pl.lit(0).cast(pl.Int64).alias("start"),
            pl.lit(1440).cast(pl.Int64).alias("end"),
            pl.lit(-1).cast(pl.Int64).alias("trip_id")
        ])
        df_final = pl.concat([df_schedules, df_immobiles])
    else:
        df_final = df_schedules

    # 6. AJOUT DU SCHEDULE_ID UNIQUE
    # C'est ce qui permettra de lier la table d'interactions proprement
    df_final = df_final.sort(["pid", "start"]).with_row_index("schedule_id")

    df_final.drop("trip_id").write_csv(output_path)
    print("Export schedules terminé.")
    return df_final


# path_csv_attr = os.path.join(cfg.CACHE_DIR, "observed_attributes_fixed.csv")
# df_attributes_ready = export_attributes_to_caveat_csv(df_egt_i, df_egt_h, path_csv_attr)

# path_csv_schedules = os.path.join(cfg.CACHE_DIR, "observed_schedules_w_trips.csv")
# export_trips_to_caveat_csv_with_trips(df_egt_t, df_attributes_ready, path_csv_schedules)

def fix_household_size(df_attributes: pl.DataFrame) -> pl.DataFrame:
    """
    force la colonne 'household_size' à être égale au nombre réel de membres présents.
    """
    print("Correction des household_size...")
    
    # Compter le nombre réel
    real_sizes = (
        df_attributes
        .group_by("hid")
        .agg(pl.col("pid").n_unique().alias("real_size"))
    )
    
    # Mettre à jour la colonne
    df_fixed = (
        df_attributes
        .drop("household_size")
        .join(real_sizes, on="hid", how="left")
        .rename({"real_size": "household_size"}) 
    )
    
    return df_fixed


# df_attributes_fixed = fix_household_size(df_attributes_ready)
# df_attributes_fixed.write_csv(path_csv_attr)
def generate_interaction_table_corrected(
    df_schedules: pl.DataFrame, 
    df_trips_rich: pl.DataFrame, 
    cfg
) -> pl.DataFrame:
    """
    Génère la table: schedule_id -> interaction_type
    0: Rien
    1: Joint Activity (sur une ligne d'activité)
    2: Joint Trip (sur une ligne trip)
    3: Escort (sur une ligne trip)
    """
    print("--- Génération de la table des interactions (Corrigée) ---")

    # A. DÉTECTION (identique à avant, basée sur les trips riches)
    if "trip_id" not in df_trips_rich.columns:
        df_rich = df_trips_rich.with_row_index("trip_id")
    else:
        df_rich = df_trips_rich
        
    trips_pairs = df_rich.join(df_rich, on=["egt_household_id", "mode"], suffix="_other")
    potential_shared = trips_pairs.filter(
        (pl.col("person_id") != pl.col("person_id_other")),
        ((pl.col("departure_time") - pl.col("departure_time_other")).abs() <= cfg.BUFFER_TIME_SEC)
    )

    # Listes des trip_ids concernés
    joint_trips_ids = potential_shared.filter(
        (pl.col("following_purpose") == pl.col("following_purpose_other")),
        ((pl.col("trip_duration") - pl.col("trip_duration_other")).abs() <= cfg.BUFFER_DURATION_SEC),
        (pl.col("origin_commune_large") == pl.col("origin_commune_large_other")),
        (pl.col("destination_commune_large") == pl.col("destination_commune_large_other"))
    ).select("trip_id").unique().to_series()

    escort_trips_ids = potential_shared.filter(
        (pl.col("following_purpose") != pl.col("following_purpose_other")),
        ((pl.col("following_purpose") == "escort") | (pl.col("following_purpose_other") == "escort"))
    ).select("trip_id").unique().to_series()

    # Pour l'activité conjointe, on regarde l'activité QUI SUIT le trip
    act_pairs = df_rich.join(df_rich, on=["egt_household_id", "following_purpose"], suffix="_other")
    joint_act_trip_ids = act_pairs.filter(
        (pl.col("person_id") != pl.col("person_id_other")),
        ((pl.col("arrival_time") - pl.col("arrival_time_other")).abs() <= cfg.BUFFER_TIME_SEC),
        ((pl.col("activity_duration") - pl.col("activity_duration_other")).abs() <= cfg.BUFFER_DURATION_SEC),
        (pl.col("destination_commune_large") == pl.col("destination_commune_large_other"))
    ).select("trip_id").unique().to_series()

    # B. APPLICATION SUR LES SCHEDULE_IDS
    # On part de la table schedules qui contient tout le monde
    df_res = df_schedules.select(["schedule_id", "trip_id", "act", "pid"]).with_columns(
        pl.lit(0).alias("interaction_type")
    )

    # CAS 1 : Joint Activity (1)
    # C'est une activité (pas un trip) DONT le trip_id est dans joint_act_trip_ids
    df_res = df_res.with_columns(
        pl.when(
            (pl.col("act") != "trip") & 
            (pl.col("trip_id").is_in(joint_act_trip_ids))
        )
        .then(1)
        .otherwise(pl.col("interaction_type"))
        .alias("interaction_type")
    )

    # CAS 2 : Joint Trip (2)
    # C'est un trip (act == 'trip') ET son trip_id est dans joint_trips_ids
    df_res = df_res.with_columns(
        pl.when(
            (pl.col("act") == "trip") & 
            (pl.col("trip_id").is_in(joint_trips_ids))
        )
        .then(2)
        .otherwise(pl.col("interaction_type"))
        .alias("interaction_type")
    )

    # CAS 3 : Escort (3)
    # C'est un trip ET son trip_id est dans escort_trips_ids
    # (Priorité sur Joint Trip si conflit)
    df_res = df_res.with_columns(
        pl.when(
            (pl.col("act") == "trip") & 
            (pl.col("trip_id").is_in(escort_trips_ids))
        )
        .then(3)
        .otherwise(pl.col("interaction_type"))
        .alias("interaction_type")
    )

    return df_res.select(["schedule_id", "interaction_type", "pid"])

# %%
# 1. Préparation trips indexés
if "trip_id" not in df_egt_t.columns:
    df_trips_indexed = df_egt_t.with_row_index("trip_id")
else:
    df_trips_indexed = df_egt_t

# 2. Table ATTRIBUTES
path_csv_attr = os.path.join(cfg.CACHE_DIR, "caveat_attributes.csv")
df_attributes_ready = export_attributes_to_caveat_csv(df_egt_i, df_egt_h, path_csv_attr)
df_attributes_fixed = fix_household_size(df_attributes_ready)
df_attributes_fixed.write_csv(path_csv_attr)

# 3. Table SCHEDULES (génère les schedule_id)
path_csv_schedules = os.path.join(cfg.CACHE_DIR, "caveat_schedules.csv")
df_schedules_final = export_schedules_for_caveat(
    df_trips=df_trips_indexed, 
    df_attributes=df_attributes_fixed, 
    output_path=path_csv_schedules
)

# 4. Table INTERACTIONS (liée par schedule_id)
path_csv_interactions = os.path.join(cfg.CACHE_DIR, "caveat_interactions.csv")
df_interactions = generate_interaction_table_corrected(
    df_schedules=df_schedules_final,
    df_trips_rich=df_trips_indexed, # On a besoin des infos riches (communes, modes...)
    cfg=cfg
)

# Validation
print("\nDistribution des interactions :")
print(df_interactions["interaction_type"].value_counts().sort("interaction_type"))

# On vérifie la cohérence Act vs Trip
check = df_schedules_final.join(df_interactions, on="schedule_id")
print("\nVérification cohérence (Type Act vs Interaction):")
print(check.group_by(["act", "interaction_type"]).len().sort(["act", "interaction_type"]))

df_interactions.write_csv(path_csv_interactions)
print(f"Export interactions terminé : {path_csv_interactions}")



