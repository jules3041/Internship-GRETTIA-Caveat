
"""
module d'évaluation des déplacements et activités conjoints au sein du ménage.

détecte et analyse les comportements conjoints (déplacements conjoints,
accompagnements, activités conjointes) dans les schedules de ménage.

basé sur les méthodes de la classe Validator de baseline.py :
- detect_joint_behaviours()
- compute_multi_model_emd()
- plot_joint_distributions()

fonctionne avec le format de schedule où "trip" est un type d'activité.
utilisé par train_hh_disc pour évaluer si le modèle reproduit bien les
interactions entre membres du ménage (est-ce qu'ils se déplacent ensemble, etc.)
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import wasserstein_distance
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union


class JointTripEvaluator:
    """
    évaluateur des comportements conjoints dans les schedules de ménage.

    détecte :
    - déplacements conjoints : même ménage, horaires de déplacement qui se chevauchent, même destination
    - accompagnements : même ménage, horaires qui se chevauchent, destinations différentes
    - activités conjointes : même ménage, même type d'activité, horaires qui se chevauchent
    """

    def __init__(
        self,
        buffer_time_min: float = 5.0,          # tolérance sur l'heure de début du déplacement/activité (en minutes)
        buffer_duration_min: float = 5.0,       # tolérance sur la durée (en minutes)
        trip_activity_name: str = "trip",       # nom de l'activité déplacement dans les schedules
        household_col: str = "hid",             # nom de la colonne identifiant le ménage
    ):
        self.buffer_time_min = buffer_time_min
        self.buffer_duration_min = buffer_duration_min
        self.trip_activity_name = trip_activity_name
        self.household_col = household_col

    def prepare_schedule_df(self, df: pd.DataFrame) -> pl.DataFrame:
        """
        convertit un DataFrame pandas en polars et s'assure que les colonnes nécessaires existent.
        """
        # colonnes obligatoires
        required = ['pid', 'act', 'start', 'end']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ajout de la durée si pas présente
        df = df.copy()
        if 'duration' not in df.columns:
            df['duration'] = df['end'] - df['start']

        # on s'assure que la colonne ménage existe (sinon on l'extrait du pid)
        if self.household_col not in df.columns:
            if 'hid' in df.columns:
                self.household_col = 'hid'
            elif df['pid'].astype(str).str.contains('_').any():
                # extraction du hid depuis le format pid type "Gen_HH123_M0"
                df['hid'] = df['pid'].astype(str).str.extract(r'HH(\d+)')[0].astype(int)
                self.household_col = 'hid'
            else:
                raise ValueError(f"Could not find or derive household column '{self.household_col}'")

        return pl.from_pandas(df)

    def extract_trips_and_activities(
        self,
        df_pl: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        sépare les déplacements des autres activités pour l'analyse.

        pour les déplacements, on a besoin d'identifier :
        - preceding_purpose : activité avant le déplacement
        - following_purpose : activité après le déplacement (= destination)

        Returns:
            tuple (trips_df, activities_df)
        """
        # tri par personne et heure de début
        df_sorted = df_pl.sort(['pid', 'start'])

        # identification des déplacements
        trips = df_sorted.filter(
            pl.col('act').str.to_lowercase() == self.trip_activity_name.lower()
        )

        # activités hors déplacements
        activities = df_sorted.filter(
            pl.col('act').str.to_lowercase() != self.trip_activity_name.lower()
        )

        # pour les déplacements, on ajoute l'activité précédente et suivante (= destination)
        if trips.height > 0:
            # ajout du preceding_purpose (activité avant le déplacement) et du following_purpose
            trips = trips.with_columns([
                pl.col('act').shift(1).over('pid').alias('preceding_purpose_temp'),
                pl.col('act').shift(-1).over('pid').alias('following_purpose_temp'),
            ])

            # on rejoint le dataframe complet pour récupérer les vraies activités contextuelles
            # lag et lead pour chaque ligne
            df_with_context = df_sorted.with_columns([
                pl.col('act').shift(1).over('pid').alias('prev_act'),
                pl.col('act').shift(-1).over('pid').alias('next_act'),
            ])

            # on filtre les déplacements et on récupère leur contexte
            trips_with_context = df_with_context.filter(
                pl.col('act').str.to_lowercase() == self.trip_activity_name.lower()
            ).with_columns([
                pl.col('prev_act').fill_null('home').alias('preceding_purpose'),
                pl.col('next_act').fill_null('home').alias('following_purpose'),
            ]).drop(['prev_act', 'next_act'])

            trips = trips_with_context

        return trips, activities

    def detect_joint_behaviours(
        self,
        df: Union[pd.DataFrame, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        détecte les déplacements conjoints, accompagnements et activités conjointes.

        critères de détection :
        - déplacement conjoint : même ménage + horaires qui se chevauchent + même destination
        - accompagnement : même ménage + horaires qui se chevauchent + destinations différentes
        - activité conjointe : même ménage + même type d'activité + horaires qui se chevauchent

        Args:
            df: DataFrame de schedules avec colonnes [pid, hid, act, start, end, duration]

        Returns:
            DataFrame avec colonnes ajoutées : is_joint_trip, is_escort_trip, is_joint_activity
        """
        if isinstance(df, pd.DataFrame):
            df_pl = self.prepare_schedule_df(df)
        else:
            df_pl = df

        # séparation des déplacements et des activités
        trips, activities = self.extract_trips_and_activities(df_pl)

        # initialisation du résultat avec les indices de lignes
        df_indexed = df_pl.with_row_index("row_id")

        joint_trip_ids = set()
        escort_trip_ids = set()
        joint_activity_ids = set()

        # détection des déplacements conjoints et accompagnements
        if trips.height > 0:
            trips_indexed = trips.with_row_index("trip_row_id")

            # auto-jointure sur le ménage pour trouver les paires potentielles
            trip_pairs = trips_indexed.join(
                trips_indexed,
                on=[self.household_col],
                suffix="_other"
            )

            # filtre : personnes différentes, horaires et durées similaires (dans le buffer)
            potential_shared = trip_pairs.filter(
                (pl.col("pid") != pl.col("pid_other")) &
                ((pl.col("start") - pl.col("start_other")).abs() <= self.buffer_time_min) &
                ((pl.col("duration") - pl.col("duration_other")).abs() <= self.buffer_duration_min)
            )

            # déplacements conjoints : même destination
            if 'following_purpose' in potential_shared.columns:
                joint_trips = potential_shared.filter(
                    pl.col("following_purpose") == pl.col("following_purpose_other")
                )
                joint_trip_ids = set(joint_trips["trip_row_id"].unique().to_list())

                # accompagnements : destinations différentes
                escort_trips = potential_shared.filter(
                    pl.col("following_purpose") != pl.col("following_purpose_other")
                )
                escort_trip_ids = set(escort_trips["trip_row_id"].unique().to_list())

        # détection des activités conjointes (hors déplacements)
        if activities.height > 0:
            activities_indexed = activities.with_row_index("act_row_id")

            # auto-jointure sur le ménage et le type d'activité
            activity_pairs = activities_indexed.join(
                activities_indexed,
                on=[self.household_col, 'act'],
                suffix="_other"
            )

            # filtre : personnes différentes, horaires et durées similaires (dans le buffer)
            joint_activities = activity_pairs.filter(
                (pl.col("pid") != pl.col("pid_other")) &
                ((pl.col("start") - pl.col("start_other")).abs() <= self.buffer_time_min) &
                ((pl.col("duration") - pl.col("duration_other")).abs() <= self.buffer_duration_min)
            )

            joint_activity_ids = set(joint_activities["act_row_id"].unique().to_list())

        # remapping vers les indices originaux
        # on a besoin de savoir quelles lignes du df original correspondent aux comportements détectés
        df_sorted = df_pl.sort(['pid', 'start']).with_row_index("sort_id")

        # création des colonnes booléennes
        is_trip = df_sorted['act'].str.to_lowercase() == self.trip_activity_name.lower()

        # indices des déplacements dans l'ordre trié
        trip_mask = df_sorted.filter(is_trip).with_row_index("trip_idx")
        activity_mask = df_sorted.filter(~is_trip).with_row_index("act_idx")

        # construction du résultat avec les flags booléens
        result = df_sorted.with_columns([
            pl.lit(False).alias("is_joint_trip"),
            pl.lit(False).alias("is_escort_trip"),
            pl.lit(False).alias("is_joint_activity"),
        ])

        # marquage des déplacements conjoints
        if joint_trip_ids:
            joint_trip_sort_ids = trip_mask.filter(
                pl.col("trip_idx").is_in(list(joint_trip_ids))
            )["sort_id"].to_list()
            result = result.with_columns(
                pl.when(pl.col("sort_id").is_in(joint_trip_sort_ids))
                .then(True)
                .otherwise(pl.col("is_joint_trip"))
                .alias("is_joint_trip")
            )

        # marquage des accompagnements
        if escort_trip_ids:
            escort_trip_sort_ids = trip_mask.filter(
                pl.col("trip_idx").is_in(list(escort_trip_ids))
            )["sort_id"].to_list()
            result = result.with_columns(
                pl.when(pl.col("sort_id").is_in(escort_trip_sort_ids))
                .then(True)
                .otherwise(pl.col("is_escort_trip"))
                .alias("is_escort_trip")
            )

        # marquage des activités conjointes
        if joint_activity_ids:
            joint_act_sort_ids = activity_mask.filter(
                pl.col("act_idx").is_in(list(joint_activity_ids))
            )["sort_id"].to_list()
            result = result.with_columns(
                pl.when(pl.col("sort_id").is_in(joint_act_sort_ids))
                .then(True)
                .otherwise(pl.col("is_joint_activity"))
                .alias("is_joint_activity")
            )

        # nettoyage des colonnes temporaires
        result = result.drop("sort_id")

        print(f"--- Joint Behavior Detection Complete ---")
        print(f"Joint Trips: {result['is_joint_trip'].sum()}")
        print(f"Escort Trips: {result['is_escort_trip'].sum()}")
        print(f"Joint Activities: {result['is_joint_activity'].sum()}")

        return result

    def compute_joint_metrics_emd(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        label_model: str = "Model"
    ) -> pd.DataFrame:
        """
        calcule la distance de Wasserstein (EMD) pour les métriques de comportements conjoints.

        métriques calculées :
        - durée des activités conjointes
        - heure de début des activités conjointes
        - heure de début des déplacements conjoints
        - heure de début des accompagnements

        Args:
            df_ref: DataFrame de référence avec les flags de comportements conjoints
            df_model: DataFrame du modèle avec les flags de comportements conjoints
            label_model: label du modèle dans les résultats

        Returns:
            DataFrame avec les scores EMD et les effectifs
        """
        results = []

        total_ref = df_ref.height
        total_model = df_model.height

        def safe_emd(s_ref: pl.Series, s_model: pl.Series) -> Optional[float]:
            """calcul de l'EMD avec gestion des cas où il n'y a pas assez de données."""
            if s_ref.len() == 0 or s_model.len() == 0:
                return None
            ref_list = s_ref.to_list()
            model_list = s_model.to_list()
            if len(ref_list) < 2 or len(model_list) < 2:
                return None
            return wasserstein_distance(ref_list, model_list)

        def process_metric(
            metric_name: str,
            filter_col: str,
            target_col: str,
            unit: str,
            purpose_col: str = "act"
        ):
            """traite une métrique unique pour le calcul de l'EMD."""
            # filtrage sur les lignes pertinentes
            d_ref = df_ref.filter(pl.col(filter_col) == True)
            d_model = df_model.filter(pl.col(filter_col) == True)

            if d_ref.height == 0 and d_model.height == 0:
                return

            # EMD combiné (toutes destinations confondues)
            if target_col in d_ref.columns and target_col in d_model.columns:
                dist = safe_emd(d_ref[target_col], d_model[target_col])
                prop_ref = d_ref.height / total_ref if total_ref > 0 else 0
                prop_model = d_model.height / total_model if total_model > 0 else 0

                results.append({
                    "Metric": metric_name,
                    "Purpose": "Combined",
                    "Unit": unit,
                    "EMD": dist,
                    "N_Ref": d_ref.height,
                    f"N_{label_model}": d_model.height,
                    "Prop_Ref": prop_ref,
                    f"Prop_{label_model}": prop_model,
                })

                # EMD par destination/activité
                if purpose_col in d_ref.columns:
                    purposes = set(d_ref[purpose_col].unique().to_list()) | \
                               set(d_model[purpose_col].unique().to_list() if purpose_col in d_model.columns else [])

                    for p in purposes:
                        if p is None or str(p).lower() == self.trip_activity_name.lower():
                            continue

                        sub_ref = d_ref.filter(pl.col(purpose_col) == p)[target_col]
                        sub_model = d_model.filter(pl.col(purpose_col) == p)[target_col] if purpose_col in d_model.columns else pl.Series([])

                        dist_p = safe_emd(sub_ref, sub_model)
                        p_ref = sub_ref.len() / total_ref if total_ref > 0 else 0
                        p_model = sub_model.len() / total_model if total_model > 0 else 0

                        results.append({
                            "Metric": metric_name,
                            "Purpose": str(p),
                            "Unit": unit,
                            "EMD": dist_p,
                            "N_Ref": sub_ref.len(),
                            f"N_{label_model}": sub_model.len(),
                            "Prop_Ref": p_ref,
                            f"Prop_{label_model}": p_model,
                        })

        # traitement de chaque type de métrique
        process_metric("Joint Activity Duration", "is_joint_activity", "duration", "Minutes")
        process_metric("Joint Activity Start Time", "is_joint_activity", "start", "Minutes")
        process_metric("Joint Trip Start Time", "is_joint_trip", "start", "Minutes")
        process_metric("Escort Trip Start Time", "is_escort_trip", "start", "Minutes")

        if not results:
            return pd.DataFrame()

        cols = ["Metric", "Purpose", "Unit", "EMD", "N_Ref", f"N_{label_model}",
                "Prop_Ref", f"Prop_{label_model}"]
        return pd.DataFrame(results)[cols]

    def plot_joint_distributions(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        emd_results: pd.DataFrame,
        write_path: Path,
        label_ref: str = "Reference",
        label_model: str = "Model",
        min_samples: int = 10,
    ):
        """
        trace les distributions des comportements conjoints en comparant référence et modèle.

        pour chaque métrique (durée, heure de début), on trace les KDE côte à côte
        pour voir visuellement si le modèle reproduit bien les patterns de la référence.

        Args:
            df_ref: DataFrame de référence avec les flags de comportements conjoints
            df_model: DataFrame du modèle avec les flags
            emd_results: DataFrame des résultats EMD
            write_path: répertoire de sauvegarde des plots
            label_ref: label de la référence dans les plots
            label_model: label du modèle dans les plots
            min_samples: nombre minimum de samples pour tracer (sinon on skip)
        """
        if emd_results.empty:
            print("No EMD results to plot")
            return

        sns.set_theme(style="whitegrid")

        metric_config = {
            "Joint Activity Duration": ("is_joint_activity", "duration"),
            "Joint Activity Start Time": ("is_joint_activity", "start"),
            "Joint Trip Start Time": ("is_joint_trip", "start"),
            "Escort Trip Start Time": ("is_escort_trip", "start"),
        }

        plots_created = 0

        for _, row in emd_results.iterrows():
            metric_name = row['Metric']
            purpose = row['Purpose']

            if metric_name not in metric_config:
                continue

            filter_col, target_col = metric_config[metric_name]

            def get_data(df: pl.DataFrame, purpose_val: str) -> np.ndarray:
                base_filter = pl.col(filter_col) == True
                if purpose_val != "Combined":
                    final_filter = base_filter & (pl.col("act") == purpose_val)
                else:
                    final_filter = base_filter

                filtered = df.filter(final_filter)
                if target_col in filtered.columns:
                    return filtered[target_col].to_numpy()
                return np.array([])

            data_ref = get_data(df_ref, purpose)
            data_model = get_data(df_model, purpose)

            # pas assez de données pour tracer, on saute
            if len(data_ref) < min_samples or len(data_model) < min_samples:
                continue

            # création du plot
            fig, ax = plt.subplots(figsize=(10, 5))

            # distribution de référence
            sns.kdeplot(
                data_ref, ax=ax,
                color="black", fill=True, alpha=0.1, linewidth=2,
                label=f"{label_ref} (N={len(data_ref)})"
            )

            # distribution du modèle
            emd_val = row['EMD']
            n_model = row.get(f'N_{label_model}', len(data_model))
            label_str = f"{label_model} (EMD: {emd_val:.2f} min, N={int(n_model)})" if pd.notnull(emd_val) else f"{label_model} (N={int(n_model)})"
            sns.kdeplot(
                data_model, ax=ax,
                color="tab:orange", linewidth=2, linestyle="--",
                label=label_str
            )

            ax.set_title(f"{metric_name} - {purpose}", fontsize=14, pad=15)
            ax.set_ylabel("Density")
            ax.legend()

            # formatage de l'axe x selon le type de métrique
            if "Start Time" in metric_name:
                ax.set_xlabel("Time of Day (Hours)")
                ax.set_xlim(0, 1440)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f"{int(x // 60)}h")
                )
                ax.xaxis.set_major_locator(ticker.MultipleLocator(240))  # Every 4 hours
            elif "Duration" in metric_name:
                ax.set_xlabel("Duration (Hours)")
                p99 = np.percentile(data_ref, 99)
                ax.set_xlim(0, max(p99 * 1.2, 60))
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f"{x / 60:.1f}h")
                )

            plt.tight_layout()

            # sauvegarde du plot
            safe_purpose = purpose.replace(" ", "_").replace("/", "_")
            safe_metric = metric_name.replace(" ", "_")
            filename = f"joint_dist_{safe_metric}_{safe_purpose}.png"
            plt.savefig(write_path / filename, dpi=150, bbox_inches='tight')
            plt.close()
            plots_created += 1

        print(f"Created {plots_created} joint distribution plots in {write_path}")

    def create_summary_table(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        label_model: str = "Model"
    ) -> pd.DataFrame:
        """
        crée un tableau récapitulatif des statistiques de comportements conjoints.
        compare les proportions ref vs modèle pour chaque type de comportement.
        """
        def get_stats(df: pl.DataFrame, label: str) -> dict:
            total = df.height
            n_trips = df.filter(pl.col('act').str.to_lowercase() == self.trip_activity_name.lower()).height
            n_joint_trips = df['is_joint_trip'].sum() if 'is_joint_trip' in df.columns else 0
            n_escort_trips = df['is_escort_trip'].sum() if 'is_escort_trip' in df.columns else 0
            n_joint_activities = df['is_joint_activity'].sum() if 'is_joint_activity' in df.columns else 0

            return {
                "Label": label,
                "Total Activities": total,
                "Total Trips": n_trips,
                "Joint Trips": n_joint_trips,
                "Joint Trip %": n_joint_trips / n_trips * 100 if n_trips > 0 else 0,
                "Escort Trips": n_escort_trips,
                "Escort Trip %": n_escort_trips / n_trips * 100 if n_trips > 0 else 0,
                "Joint Activities": n_joint_activities,
                "Joint Activity %": n_joint_activities / (total - n_trips) * 100 if (total - n_trips) > 0 else 0,
            }

        ref_stats = get_stats(df_ref, "Reference")
        model_stats = get_stats(df_model, label_model)

        return pd.DataFrame([ref_stats, model_stats])

    def evaluate(
        self,
        df_ref: pd.DataFrame,
        df_model: pd.DataFrame,
        write_path: Path,
        label_model: str = "Model",
        generate_plots: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        pipeline complet d'évaluation des comportements conjoints.

        lance la détection des comportements conjoints sur la ref et le modèle,
        calcule les EMD, crée le tableau récapitulatif et génère les plots.

        Args:
            df_ref: DataFrame des schedules de référence
            df_model: DataFrame des schedules générés par le modèle
            write_path: répertoire de sauvegarde des résultats
            label_model: label du modèle
            generate_plots: si True, génère les plots de distribution

        Returns:
            dictionnaire avec 'emd_results', 'summary', 'ref_detected', 'model_detected'
        """
        write_path = Path(write_path)
        write_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Joint Trip Evaluation: Reference vs {label_model}")
        print(f"{'='*50}")

        # détection des comportements conjoints dans la référence
        print("\n[1/4] Detecting joint behaviors in Reference...")
        df_ref_detected = self.detect_joint_behaviours(df_ref)

        # détection des comportements conjoints dans le modèle
        print(f"\n[2/4] Detecting joint behaviors in {label_model}...")
        df_model_detected = self.detect_joint_behaviours(df_model)

        # calcul des métriques EMD
        print("\n[3/4] Computing EMD metrics...")
        emd_results = self.compute_joint_metrics_emd(
            df_ref_detected,
            df_model_detected,
            label_model
        )

        # création du récapitulatif
        summary = self.create_summary_table(
            df_ref_detected,
            df_model_detected,
            label_model
        )

        # sauvegarde des résultats
        if not emd_results.empty:
            emd_results.to_csv(write_path / "joint_behavior_emd.csv", index=False)
        summary.to_csv(write_path / "joint_behavior_summary.csv", index=False)

        print("\n[4/4] Summary:")
        print(summary.to_string(index=False))

        # génération des plots
        if generate_plots and not emd_results.empty:
            print(f"\nGenerating distribution plots...")
            self.plot_joint_distributions(
                df_ref_detected,
                df_model_detected,
                emd_results,
                write_path,
                label_model=label_model,
            )

        return {
            'emd_results': emd_results,
            'summary': summary,
            'ref_detected': df_ref_detected,
            'model_detected': df_model_detected,
        }


def run_joint_trip_evaluation(
    reference_schedules: pd.DataFrame,
    model_schedules: pd.DataFrame,
    write_path: Path,
    model_name: str = "Model",
    buffer_time_min: float = 5.0,
    buffer_duration_min: float = 5.0,
    trip_activity_name: str = "trip",
    generate_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    fonction utilitaire pour lancer l'évaluation des déplacements conjoints.

    c'est le point d'entrée principal utilisé par train_hh_disc pour évaluer
    les interactions entre membres du ménage dans les schedules générés.

    Args:
        reference_schedules: DataFrame des schedules de référence
        model_schedules: DataFrame des schedules générés
        write_path: répertoire de sauvegarde
        model_name: nom du modèle pour le labeling
        buffer_time_min: tolérance sur l'heure de début (en minutes)
        buffer_duration_min: tolérance sur la durée (en minutes)
        trip_activity_name: nom de l'activité déplacement dans les schedules
        generate_plots: si True, génère les plots

    Returns:
        dictionnaire avec les résultats d'évaluation
    """
    evaluator = JointTripEvaluator(
        buffer_time_min=buffer_time_min,
        buffer_duration_min=buffer_duration_min,
        trip_activity_name=trip_activity_name,
    )

    return evaluator.evaluate(
        df_ref=reference_schedules,
        df_model=model_schedules,
        write_path=write_path,
        label_model=model_name,
        generate_plots=generate_plots,
    )
