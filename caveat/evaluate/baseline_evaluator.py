"""
module d'évaluation thématique et visuelle pour les schedules générés.

reprend la logique de ThemeEvaluator et Visualizer de baseline.py pour
comparer les distributions d'activités entre référence et modèle.
utilisé par train_hh_disc pour évaluer la qualité des schedules synthétiques.
"""
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import wasserstein_distance
from pathlib import Path
from typing import Optional, Dict, List


class ThemeEvaluator:

    def __init__(self, config=None):
        self.config = config or {}

    def _safe_emd(self, d1, d2):
        if len(d1) == 0 or len(d2) == 0:
            return 999.0
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

        col_name = "Activity" if "Activity" in df_ref.columns else "following_purpose"

        cols_ref = df_ref[col_name].cast(pl.Utf8).unique().to_list()
        cols_mod = df_model[col_name].cast(pl.Utf8).unique().to_list()
        purposes = list(set(cols_ref) | set(cols_mod))

        scores = []
        for p in purposes:
            def get_counts(dframe):
                counts = dframe.filter(
                    pl.col(col_name).cast(pl.Utf8) == p
                ).group_by("person_id").len()
                all_ids = dframe.select("person_id").unique()
                final = all_ids.join(counts, on="person_id", how="left").fill_null(0)
                return final["len"].to_list()

            c_ref = get_counts(df_ref)
            c_mod = get_counts(df_model)
            scores.append(self._safe_emd(c_ref, c_mod))

        return {
            "Theme": "Participation",
            "Metric": "Avg Activity Count EMD",
            "Label": label,
            "Score": np.mean(scores)
        }

    def compute_ngram_transition(self, df_ref, df_model, label, n=2):

        col_name = "Activity" if "Activity" in df_ref.columns else "following_purpose"
        sort_col = "start_time" if "start_time" in df_ref.columns else "departure_time"

        def get_ngrams_dist(df, n_gram):
            df_sorted = df.sort(["person_id", sort_col])
            cols_to_concat = [pl.col(col_name).cast(pl.Utf8).alias("t0")]

            for i in range(1, n_gram):
                cols_to_concat.append(
                    pl.col(col_name).cast(pl.Utf8)
                    .shift(-i).over("person_id")
                    .alias(f"t{i}")
                )
    
            transitions = df_sorted.select(cols_to_concat).drop_nulls()
            ngram_col = pl.concat_str(transitions.columns, separator="->").alias("gram")

            return transitions.select(ngram_col)["gram"].value_counts(normalize=True)

        dist_ref = get_ngrams_dist(df_ref, n)
        dist_mod = get_ngrams_dist(df_model, n)

        comp = dist_ref.join(dist_mod, on="gram", how="full", suffix="_mod").fill_null(0)
        l1_diff = (comp["proportion"] - comp["proportion_mod"]).abs().sum()

        return {
            "Theme": "Transition",
            "Metric": f"{n}-gram Distribution L1",
            "Label": label,
            "Score": l1_diff
        }

    def compute_timing(self, df_ref, df_model, label):
        # On est dans le cas "Activity" / minutes
        if "start_time" in df_ref.columns:
            start_col, dur_col = "start_time", "activity_duration"
        else:
            start_col, dur_col = "departure_time", "activity_duration"

        # Tout est déjà en minutes
        start_ref = df_ref[start_col].cast(pl.Float64).drop_nulls().to_list()
        start_mod = df_model[start_col].cast(pl.Float64).drop_nulls().to_list()

        # Si jamais activity_duration n'est pas présent, on le calcule en minutes
        if dur_col not in df_ref.columns:
            df_ref = df_ref.with_columns(
                (pl.col("end_time") - pl.col("start_time")).alias(dur_col)
            )
        if dur_col not in df_model.columns:
            df_model = df_model.with_columns(
                (pl.col("end_time") - pl.col("start_time")).alias(dur_col)
            )

        dur_ref = df_ref[dur_col].cast(pl.Float64).drop_nulls().to_list()
        dur_mod = df_model[dur_col].cast(pl.Float64).drop_nulls().to_list()

        # EMD déjà en minutes, donc pas de /60
        emd_start = self._safe_emd(start_ref, start_mod)
        emd_dur   = self._safe_emd(dur_ref,   dur_mod)

        return [
            {"Theme": "Timing", "Metric": "Start Time EMD (min)", "Label": label, "Score": emd_start},
            {"Theme": "Timing", "Metric": "Duration EMD (min)",   "Label": label, "Score": emd_dur}
        ]


    def compute_diversity(self, df_ref, df_model, label):

        s_ref = self._get_sequences(df_ref)["sequence"]
        s_mod = self._get_sequences(df_model)["sequence"]

        seq_ref = set(tuple(x) for x in s_ref.to_list() if x is not None)
        seq_mod = set(tuple(x) for x in s_mod.to_list() if x is not None)

        intersection = seq_ref.intersection(seq_mod)
        coverage = len(intersection) / len(seq_ref) if len(seq_ref) > 0 else 0

        return {
            "Theme": "Diversity",
            "Metric": "Real Seq Coverage %",
            "Label": label,
            "Score": coverage * 100
        }

    def compute_validity(self, df_model, label):

        seq = self._get_sequences(df_model)

        score = seq.with_columns(
            (pl.col("sequence").list.last().cast(pl.Utf8).str.to_lowercase() == "home").alias("ends_home")
        )["ends_home"].mean()

        return {
            "Theme": "Validity",
            "Metric": "% Ends at Home",
            "Label": label,
            "Score": score * 100 if score is not None else 0.0
        }

    def evaluate_all(self, df_ref, df_model, label_model="Model"):

        print(f"--- Evaluating {label_model} ---")
        results = []
        results.append(self.compute_participation(df_ref, df_model, label_model))
        results.append(self.compute_ngram_transition(df_ref, df_model, label_model, n=2))
        results.append(self.compute_ngram_transition(df_ref, df_model, label_model, n=3))
        results.extend(self.compute_timing(df_ref, df_model, label_model))
        results.append(self.compute_diversity(df_ref, df_model, label_model))
        results.append(self.compute_validity(df_model, label_model))

        return pd.DataFrame(results)


class Visualizer:


    def __init__(self, config=None):
        self.config = config or {}
        sns.set_theme(style="ticks", context="talk", font_scale=0.9)

    def _prepare_single_df(self, df_pl):
        df = df_pl.select([
            pl.col("Activity"),
            (pl.col("start_time") / 60.0).alias("Start Time (h)"),   # minutes -> heures
            (pl.col("end_time")   / 60.0).alias("End Time (h)"),
            pl.col("activity_duration").alias("Duration (min)"),     # déjà en minutes
        ]).to_pandas()
        return df


    def get_consistent_activity_colormap(self, list_of_dfs):
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

    def plot_marginal_distributions(self, df_ref, df_model, write_path: Path,
                                   model_name: str = "Model"):

        print(f"--- Creating marginal distribution plots for {model_name} ---")

        color_map, sorted_activities = self.get_consistent_activity_colormap([df_ref, df_model])

        metrics_cfg = [
            ("Start Time (h)", (0, 24)),
            ("End Time (h)", (0, 30)),
            ("Duration (min)", (0, 600))
        ]

        n_rows = len(metrics_cfg)
        n_cols = len(sorted_activities)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey="row")

        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for col_idx, act in enumerate(sorted_activities):
            for row_idx, (metric_name, xlim) in enumerate(metrics_cfg):
                ax = axes[row_idx, col_idx]

                # courbe de la distribution de référence
                self._plot_single_distribution_curve(
                    ax, df_ref, metric_name, act, "Reference", "black", "-", 2.5
                )

                # courbe de la distribution du modèle
                self._plot_single_distribution_curve(
                    ax, df_model, metric_name, act, model_name, "#D95F02", "--", 2.5
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
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                  ncol=2, frameon=False, fontsize=14)

        plt.tight_layout()
        plt.savefig(write_path / f"marginal_distributions_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved marginal distributions to {write_path}")

    def _plot_single_distribution_curve(self, ax, df_pl, metric_col, activity,
                                       label, color, linestyle, linewidth=2.0):
        df = self._prepare_single_df(df_pl)
        subset = df[df["Activity"] == activity]
        data = subset[metric_col].dropna()
        data = data[np.isfinite(data)]

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
            except Exception:
                pass

    def plot_joint_distribution_2d(self, df_ref, df_model, write_path: Path,
                                   model_name: str = "Model",
                                   target_acts: List[str] = None):

        print(f"--- Creating joint distribution plots for {model_name} ---")

        if target_acts is None:
            target_acts = ["work", "shop", "education", "home"]

        n_models = 2  # référence et modèle
        n_acts = len(target_acts)

        fig, axes = plt.subplots(n_models, n_acts, figsize=(4 * n_acts, 3.5 * n_models),
                                sharex=True, sharey=True)

        if n_acts == 1:
            axes = axes.reshape(-1, 1)

        models_data = [
            ("Reference", df_ref, "Blues"),
            (model_name, df_model, "Oranges")
        ]

        for row_idx, (name, df, cmap) in enumerate(models_data):
            for col_idx, act in enumerate(target_acts):
                ax = axes[row_idx, col_idx]
                self._plot_joint_2d_single(ax, df, act, cmap)

                if col_idx == 0:
                    ax.text(-0.3, 0.5, name, transform=ax.transAxes,
                           va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

                if row_idx == 0:
                    ax.set_title(act.upper(), fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(write_path / f"joint_distributions_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved joint distributions to {write_path}")

    def _plot_joint_2d_single(self, ax, df_pl, activity, cmap="Blues", x_bins=48, y_bins=48):
        """trace une distribution jointe 2D (start time vs durée) pour une seule activité."""
        df = self._prepare_single_df(df_pl)
        subset = df[df["Activity"] == activity]

        x = subset["Start Time (h)"].values
        y = subset["Duration (min)"].values / 60.0

        ax.hist2d(x, y, bins=[x_bins, y_bins], range=[[0, 24], [0, 16]],
                 cmap=cmap, density=True, cmin=0.0001)

        ax.set_title(activity.title(), fontsize=12, fontweight='bold')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Duration (h)")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Start Time (h)")

        ax.grid(False)

    def plot_stacked_tempograms(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        write_path,
        model_name: str = "Model",
        resolution_min: int = 10,
        title_ref: str = "Reference",
        title_model: str = None,
    ):
        if title_model is None:
            title_model = model_name

        color_map, sorted_activities = self.get_consistent_activity_colormap([df_ref, df_model])

        fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharex=True, sharey=True)

        self.plot_single_tempogram(
            ax=axes[0],
            df_pl=df_ref,
            title=title_ref,
            color_map=color_map,
            sorted_activities=sorted_activities,
            resolution_min=resolution_min,
        )

        self.plot_single_tempogram(
            ax=axes[1],
            df_pl=df_model,
            title=title_model,
            color_map=color_map,
            sorted_activities=sorted_activities,
            resolution_min=resolution_min,
        )

        # légende unique (on récupère depuis le 2e axe)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False)

        for ax in axes:
            ax.set_xlabel("Time of day (h)")
        axes[0].set_ylabel("Share of population")

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        out = write_path / f"stacked_tempograms_{model_name}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved stacked tempograms to {out}")



    def plot_single_tempogram(
        self,
        ax,
        df_pl: pl.DataFrame,
        title: str,
        color_map: dict,
        sorted_activities: list,
        resolution_min: int = 10,
    ):

        # --- grille temporelle ---
        n_bins = int((24 * 60) / resolution_min) + 1  # inclut 24:00
        time_axis = np.linspace(0, 24, n_bins)

        # --- taille de la population (pour normalisation) ---
        if "person_id" in df_pl.columns:
            n_people = df_pl.select(pl.col("person_id").n_unique()).item()
        else:

            n_people = None

        # --- ordre d'empilement des activités sur le stackplot ---
        acts_to_plot = list(sorted_activities)
        curr_acts = df_pl.select(pl.col("Activity").cast(pl.Utf8)).unique().to_series().to_list()
        has_home = any((a is not None) and (str(a).lower() == "home") for a in curr_acts)
        if has_home:
            acts_to_plot.append("home")  # home en dernier pour lisibilité

        curves = {}
        total_pop = np.zeros(n_bins, dtype=float)

        # --- calcul de la courbe de participation pour chaque activité ---
        for act in acts_to_plot:
            try:
                subset = (
                    df_pl
                    .filter(pl.col("Activity").cast(pl.Utf8) == str(act))
                    .select([
                        (pl.col("start_time") / resolution_min).floor().cast(pl.Int32).alias("start_idx"),
                        (pl.col("end_time")   / resolution_min).floor().cast(pl.Int32).alias("end_idx"),
                    ])
                )

                starts = np.clip(subset["start_idx"].to_numpy(), 0, n_bins - 1)
                ends   = np.clip(subset["end_idx"].to_numpy(),   0, n_bins - 1)

                deltas = np.zeros(n_bins, dtype=float)
                np.add.at(deltas, starts, 1.0)
                np.add.at(deltas, ends,  -1.0)

                counts = np.cumsum(deltas)
                counts = np.clip(counts, 0, None)  # sécurité

                curves[act] = counts
                total_pop += counts

            except Exception:
                curves[act] = np.zeros(n_bins, dtype=float)

        # --- normalisation par la taille de la population pour avoir des proportions ---
        if n_people is None or n_people <= 0:
            denom = np.max(total_pop) if np.max(total_pop) > 0 else 1.0
        else:
            denom = float(n_people)

        stack_values_pct, labels_stack, stack_colors = [], [], []

        for act in acts_to_plot:
            vals = curves.get(act, np.zeros(n_bins, dtype=float)) / denom
            stack_values_pct.append(vals)
            labels_stack.append(str(act).title())
            stack_colors.append(color_map.get(act, "#333333"))

        ax.stackplot(time_axis, *stack_values_pct, labels=labels_stack, colors=stack_colors, alpha=0.9)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.axvline(12, color="black", linestyle=":", alpha=0.3)
        ax.grid(False)
        ax.set_ylabel("Share of population")

def prepare_eval_df_from_minutes(df_minutes: pl.DataFrame) -> pl.DataFrame:
    attrs_cols = [c for c in ["age","sex","employed","studies","socioprofessional_class","commute_mode","number_of_vehicles","income_class","urban_type","departement_id","household_size"] if c in df_minutes.columns]

    return (
        df_minutes
        .select([
            pl.col("pid").alias("person_id"),
            pl.col("act").alias("Activity"),
            pl.col("start").cast(pl.Float64).alias("start_time"),   # minutes
            pl.col("end").cast(pl.Float64).alias("end_time"),       # minutes
            *attrs_cols
        ])
        .with_columns(
            (pl.col("end_time") - pl.col("start_time"))
            .alias("activity_duration")  # minutes
        )
        .sort(["person_id", "start_time"])
    )


# ==============================================================================
# FONCTION UTILITAIRE POUR LES GROUPES D'ÂGE
# ==============================================================================

def create_age_groups_from_bins(age_col: pl.Expr, bins: list) -> pl.Expr:
    """
    crée des labels de groupe d'âge à partir des bornes de la config.
    par exemple bins=[6, 11, 15, 18, 30, 40, 50, 60, 70]
    produit : 0-5, 6-10, 11-14, 15-17, 18-29... 70+
    c'est nécessaire pour le conditionnement par tranche d'âge dans l'évaluation.
    """
    # 1. en dessous de la première borne (0-5)
    expr = pl.when(age_col < bins[0]).then(pl.lit(f"0-{bins[0]-1}"))

    # 2. tranches intermédiaires
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i+1]
        expr = expr.when(
            (age_col >= lower) & (age_col < upper)
        ).then(pl.lit(f"{lower}-{upper-1}"))

    # 3. dernière tranche (70+)
    expr = expr.otherwise(pl.lit(f"{bins[-1]}+"))

    return expr


# ==============================================================================
# CLASSES DE MÉTRIQUES CONDITIONNELLES
# ==============================================================================

class ConditionalMetrics:
    """
    calcule les métriques d'évaluation conditionnées sur les attributs socio-démo.

    permet de calculer participation, timing, validité pour chaque sous-groupe
    démographique (par âge, sexe, statut d'emploi, etc.).
    c'est ce qui nous permet de vérifier que le modèle génère correctement
    pour chaque catégorie de population et pas juste en moyenne.
    """

    def __init__(self, config: dict = None, evaluator=None):
        self.config = config or {}
        self.evaluator = evaluator if evaluator else ThemeEvaluator(self.config)

    def add_attributes_to_schedules(
        self,
        schedules: pl.DataFrame,
        attributes: pd.DataFrame,
        join_col: str = "person_id"
    ) -> pl.DataFrame:
        """fusionne les attributs (age, sex, etc.) avec les schedules pour pouvoir filtrer par groupe."""
        if isinstance(attributes, pd.DataFrame):
            attrs_pl = pl.from_pandas(attributes)
        else:
            attrs_pl = attributes

        if "pid" in attrs_pl.columns and join_col == "person_id":
            attrs_pl = attrs_pl.rename({"pid": "person_id"})

        if join_col in schedules.columns:
            schedules = schedules.with_columns(pl.col(join_col).cast(pl.Int64))
            attrs_pl = attrs_pl.with_columns(pl.col(join_col).cast(pl.Int64))

        cols_to_keep = [join_col] + [
            c for c in attrs_pl.columns
            if c not in schedules.columns and c != join_col
        ]
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
        calcule les métriques pour chaque attribut dans condition_on.

        on filtre les schedules par catégorie (ex: age_group, sex) et on calcule
        les métriques indépendamment pour chaque sous-groupe. ça permet de détecter
        si le modèle est mauvais sur certaines catégories (ex: les jeunes, les femmes, etc.)

        Args:
            df_ref: schedules de référence (avec person_id, Activity, start_time, etc.)
            df_model: schedules du modèle (même format)
            ref_attributes: DataFrame d'attributs de la référence (person_id, age, sex, etc.)
            model_attributes: DataFrame d'attributs du modèle (même format)
            condition_on: liste des attributs sur lesquels conditionner (ex: ["age", "sex"])
            label_model: label du modèle dans les résultats
            age_bins: bornes pour les groupes d'âge (ex: [6, 11, 15, 18, 30, 40, 50, 60, 70])

        Returns:
            DataFrame avec les métriques conditionnelles
        """
        # df_ref_full = self.add_attributes_to_schedules(df_ref, ref_attributes)
        # df_model_full = self.add_attributes_to_schedules(df_model, model_attributes)
        df_ref_full = df_ref
        df_model_full = df_model
        print(df_ref_full)
        print(df_model_full)
        # gestion du groupement par âge (on transforme l'âge continu en tranches)
        if "age" in condition_on or "age_group" in condition_on:
            if "age_group" not in df_ref_full.columns:
                if "age" in df_ref_full.columns and age_bins is not None:
                    df_ref_full = df_ref_full.with_columns(
                        create_age_groups_from_bins(pl.col("age").cast(pl.Int64), age_bins).alias("age_group")
                    )

            if "age_group" not in df_model_full.columns:
                if "age" in df_model_full.columns and age_bins is not None:
                    df_model_full = df_model_full.filter(pl.col("age") != "PAD").with_columns(
                        create_age_groups_from_bins(pl.col("age").cast(pl.Int64), age_bins).alias("age_group")
                    )

            if "age" in condition_on:
                condition_on = [c for c in condition_on if c != "age"]
                if "age_group" not in condition_on:
                    condition_on.append("age_group")

        all_results = []

        # on boucle sur chaque attribut et on calcule les métriques par catégorie
        for attr in condition_on:
            if attr not in df_ref_full.columns or attr not in df_model_full.columns:
                continue

            # récupère les catégories uniques dans la ref et le modèle
            cats_ref = df_ref_full[attr].unique().to_list()
            cats_mod = df_model_full[attr].unique().to_list()
            all_cats = sorted([c for c in list(set(cats_ref) | set(cats_mod)) if c is not None])

            # pour chaque valeur de catégorie on filtre et on évalue
            for cat in all_cats:
                sub_ref = df_ref_full.filter(pl.col(attr) == cat)
                sub_mod = df_model_full.filter(pl.col(attr) == cat)

                n_ref = sub_ref.select("person_id").n_unique()
                n_model = sub_mod.select("person_id").n_unique()

                # on saute les sous-groupes trop petits, pas assez de données pour être fiable
                if n_ref < 10 or n_model < 10:
                    continue

                # calcul des métriques (participation, timing, validité) pour ce sous-groupe
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
    """visualisation des métriques conditionnelles par attribut socio-démo."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)

    def plot_heatmap(
        self,
        df_results: pd.DataFrame,
        attribute: str,
        metric_name: str,
        model_name: str = "Model",
        write_path: Path = None
    ):
        """
        heatmap des scores par catégorie pour un attribut et une métrique donnés.

        permet de visualiser d'un coup d'oeil quelles catégories sont bien/mal reproduites.

        Args:
            df_results: DataFrame de résultats issu de compute_conditional_metrics
            attribute: attribut à afficher (ex: "age_group", "sex")
            metric_name: métrique à filtrer (ex: "EMD", "Count")
            model_name: nom du modèle pour le titre
            write_path: si fourni, sauvegarde le plot à ce chemin
        """
        subset = df_results[
            (df_results["Attribute"] == attribute) &
            (df_results["Metric"].str.contains(metric_name, case=False))
        ]

        if subset.empty:
            print(f"No data to plot for {attribute} / {metric_name}")
            return

        # pivot : lignes=métrique, colonnes=catégorie
        pivot = subset.pivot_table(index="Metric", columns="Category", values="Score")

        # tri des colonnes pour un affichage cohérent
        cols = sorted(pivot.columns)
        pivot = pivot[cols]

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
            cbar_kws={'label': 'Error (EMD/L1)'}
        )
        plt.title(f"Performance by {attribute} - {metric_name} ({model_name})")
        plt.tight_layout()

        if write_path:
            plt.savefig(write_path / f"conditional_heatmap_{attribute}_{metric_name}.png",
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_distributions_by_condition(
        self,
        df_ref: pl.DataFrame,
        df_model: pl.DataFrame,
        ref_attrs: pd.DataFrame,
        mod_attrs: pd.DataFrame,
        condition_attr: str,
        age_bins: list,
        activity: str = "work",
        write_path: Path = None
    ):
        """
        trace les distributions de start time par catégorie d'un attribut conditionnel.

        pour chaque valeur de l'attribut (ex: chaque tranche d'âge), on trace la
        distribution de référence vs modèle. ça permet de voir visuellement si le
        modèle reproduit bien les horaires pour chaque sous-groupe.

        Args:
            df_ref: schedules de référence
            df_model: schedules du modèle
            ref_attrs: attributs de la référence
            mod_attrs: attributs du modèle
            condition_attr: attribut sur lequel conditionner (ex: "age_group", "sex")
            age_bins: bornes pour les groupes d'âge
            activity: type d'activité à tracer (ex: "work")
            write_path: si fourni, sauvegarde le plot à ce chemin
        """
        cm = ConditionalMetrics()
        df_r = df_ref
        df_m = df_model.filter(pl.col("age") != "PAD")

        if condition_attr == "age_group":
            df_r = df_r.with_columns(
                create_age_groups_from_bins(pl.col("age").cast(pl.Int64), age_bins).alias("age_group")
            )
            df_m = df_m.with_columns(
                create_age_groups_from_bins(pl.col("age").cast(pl.Int64), age_bins).alias("age_group")
            )

        cats = sorted([c for c in df_r[condition_attr].unique().to_list() if c is not None])

        # création de la grille de subplots (un par catégorie)
        n_cols = 4
        n_rows = (len(cats) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
            sharex=True, sharey=True
        )
        axes = axes.flatten()

        col_act = "Activity" if "Activity" in df_r.columns else "following_purpose"

        # pour chaque catégorie on trace la KDE ref vs modèle
        for i, cat in enumerate(cats):
            ax = axes[i]

            # conversion des start times de minutes en heures pour l'affichage
            sr = df_r.filter(
                (pl.col(condition_attr) == cat) & (pl.col(col_act) == activity)
            )["start_time"].to_numpy() / 60.0  # minutes to hours

            sm = df_m.filter(
                (pl.col(condition_attr) == cat) & (pl.col(col_act) == activity)
            )["start_time"].to_numpy() / 60.0  # minutes to hours

            if len(sr) > 10:
                sns.kdeplot(sr, ax=ax, fill=True, color="black", alpha=0.1, label="Ref")
            if len(sm) > 10:
                sns.kdeplot(sm, ax=ax, color="tab:orange", linestyle="--", label="Model")

            ax.set_title(f"{cat} (N={len(sr)})")
            ax.set_xlim(0, 24)
            if i == 0:
                ax.legend()

        plt.suptitle(f"Start Time: {activity.upper()} by {condition_attr}")
        plt.tight_layout()

        if write_path:
            plt.savefig(
                write_path / f"conditional_dist_{condition_attr}_{activity}.png",
                dpi=150, bbox_inches='tight'
            )
            plt.close()
        else:
            plt.show()
