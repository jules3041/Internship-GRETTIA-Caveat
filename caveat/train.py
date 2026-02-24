import yaml
import os
from caveat import runners
from caveat.models import library
from caveat.models.discrete.household_vae import HouseholdDiscreteVAE
from caveat.evaluate.baseline_evaluator import (
    ThemeEvaluator,
    Visualizer,
    prepare_eval_df_from_minutes,
    ConditionalMetrics,
    ConditionalVisualizer
)
from caveat.evaluate.joint_trip_evaluator import JointTripEvaluator, run_joint_trip_evaluation
import polars as pl
import pandas as pd

from pathlib import Path

def main():

    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs" / "household_VAE.yaml"

    print(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    library["household_vae"] = HouseholdDiscreteVAE

    print(f"   Modèle : {config['model_params']['name']}")
    print(f"   Epochs : {config.get('trainer_params', {}).get('max_epochs', 'défaut')}")

    # runners.run_command(
    #     config=config,
    #     verbose=True,
    #     gen=True,
    #     test=True,
    #     infer=True
    # )

    eval_params = config.get("baseline_evaluation", {})
    if eval_params.get("enabled", True):
        print("\n" + "="*70)
        print("   RUNNING BASELINE EVALUATION")
        print("="*70)

        run_enhanced_evaluation(config, eval_params)

def run_enhanced_evaluation(config, eval_params):

    # logger path
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(logger_params.get("name", "household_vae"))

    # find the most recent version directory
    experiment_dir = log_dir / name
    if not experiment_dir.exists():
        print(f"Warning: Experiment directory {experiment_dir} not found. Skipping enhanced evaluation.")
        return

    version_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]

    if not version_dirs:
        print("Warning: No version directories found. Skipping enhanced evaluation.")
        return

    versions = sorted(version_dirs, key=lambda d: int(d.name.split('_')[-1]))

    latest_version = versions[-1]
    print(f"Using results from: {latest_version}")

    synthetic_path = latest_version / "synthetic_schedules_with_attributes.csv"
    if not synthetic_path.exists():
        print(f"Warning: Synthetic schedules not found at {synthetic_path}")
        return

    print(f"Loading synthetic schedules from {synthetic_path}")
    df_synthetic_schedules = pd.read_csv(synthetic_path)

    schedules_path = Path(config["schedules_path"])
    print(f"Loading observed schedules from {schedules_path}")
    df_observed_schedules = pd.read_csv(schedules_path)

    attributes_path = Path(config["attributes_path"])
    df_observed_attributes = pd.read_csv(attributes_path)
    df_observed_schedules = df_observed_schedules.merge(df_observed_attributes, on = "pid", how = "left")
    df_syn_pl = pl.from_pandas(df_synthetic_schedules)
    df_obs_pl = pl.from_pandas(df_observed_schedules)

    df_syn_continuous = prepare_eval_df_from_minutes(df_syn_pl)
    df_obs_continuous = prepare_eval_df_from_minutes(df_obs_pl)

    evaluator = ThemeEvaluator(config=eval_params)
    visualizer = Visualizer(config=eval_params)

    print("\n" + "-"*70)
    print("   THEME EVALUATOR METRICS")
    print("-"*70)

    model_name = config["model_params"]["name"]
    df_scores = evaluator.evaluate_all(
        df_ref=df_obs_continuous,
        df_model=df_syn_continuous,
        label_model=model_name
    )

    # Display results
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

    if eval_params.get("generate_plots", True):
        print("\n" + "-"*70)
        print("   GENERATING DISTRIBUTION PLOTS")
        print("-"*70)

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
    if joint_trip_params.get("enabled", True):
        print("\n" + "-"*70)
        print("   JOINT TRIP/ACTIVITY EVALUATION")
        print("-"*70)

        # vérification que l'activité 'trip' existe dans les données
        trip_activity_name = joint_trip_params.get("trip_activity_name", "trip")
        has_trips_syn = df_synthetic_schedules['act'].str.lower().str.contains(trip_activity_name.lower()).any()
        has_trips_obs = df_observed_schedules['act'].str.lower().str.contains(trip_activity_name.lower()).any()

        if has_trips_syn and has_trips_obs:
            print(f"Trip activity '{trip_activity_name}' found in both datasets.")

            joint_eval_dir = eval_output_dir / "joint_trip_evaluation"

            joint_trip_results = run_joint_trip_evaluation(
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

            # on lance seulement l'analyse des activités conjointes (sans les déplacements)
            joint_eval_dir = eval_output_dir / "joint_activity_evaluation"

            evaluator_joint = JointTripEvaluator(
                buffer_time_min=joint_trip_params.get("buffer_time_min", 5.0),
                buffer_duration_min=joint_trip_params.get("buffer_duration_min", 5.0),
                trip_activity_name=trip_activity_name,
            )

            joint_results = evaluator_joint.evaluate(
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
        print("\n" + "-"*70)
        print("   CONDITIONAL EVALUATION (BY DEMOGRAPHICS)")
        print("-"*70)

        # chargement des attributs si disponibles
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

            # récupération des paramètres d'évaluation
            condition_on = conditional_params.get("condition_on", ["age", "sex"])
            age_bins = conditional_params.get("age_bins", [6, 11, 15, 18, 30, 40, 50, 60, 70])

            # initialisation des évaluateurs conditionnels
            cond_eval = ConditionalMetrics(config=eval_params, evaluator=evaluator)
            cond_vis = ConditionalVisualizer(config=eval_params)

            print(f"Computing conditional metrics for: {condition_on}")

            # calcul des métriques conditionnelles
            df_cond = cond_eval.compute_conditional_metrics(
                df_ref=df_obs_continuous,
                df_model=df_syn_continuous,
                ref_attributes=df_obs_attrs,
                model_attributes=df_syn_attrs,
                condition_on=condition_on,
                age_bins=age_bins,
                label_model=model_name
            )

            # sauvegarde des résultats
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

    print("\n" + "="*70)
    print("  EVALUATION COMPLETED")
    print("="*70 + "\n")
if __name__ == "__main__":
    main()