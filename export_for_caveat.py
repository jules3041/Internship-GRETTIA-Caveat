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
    df = df.sort(["person_id", "departure_time"])

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
        pl.when(pl.col("following_purpose").cast(pl.Utf8).str.to_lowercase() == "escort").then(pl.lit("other")).otherwise(pl.col("following_purpose").cast(pl.Utf8).str.to_lowercase())
        .alias("act"),
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
        # Garde-fou global: toute occurrence résiduelle de "escort" est mappée vers "other"
        .with_columns(
            pl.when(pl.col("act").cast(pl.Utf8).str.to_lowercase() == "escort")
            .then(pl.lit("other"))
            .otherwise(pl.col("act").cast(pl.Utf8).str.to_lowercase())
            .alias("act")
        )
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


def filter_people_for_caveat(
    df_attributes: pl.DataFrame,
    min_age: int = 5,
    max_people_per_household: int = 5,
) -> pl.DataFrame:
    """
    1) Remove persons younger than min_age and fix household_size accordingly.
    2) Keep only the max_people_per_household oldest persons per household.
       household_size is intentionally not re-fixed after this second filter.
    """
    initial_n = df_attributes.height

    df_no_young = df_attributes.filter(
        pl.col("age").is_null() | (pl.col("age") >= min_age)
    )
    print(
        f"Filtre age >= {min_age}: {initial_n} -> {df_no_young.height} personnes."
    )

    df_no_young_fixed = fix_household_size(df_no_young)

    df_top_oldest = (
        df_no_young_fixed
        .with_columns(pl.col("age").fill_null(-1).alias("_age_for_rank"))
        .sort(["hid", "_age_for_rank", "pid"], descending=[False, True, False])
        .group_by("hid", maintain_order=True)
        .head(max_people_per_household)
        .drop("_age_for_rank")
    )
    print(
        "Filtre top personnes par menage: "
        f"{df_no_young_fixed.height} -> {df_top_oldest.height} "
        f"(max={max_people_per_household}, household_size conserve)."
    )

    return df_top_oldest


# df_attributes_fixed = fix_household_size(df_attributes_ready)
# df_attributes_fixed.write_csv(path_csv_attr)
import polars as pl

def generate_interaction_table_corrected(
    df_schedules: pl.DataFrame,
    df_trips_rich: pl.DataFrame,
    cfg
) -> pl.DataFrame:
    """
    Génère la table d'interactions pour CAVEAT.

    Sortie (format étendu):
    - schedule_id
    - pid
    - interaction_bin (1 pour interaction positive)
    - overlap_start (minutes)
    - overlap_end (minutes)

    La logique Joint Trip reste inchangée.
    La logique Joint Activity est définie par recouvrement temporel strict
    (sur les activités de même act et même destination).
    """
    print("--- Génération de la table des interactions (Joint Trip + Joint Activity) ---")

    # A. Préparation (trips riches)
    if "trip_id" not in df_trips_rich.columns:
        df_rich = df_trips_rich.with_row_index("trip_id")
    else:
        df_rich = df_trips_rich

    # 1) Unifier car + car_passenger (mode_group)
    df_rich = df_rich.with_columns(
        pl.when(pl.col("mode").is_in(["car", "car_passenger"]))
          .then(pl.lit("car"))
          .otherwise(pl.col("mode"))
          .alias("mode_group")
    )

    # 2) Couples potentiellement partagés: même ménage + même mode_group
    trips_pairs = df_rich.join(
        df_rich,
        on=["egt_household_id", "mode_group"],
        suffix="_other"
    )

    potential_shared = trips_pairs.filter(
        (pl.col("person_id") != pl.col("person_id_other")) &
        ((pl.col("departure_time") - pl.col("departure_time_other")).abs() <= cfg.BUFFER_TIME_SEC)
    )

    # --- Joint Trip (2) ---
    jt_pairs = potential_shared.filter(
        (pl.col("following_purpose") == pl.col("following_purpose_other")) &
        ((pl.col("trip_duration") - pl.col("trip_duration_other")).abs() <= cfg.BUFFER_DURATION_SEC) &
        (pl.col("origin_commune_large") == pl.col("origin_commune_large_other")) &
        (pl.col("destination_commune_large") == pl.col("destination_commune_large_other"))
    )

    # Union des deux côtés (évite l'effet "miroir")
    joint_trips_ids = (
        pl.concat([
            jt_pairs.select(pl.col("trip_id").cast(pl.Int64).alias("trip_id")),
            jt_pairs.select(pl.col("trip_id_other").cast(pl.Int64).alias("trip_id")),
        ])
        .unique()
        .get_column("trip_id")
    )

    # --- (ESCORT) désactivé ---
    # escort_pairs = potential_shared.filter(
    #     (pl.col("following_purpose") != pl.col("following_purpose_other")) &
    #     ((pl.col("following_purpose") == "escort") | (pl.col("following_purpose_other") == "escort")) &
    #     (pl.col("origin_commune_large") == pl.col("origin_commune_large_other")) &
    #     (pl.col("destination_commune_large") == pl.col("destination_commune_large_other")) &
    #     ((pl.col("arrival_time") - pl.col("arrival_time_other")).abs() <= cfg.BUFFER_TIME_SEC) &
    #     ((pl.col("trip_duration") - pl.col("trip_duration_other")).abs() <= cfg.BUFFER_DURATION_SEC)
    # )
    # escort_trips_ids = (
    #     pl.concat([
    #         escort_pairs.select(pl.col("trip_id").cast(pl.Int64).alias("trip_id")),
    #         escort_pairs.select(pl.col("trip_id_other").cast(pl.Int64).alias("trip_id")),
    #     ])
    #     .unique()
    #     .get_column("trip_id")
    # )

    # --- Joint Activity (1) ---
    # On détecte les recouvrements stricts sur les lignes d'activité des schedules
    # avec même ménage, même act et même destination.
    #
    # Important: l'activité initiale a trip_id = -1.
    # Pour ne pas la perdre, on lui attribue la commune d'origine du 1er trip
    # de la personne (quand disponible).
    trip_context = df_rich.select([
        pl.col("trip_id").cast(pl.Int64),
        pl.col("egt_household_id").cast(pl.Int64),
        pl.col("destination_commune_large").cast(pl.Utf8),
    ]).unique(subset=["trip_id"], keep="first")

    initial_activity_context = (
        df_rich
        .sort(["person_id", "departure_time"])
        .group_by("person_id", maintain_order=True)
        .agg([
            pl.col("egt_household_id").first().cast(pl.Int64).alias("initial_household_id"),
            pl.col("origin_commune_large").first().cast(pl.Utf8).alias("initial_commune_large"),
        ])
        .rename({"person_id": "pid"})
    )

    activity_rows = (
        df_schedules
        .filter(pl.col("act") != "trip")
        .select([
            pl.col("schedule_id").cast(pl.Int64),
            pl.col("pid"),
            pl.col("trip_id").cast(pl.Int64),
            pl.col("act").cast(pl.Utf8),
            pl.col("start").cast(pl.Float64),
            pl.col("end").cast(pl.Float64),
        ])
        .join(trip_context, on="trip_id", how="left")
        .join(initial_activity_context, on="pid", how="left")
        .with_columns([
            pl.when(pl.col("trip_id") >= 0)
            .then(pl.col("egt_household_id"))
            .otherwise(pl.col("initial_household_id"))
            .alias("activity_household_id"),
            pl.when(pl.col("trip_id") >= 0)
            .then(pl.col("destination_commune_large"))
            .otherwise(pl.col("initial_commune_large"))
            .alias("activity_commune_large"),
        ])
        .filter(
            pl.col("activity_household_id").is_not_null()
            & pl.col("activity_commune_large").is_not_null()
        )
    )

    activity_pairs = activity_rows.join(
        activity_rows,
        on=["activity_household_id", "act", "activity_commune_large"],
        suffix="_other",
    )

    activity_pairs = (
        activity_pairs
        .filter(
            (pl.col("pid") != pl.col("pid_other"))
            & (pl.col("schedule_id") < pl.col("schedule_id_other"))
        )
        .with_columns([
            pl.max_horizontal(pl.col("start"), pl.col("start_other")).alias("overlap_start"),
            pl.min_horizontal(pl.col("end"), pl.col("end_other")).alias("overlap_end"),
        ])
        .filter(pl.col("overlap_start") < pl.col("overlap_end"))
    )

    joint_activity_intervals = (
        pl.concat([
            activity_pairs.select([
                pl.col("schedule_id").cast(pl.Int64).alias("schedule_id"),
                pl.col("pid"),
                pl.col("overlap_start").cast(pl.Float64),
                pl.col("overlap_end").cast(pl.Float64),
            ]),
            activity_pairs.select([
                pl.col("schedule_id_other").cast(pl.Int64).alias("schedule_id"),
                pl.col("pid_other").alias("pid"),
                pl.col("overlap_start").cast(pl.Float64),
                pl.col("overlap_end").cast(pl.Float64),
            ]),
        ], how="vertical_relaxed")
        .unique(subset=["schedule_id", "pid", "overlap_start", "overlap_end"])
        .with_columns(pl.lit(1).cast(pl.Int8).alias("interaction_bin"))
        .select(["schedule_id", "pid", "interaction_bin", "overlap_start", "overlap_end"])
    )

    # B. Joint Trip inchangé en détection, mais exporté avec intervalle [start, end]
    joint_trip_intervals = (
        df_schedules
        .filter((pl.col("act") == "trip") & pl.col("trip_id").is_in(joint_trips_ids))
        .select([
            pl.col("schedule_id").cast(pl.Int64).alias("schedule_id"),
            pl.col("pid"),
            pl.lit(1).cast(pl.Int8).alias("interaction_bin"),
            pl.col("start").cast(pl.Float64).alias("overlap_start"),
            pl.col("end").cast(pl.Float64).alias("overlap_end"),
        ])
        .filter(pl.col("overlap_start") < pl.col("overlap_end"))
    )

    if joint_activity_intervals.height == 0 and joint_trip_intervals.height == 0:
        return pl.DataFrame(
            schema={
                "schedule_id": pl.Int64,
                "pid": df_schedules.schema.get("pid", pl.Utf8),
                "interaction_bin": pl.Int8,
                "overlap_start": pl.Float64,
                "overlap_end": pl.Float64,
            }
        )

    return (
        pl.concat([joint_activity_intervals, joint_trip_intervals], how="vertical_relaxed")
        .unique(subset=["schedule_id", "pid", "overlap_start", "overlap_end"])
        .select(["schedule_id", "pid", "interaction_bin", "overlap_start", "overlap_end"])
    )
# 1. Préparation trips indexés
if "trip_id" not in df_egt_t.columns:
    df_trips_indexed = df_egt_t.with_row_index("trip_id")
else:
    df_trips_indexed = df_egt_t

# 2. Table ATTRIBUTES
path_csv_attr = os.path.join(cfg.CACHE_DIR, "caveat_attributes.csv")
df_attributes_ready = export_attributes_to_caveat_csv(df_egt_i, df_egt_h, path_csv_attr)
df_attributes_filtered = filter_people_for_caveat(
    df_attributes_ready,
    min_age=5,
    max_people_per_household=5,
)
df_attributes_filtered.write_csv(path_csv_attr)

# Keep trips only for retained persons (same population across attributes/schedules/interactions)
retained_pids = (
    df_attributes_filtered
    .select(pl.col("pid").alias("person_id"))
    .unique()
)
df_trips_filtered = df_trips_indexed.join(retained_pids, on="person_id", how="inner")
print(
    f"Trips filtres sur personnes retenues: {df_trips_indexed.height} -> "
    f"{df_trips_filtered.height}."
)

# 3. Table SCHEDULES (génère les schedule_id)
path_csv_schedules = os.path.join(cfg.CACHE_DIR, "caveat_schedules.csv")
df_schedules_final = export_schedules_for_caveat(
    df_trips=df_trips_filtered, 
    df_attributes=df_attributes_filtered, 
    output_path=path_csv_schedules
)

# 4. Table INTERACTIONS (liée par schedule_id)
path_csv_interactions = os.path.join(cfg.CACHE_DIR, "caveat_interactions.csv")
df_interactions = generate_interaction_table_corrected(
    df_schedules=df_schedules_final,
    df_trips_rich=df_trips_filtered, # On a besoin des infos riches (communes, modes...)
    cfg=cfg
)

# Validation
print("\nDistribution des interactions :")
print(df_interactions["interaction_bin"].value_counts().sort("interaction_bin"))

# On vérifie la cohérence Act vs Trip
check = (
    df_schedules_final
    .join(
        df_interactions.group_by("schedule_id").agg(
            pl.col("interaction_bin").max().alias("interaction_bin")
        ),
        on="schedule_id",
        how="left",
    )
    .with_columns(pl.col("interaction_bin").fill_null(0))
)
print("\nVérification cohérence (Type Act vs Interaction):")
print(check.group_by(["act", "interaction_bin"]).len().sort(["act", "interaction_bin"]))

df_interactions.write_csv(path_csv_interactions)
print(f"Export interactions terminé : {path_csv_interactions}")
