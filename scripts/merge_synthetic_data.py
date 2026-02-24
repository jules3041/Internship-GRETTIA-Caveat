#!/usr/bin/env python3
"""
Script utilitaire pour merger synthetic_schedules.csv et synthetic_attributes.csv

Usage:
    python scripts/merge_synthetic_data.py --schedules synthetic_schedules.csv --attributes synthetic_attributes.csv --output merged_data.csv

Description:
    - Charge les deux fichiers CSV
    - Les joint sur (hid, pid, member_idx)
    - Sauvegarde le résultat combiné avec schedules + attributs pour chaque activité
"""

import argparse
import pandas as pd
from pathlib import Path


def merge_synthetic_data(schedules_path, attributes_path, output_path=None):
    """
    Merge synthetic schedules and attributes into a single DataFrame.

    Args:
        schedules_path: Path to synthetic_schedules.csv
        attributes_path: Path to synthetic_attributes.csv
        output_path: Optional path to save merged CSV

    Returns:
        DataFrame with merged data
    """
    print(f"📂 Chargement de {schedules_path}...")
    schedules = pd.read_csv(schedules_path)
    print(f"   → {len(schedules)} lignes (activités)")

    print(f"📂 Chargement de {attributes_path}...")
    attributes = pd.read_csv(attributes_path)
    print(f"   → {len(attributes)} lignes (membres)")

    # Vérifier les colonnes de jointure
    join_cols = []
    if 'hid' in schedules.columns and 'hid' in attributes.columns:
        join_cols.append('hid')
    if 'pid' in schedules.columns and 'pid' in attributes.columns:
        join_cols.append('pid')
    if 'member_idx' in schedules.columns and 'member_idx' in attributes.columns:
        join_cols.append('member_idx')

    if not join_cols:
        print("❌ Erreur : Aucune colonne de jointure commune trouvée (hid, pid, ou member_idx)")
        print(f"   Colonnes schedules : {list(schedules.columns)}")
        print(f"   Colonnes attributes : {list(attributes.columns)}")
        return None

    print(f"🔗 Jointure sur : {join_cols}")

    # Merge
    merged = pd.merge(schedules, attributes, on=join_cols, how='left')
    print(f"✓ Merge réussi : {len(merged)} lignes")

    # Vérifier s'il y a des valeurs manquantes
    missing = merged.isna().sum()
    if missing.any():
        print("\n⚠ Valeurs manquantes après merge :")
        print(missing[missing > 0])

    # Sauvegarder
    if output_path:
        merged.to_csv(output_path, index=False)
        print(f"💾 Sauvegarde dans {output_path}")

    # Afficher aperçu
    print("\n📊 Aperçu des données mergées :")
    print(merged.head(10))

    # Stats par ménage
    if 'hid' in merged.columns:
        hh_stats = merged.groupby('hid').agg({
            'pid': 'nunique',
            'act': 'count'
        }).rename(columns={'pid': 'n_members', 'act': 'n_activities'})
        print("\n📈 Statistiques par ménage :")
        print(hh_stats.head(10))

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge synthetic schedules and attributes")
    parser.add_argument('--schedules', '-s', required=True, help='Path to synthetic_schedules.csv')
    parser.add_argument('--attributes', '-a', required=True, help='Path to synthetic_attributes.csv')
    parser.add_argument('--output', '-o', default='merged_synthetic_data.csv', help='Output path')

    args = parser.parse_args()

    # Vérifier que les fichiers existent
    if not Path(args.schedules).exists():
        print(f"❌ Fichier non trouvé : {args.schedules}")
        return

    if not Path(args.attributes).exists():
        print(f"❌ Fichier non trouvé : {args.attributes}")
        return

    # Merge
    merged = merge_synthetic_data(args.schedules, args.attributes, args.output)

    if merged is not None:
        print(f"\n✅ Succès ! Fichier merged disponible : {args.output}")
        print(f"   Colonnes : {list(merged.columns)}")


if __name__ == "__main__":
    main()
