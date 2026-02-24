import yaml
import torch
import pandas as pd
from pathlib import Path
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from caveat import runners, data
from caveat.models import library
from caveat.models.discrete.household_vae import HouseholdDiscreteVAE 

# --- CONFIGURATION ---
# Assure-toi que ce chemin est correct
CHECKPOINT_PATH = r"C:\Users\hajjar\Desktop\code_caveat\logs\household_project\household_vae\version_8\checkpoints\epoch=4-step=255.ckpt"
CONFIG_PATH = r"C:\Users\hajjar\Desktop\code_caveat\caveat\configs\household_VAE.yaml"
OUTPUT_DIR = Path("outputs_generation")
# ---------------------

def main():
    print("--- 1. Initialisation ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    library["household_vae"] = HouseholdDiscreteVAE

    # --- 2. Chargement des Données (Juste pour récupérer l'encodeur et les attributs) ---
    print("--- 2. Préparation des Données ---")
    
    # On utilise runners pour charger les CSV proprement (respecte tes renommages)
    input_schedules, input_attributes, synthetic_attributes = runners.load_data(config)
    
    # On crée le dossier temporaire pour éviter le crash de pickle
    tmp_log_dir = Path("logs/tmp")
    tmp_log_dir.mkdir(parents=True, exist_ok=True)

    # On reconstruit l'encodeur (nécessaire pour décoder la sortie du modèle plus tard)
    print("🛠️  Reconstruction de l'HouseholdEncoder...")
    schedule_encoder, encoded_schedules, _ = runners.encode_schedules(
        tmp_log_dir,
        input_schedules,
        None, 
        None,
        config
    )

    # --- 3. Extraction des Attributs (Labels) pour la Génération ---
    # C'est l'étape clé : on récupère les labels encodés (indices) depuis le dataset créé
    # Le dataset a déjà fait le travail difficile (Tri par âge, Groupby HID, Mapping str->int)
    print("📥 Extraction des attributs conditionnels...")
    
    # On parcourt le dataset pour extraire tous les labels dans l'ordre
    all_labels = []
    # On utilise le dataset interne de l'encodeur qui est déjà prêt
    dataset = encoded_schedules 
    
    for i in range(len(dataset)):
        batch = dataset[i] # Récupère {x, mask, labels}
        all_labels.append(batch["labels"])
    
    # On empile tout pour avoir un tenseur [N_Menages, 5, N_Attributs]
    conditional_labels = torch.stack(all_labels)
    print(f"✅ Attributs prêts. Forme : {conditional_labels.shape}")

    # --- 4. Chargement du Modèle ---
    print(f"🧠 Chargement du modèle depuis : {CHECKPOINT_PATH}")
    model = HouseholdDiscreteVAE.load_from_checkpoint(
        CHECKPOINT_PATH,
        in_shape=encoded_schedules.shape(),
        encodings=encoded_schedules.activity_encodings,
        encoding_weights=encoded_schedules.encoding_weights,
        label_cardinalities=encoded_schedules.label_cardinalities,
        **config["model_params"]
    )
    model.eval()

    # --- 5. Préparation du DataLoader de Génération ---
    # On crée des vecteurs Z aléatoires (Bruit Gaussien)
    # Autant de Z que de ménages dans nos labels
    num_samples = conditional_labels.shape[0]
    latent_dim = model.latent_dim_hh # 32 ou 64 selon ta config
    
    print(f"🎲 Génération de {num_samples} vecteurs latents (z)...")
    z = torch.randn(num_samples, latent_dim)
    
    # On crée un DataLoader simple qui donne des paires (Labels, Z)
    # C'est ce que ton predict_step attend !
    gen_dataset = TensorDataset(conditional_labels, z)
    gen_loader = DataLoader(gen_dataset, batch_size=256, num_workers=0)

    # --- 6. Exécution (Inférence) ---
    print("🚀 Lancement de la génération...")
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)
    
    # C'EST ICI LA CORRECTION : On passe le modèle explicitement
    predictions = trainer.predict(model, dataloaders=gen_loader)
    
    # --- 7. Traitement des Résultats ---
    print("💾 Traitement et Sauvegarde...")
    
    # predict_step renvoie (labels, log_probs, z) pour le cas génération
    # On "dézippe" la liste des résultats
    _, output_log_probs, _ = zip(*predictions)
    
    # On concatène tous les batches
    output_tensor = torch.cat(output_log_probs, dim=0) # [Total, 5, 144, Classes]
    
    # --- 8. Décodage (Tenseur -> CSV lisible) ---
    # Le HouseholdEncoder actuel ne gère pas le décodage complexe vers CSV
    # On va faire un décodage basique ici pour valider
    
    # Argmax pour avoir les indices d'activités les plus probables
    generated_acts_idx = output_tensor.argmax(dim=-1) # [Total, 5, 144]
    
    # Conversion en DataFrame "plat" (PID, Start, End, Act)
    # C'est un peu technique car on doit "déplier" les ménages
    rows = []
    timestep_min = 1440 / 144 # 10 minutes
    
    # On récupère le mapping inverse (0 -> "Dormir", 1 -> "Travail"...)
    idx_to_act = schedule_encoder.idx_to_act 
    
    print("📝 Écriture du CSV...")
    # Itération sur chaque ménage généré
    for i in range(len(generated_acts_idx)):
        # Récupération du PID de base (celui des données d'entrée) pour garder le lien
        # Attention : C'est une simplification, idéalement on génère de nouveaux PIDs
        # Ici on réutilise les HID/PID d'entrée pour comparer facilement
        
        household_indices = generated_acts_idx[i] # [5, 144]
        
        # Pour chaque membre du ménage (0..4)
        for member_slot in range(5):
            acts = household_indices[member_slot].tolist()
            
            # Si le slot est vide (tout à 0 ou PAD), on ignore (détection basique)
            if sum(acts) == 0: 
                continue
                
            # Algorithme RLE (Run Length Encoding) pour passer de [1,1,1,2,2] à (Act1, 0, 30), (Act2, 30, 50)
            current_act = acts[0]
            start_step = 0
            
            # On génère un PID unique fictif : "Gen_MenageID_MembreID"
            gen_pid = f"Gen_{i}_{member_slot}"
            
            for t, act in enumerate(acts):
                if act != current_act:
                    # Fin de l'activité précédente
                    act_name = idx_to_act.get(current_act, "Unknown")
                    if act_name != "PAD": # On n'écrit pas le padding
                        rows.append({
                            "pid": gen_pid,
                            "hid": i, # Lien avec le ménage i
                            "member_idx": member_slot,
                            "act": act_name,
                            "start": start_step * timestep_min,
                            "end": t * timestep_min,
                            "duration": (t - start_step) * timestep_min
                        })
                    current_act = act
                    start_step = t
            
            # Dernière activité
            act_name = idx_to_act.get(current_act, "Unknown")
            if act_name != "PAD":
                rows.append({
                    "pid": gen_pid,
                    "hid": i,
                    "member_idx": member_slot,
                    "act": act_name,
                    "start": start_step * timestep_min,
                    "end": 1440, # Fin de journée
                    "duration": (144 - start_step) * timestep_min
                })

    df_result = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "synthetic_schedules.csv"
    df_result.to_csv(out_path, index=False)
    
    print(f"✅ Terminé ! Fichier généré : {out_path}")
    print(f"   Nombre d'activités générées : {len(df_result)}")

if __name__ == "__main__":
    main()