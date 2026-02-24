import torch
import yaml
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from caveat.data.household_dataset import HouseholdDataset
from caveat.models.discrete.household_vae import HouseholdDiscreteVAE

def load_checkpoint(config_path, checkpoint_path):
    # 1. Charger la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Recharger le dataset pour avoir les dictionnaires (idx_to_act)
    print("Chargement des données...")
    schedules = pd.read_csv(config["schedules_path"])
    attributes = pd.read_csv(config["attributes_path"])
    ds = HouseholdDataset(config['encoder_params'], schedules, attributes)
    
    # 3. Recharger le modèle
    print("Chargement du modèle...")
    model = HouseholdDiscreteVAE.load_from_checkpoint(
        checkpoint_path,
        in_shape=ds.shape(),
        encodings=ds.activity_encodings,
        encoding_weights=ds.encoding_weights,
        label_cardinalities=ds.label_cardinalities,
        **config["model_params"]
    )
    model.eval() # Mode évaluation (désactive dropout, etc)
    return model, ds

def visualize_reconstruction(model, dataset, num_samples=3):
    """Affiche Original vs Reconstruction pour quelques ménages"""
    idx_to_act = dataset.idx_to_act
    
    # Prendre un batch aléatoire
    dl = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dl))
    
    with torch.no_grad():
        # Forward pass
        # outputs = [log_probs, mu, logvar, z, mask]
        outputs = model(batch)
        log_probs = outputs[0] # Shape: [B, 5, 144, Classes]
        
        # On prend l'argmax pour avoir l'activité la plus probable
        reconstruction_idx = torch.argmax(log_probs, dim=-1) # Shape: [B, 5, 144]
        original_idx = batch['x'] # Shape: [B, 5, 144]
        mask = batch['mask']      # Shape: [B, 5]

    print("\n" + "="*50)
    print(f"TEST VISUEL SUR {num_samples} MENAGES")
    print("="*50)

    # Pour chaque ménage du batch
    for b in range(num_samples):
        print(f"\n--- MENAGE {b} ---")
        
        # Pour chaque membre du ménage
        for m in range(dataset.max_members):
            if mask[b, m] == 0: continue # Membre absent (padding)
            
            orig_seq = original_idx[b, m].tolist()
            recon_seq = reconstruction_idx[b, m].tolist()
            
            # On convertit en noms d'activités (on prend juste les 10 premiers pas de temps pour pas spammer)
            orig_str = [idx_to_act[i] for i in orig_seq[10:45]] # Regarde vers 12h (index 30-45 si step=144)
            recon_str = [idx_to_act[i] for i in recon_seq[10:45]]
            
            
            # Calcul de précision brute sur toute la séquence
            matches = (original_idx[b, m] == reconstruction_idx[b, m]).float().mean().item()
            
            print(f"Membre {m} (Précision: {matches:.2%})")
            print(f"  Orig (10h-15h): {orig_str}")
            print(f"  Reco (10h-15h): {recon_str}")

# --- CONFIGURATION ---
# Mets ici le chemin vers ton fichier YAML
CONFIG_PATH = "caveat/configs/household_VAE.yaml"
# Mets ici le chemin vers ton fichier .ckpt (dans logs/household_project/...)
CHECKPOINT_PATH = "logs/household_project/household_vae/version_11/checkpoints/epoch=4-step=255.ckpt" 

if __name__ == "__main__":
    # Si tu n'as pas encore de checkpoint, commente la ligne load_checkpoint 
    # et utilise ce bloc pour tester juste l'initialisation comme dans check_run
    try:
        model, ds = load_checkpoint(CONFIG_PATH, CHECKPOINT_PATH)
        visualize_reconstruction(model, ds)
    except FileNotFoundError:
        print("Fichier checkpoint non trouvé. Fais tourner 'train.py' au moins une fois.")