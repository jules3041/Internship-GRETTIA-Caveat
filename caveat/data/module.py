"""
Module de données PyTorch Lightning pour CAVEAT.

Ce fichier fait le pont entre les données encodées (tenseurs produits par encoding/) et
le système d'entraînement Lightning (experiment.py). Il gère :
  - Le découpage train/val/test des données encodées
  - La création des DataLoaders avec batching, shuffling et parallélisme
  - Les datasets spécialisés pour la phase de génération (latent sampling)

Lien avec les autres fichiers :
  - runners.py appelle build_dataloader() qui instancie DataModule avec le Dataset encodé
  - experiment.py accède aux dataloaders via self.trainer.datamodule (injecté par Lightning)
  - Les classes Z*Dataset et Conditional*Dataset sont utilisées par generate() et generate_from_attributes()
    dans runners.py pour échantillonner l'espace latent pendant la génération

"""

from typing import Callable, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import OneHotCategorical
from torch.utils.data import DataLoader, Dataset


class DataModule(LightningDataModule):
    """DataModule Lightning qui gère le split et le chargement des données encodées.

    Reçoit un Dataset (produit par encoding/base.py) et le découpe en train/val/test.
    Les proportions sont configurables via le YAML (loader_params.val_split, test_split).
    """

    def __init__(
        self,
        data: Dataset,
        val_split: float = 0.1,
        test_split: Optional[float] = None,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = True,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Initialise le DataModule avec les paramètres de chargement.

        Args:
            data (Dataset): Dataset encodé (BaseDataset, HouseholdDataset, etc.)
            val_split (float): Proportion des données pour la validation (ex: 0.1 = 10%).
            test_split (float, optional): Proportion pour le test. Si None, val = test.
            train_batch_size (int): Taille de batch pour l'entraînement.
            val_batch_size (int): Taille de batch pour la validation.
            test_batch_size (int): Taille de batch pour le test/inférence.
            num_workers (int): Nombre de processus de chargement parallèle.
            pin_memory (bool): Pré-charge les tenseurs en mémoire GPU (accélère le transfert).
            collate_fn (Callable, optional): Fonction de collation personnalisée pour les batches
                (nécessaire pour les modèles Household qui ont des structures de batch complexes).
        """
        super().__init__()

        self.data = data
        self.val_split = val_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.mapping = None
        self.collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        """Découpe le dataset en train/val/test via random_split.

        Pourquoi random_split et pas un split séquentiel : les emplois du temps
        ne sont pas ordonnés temporellement (contrairement aux séries temporelles),
        donc un split aléatoire est approprié et évite les biais de sélection.
        Si test_split est None, le jeu de test est identique au jeu de validation
        (utile pour les petits datasets où on ne peut pas se permettre 3 splits).
        """
        if self.test_split is None:
            (
                self.train_dataset,
                self.val_dataset,
            ) = torch.utils.data.random_split(
                self.data, [1 - self.val_split, self.val_split]
            )
            self.test_dataset = self.val_dataset
        else:
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = torch.utils.data.random_split(
                self.data,
                [
                    1 - self.val_split - self.test_split,
                    self.val_split,
                    self.test_split,
                ],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )


class ZNormDataset(Dataset):
    def __init__(self, num_samples: int, latent_dim: int):
        self.z = torch.randn(num_samples, latent_dim)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], torch.tensor(0)


class ZCatDataset(Dataset):
    def __init__(
        self, num_samples: int, latent_dim: int, latent_categories: int
    ):
        uniform = torch.ones((num_samples, latent_dim, latent_categories))
        uniform /= uniform.sum(dim=-1, keepdim=True)
        self.z = OneHotCategorical(uniform).sample()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], torch.tensor(0)


def build_latent_dataloader(
    num_samples,
    latent_dim: int,
    latent_categories: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = True,
):
    if latent_categories is None:
        return DataLoader(
            ZNormDataset(num_samples, latent_dim),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        return DataLoader(
            ZCatDataset(num_samples, latent_dim, latent_categories),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )


class ConditionalDataset(Dataset):
    """Dataset pour la génération CONDITIONNELLE : couple (z, attributs).

    Utilisé par generate_from_attributes() dans runners.py. Pour chaque individu
    (ou ménage), on associe un vecteur latent z aléatoire ~ N(0,I) aux attributs
    socio-démo encodés. Le modèle conditionnel utilise les deux pour générer
    un emploi du temps cohérent avec le profil demandé.
    """

    def __init__(self, attributes: torch.Tensor, latent_dim: int):
        # On détecte si les attributs sont au format ménage [N, M, num_attrs]
        # ou au format individuel [N, num_attrs] pour créer les bons latents.
        # En mode Household, chaque MEMBRE du ménage a son propre z_indiv,
        # d'où la forme 3D [N, M, latent_dim].
        if attributes.dim() == 3:
            # format ménage : on crée des latents 2D [N, M, latent_dim] (un par membre)
            N, M, _ = attributes.shape
            self.z = torch.randn(N, M, latent_dim)
        else:
            # format individuel : latents 1D [N, latent_dim]
            self.z = torch.randn(len(attributes), latent_dim)
        self.attributes = attributes

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        return self.z[idx], self.attributes[idx]


def build_latent_conditional_dataloader(
    attributes: torch.Tensor,
    latent_dim: int,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = True,
):
    print(
        f"Building dataloader with {len(attributes)} samples, latent_dim={latent_dim}, batch_size={batch_size}, num_workers={num_workers}"
    )
    return DataLoader(
        ConditionalDataset(attributes, latent_dim),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


class CustomGenDataset(Dataset):
    def __init__(self, attributes: torch.Tensor, z: torch.Tensor):
        self.z = z
        self.attributes = attributes

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        return self.z[idx], self.attributes[idx]


def build_custom_gen_dataloader(
    attributes: torch.Tensor,
    z: torch.Tensor,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = True,
):
    print(
        f"Building dataloader with {len(attributes)} samples, latent_dim={len(z)}, batch_size={batch_size}, num_workers={num_workers}"
    )
    return DataLoader(
        CustomGenDataset(attributes, z),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


class CustomIndivGenDataset(Dataset):
    def __init__(self, z_indiv: torch.Tensor, attributes: Optional[torch.Tensor] = None):
        self.z_indiv = z_indiv
        self.attributes = attributes

    def __len__(self):
        return len(self.z_indiv)

    def __getitem__(self, idx):
        item = {"z_indiv": self.z_indiv[idx]}
        if self.attributes is not None:
            item["labels"] = self.attributes[idx]
        return item


def build_custom_indiv_gen_dataloader(
    z_indiv: torch.Tensor,
    attributes: Optional[torch.Tensor] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = True,
):
    if attributes is not None and len(attributes) != len(z_indiv):
        raise ValueError(
            f"Mismatched lengths for custom indiv generation: len(attributes)={len(attributes)} "
            f"!= len(z_indiv)={len(z_indiv)}"
        )
    latent_shape = tuple(z_indiv.shape[1:]) if z_indiv.ndim >= 2 else (z_indiv.shape[-1],)
    print(
        "Building custom indiv generation dataloader with "
        f"{len(z_indiv)} samples, z_indiv_shape={latent_shape}, batch_size={batch_size}, "
        f"num_workers={num_workers}"
    )
    return DataLoader(
        CustomIndivGenDataset(z_indiv, attributes),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
