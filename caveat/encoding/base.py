"""
Classes de base pour l'encodage des emplois du temps (schedules).

Ce fichier définit les interfaces abstraites que chaque encodeur concret doit implémenter :
  - BaseEncoder : transforme un DataFrame brut en tenseurs PyTorch
  - BaseDataset : un Dataset PyTorch qui structure les données pour l'entraînement

Lien avec les autres fichiers :
  - encoding/__init__.py contient le registre des encodeurs concrets
    (discrete, continuous, household, seq2seq...) qui héritent de BaseEncoder
  - runners.py appelle build_encoder(config) → schedule_encoder.encode() → BaseDataset
  - Le BaseDataset retourne des triplets ((x, weights), (y, weights), (labels, l_weights))
    qui sont consommés par training_step/validation_step dans experiment.py

Pourquoi cette structure de triplets :
  Le VAE a besoin de :
    - (x, weights) : l'entrée et ses poids (pour la loss pondérée par activité)
    - (y, weights) : la cible de reconstruction (souvent = x, sauf pour seq2seq)
    - (labels, l_weights) : les attributs socio-démo pour le conditionnement
  Chaque composante a ses propres poids d'équilibrage pour gérer les déséquilibres
  (activités rares, profils démographiques sous-représentés, etc.)

Variantes de Dataset :
  - BaseDataset : standard, x = y (auto-encodeur)
  - PaddedDatatset : ajoute des tokens SOS/EOS pour les modèles seq2seq
  - StaggeredDataset : décalage temporel pour la prédiction du pas suivant
  - LHS2RHSDataset : encodage asymétrique entrée ≠ sortie (ex: activité → mode de transport)
"""

from abc import ABC
from typing import Optional

from pandas import DataFrame
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset

from caveat.data import ScheduleAugment


class BaseEncoder(ABC):
    """Interface abstraite pour les encodeurs de schedules.

    Chaque encodeur concret (discrete, continuous, household...) implémente :
      - __init__() : apprend le vocabulaire d'activités depuis les données
      - encode() : transforme un DataFrame de schedules en Dataset PyTorch
      - decode() : transforme les tenseurs de sortie du modèle en DataFrame lisible

    L'encodeur est sérialisé (pickle) par runners.py pour être réutilisé lors
    de la génération depuis un checkpoint (le mapping activité→indice doit être identique).
    """

    def __init__(self, schedules: DataFrame, **kwargs) -> None:
        raise NotImplementedError

    def encode(
        self,
        schedules: DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
    ) -> Dataset:
        raise NotImplementedError

    def decode(self, schedules: Tensor) -> DataFrame:
        raise NotImplementedError


class BaseDataset(Dataset):
    """Dataset standard pour les emplois du temps encodés.

    Stocke les tenseurs de schedules encodés avec tous leurs poids associés :
      - act_weights : poids par activité dans la séquence (masque les positions PAD)
      - seq_weights : poids global par séquence (compense les séquences rares)
      - label_weights : poids par label (compense les profils sous-représentés)
      - joint_weights : poids combinant label et séquence (pour l'équilibrage joint)
      - encoding_weights : poids globaux par catégorie d'activité (pour NLLLoss pondérée)

    Le __getitem__ retourne le triplet ((x, weights), (y, weights), (labels, l_weights))
    attendu par experiment.py.
    """

    def __init__(
        self,
        schedules: Tensor,
        act_weights: Optional[Tensor],
        seq_weights: Optional[Tensor],
        activity_encodings: int,
        activity_weights: Optional[Tensor],
        augment: Optional[ScheduleAugment],
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
        joint_weights: Optional[Tensor],
    ):
        super(BaseDataset, self).__init__()
        self.schedules = schedules
        self.act_weights = act_weights
        self.seq_weights = seq_weights
        self.activity_encodings = activity_encodings
        self.encoding_weights = activity_weights
        self.augment = augment
        self.labels = labels
        self.label_weights = label_weights
        self.joint_weights = joint_weights
        self.labels_shape = labels.shape[-1] if labels is not None else None

    def shape(self):
        return self.schedules[0].shape

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([1])

        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = Tensor([1])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([1])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([1])

        return (
            (sample, (weights, seq_weights)),
            (sample, (weights, seq_weights)),
            (labels, (label_weights, joint_weights)),
        )


class PaddedDatatset(BaseDataset):
    """Dataset avec padding SOS/EOS pour les modèles auto-régressifs (seq2seq).

    Pourquoi : les décodeurs LSTM ont besoin d'un token de début (SOS) pour démarrer
    la génération et d'un token de fin (EOS) pour savoir quand s'arrêter.
    L'entrée est paddée à gauche (SOS + séquence), la cible à droite (séquence + EOS).
    """

    def shape(self):
        _, L = self.schedules.shape
        return (L + 1,)

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([1])

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = Tensor([1])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([1])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([1])

        pad_left = pad(sample, (1, 0))
        pad_right = pad(sample, (0, 1))
        return (
            (pad_left, (weights, seq_weights)),
            (pad_right, (weights, seq_weights)),
            (label, (label_weights, joint_weights)),
        )


class StaggeredDataset(BaseDataset):
    """Dataset avec décalage temporel : l'entrée est [t0..tN-1], la cible est [t1..tN].

    Pourquoi : pour les modèles de prédiction du pas suivant (next-step prediction),
    le modèle doit prédire l'activité au temps t+1 à partir de l'état au temps t.
    """

    def shape(self):
        return len(self.schedules[0]) - 1, 2

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([])

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = Tensor([])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([])

        return (
            (sample[:-1, :], (weights[:-1], seq_weights)),
            (sample[1:, :], (weights[1:], seq_weights)),
            (label, (label_weights, joint_weights)),
        )


class LHS2RHSDataset(Dataset):
    """Dataset asymétrique : l'entrée (LHS) et la sortie (RHS) ont des encodages différents.

    Pourquoi : certains modèles (seq2seq, seq2score) transforment une représentation
    en une autre (ex: séquence d'activités → score de mobilité, ou activités → modes de transport).
    L'entrée et la sortie peuvent avoir des dimensions et vocabulaires différents.
    """

    def __init__(
        self,
        lhs: Tensor,
        rhs: Tensor,
        lhs_weights: Optional[Tensor],
        rhs_weights: Optional[Tensor],
        act_encodings: int,
        mode_encodings: int,
        activity_weights: Optional[Tensor],
        augment: Optional[ScheduleAugment],
        labels: Optional[Tensor],
    ):
        super(LHS2RHSDataset, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_weights = lhs_weights
        self.rhs_weights = rhs_weights
        self.activity_encodings = (act_encodings, mode_encodings)
        self.encoding_weights = activity_weights
        self.augment = augment
        self.labels = labels
        self.labels_shape = labels.shape[-1] if labels is not None else None

    def shape(self):
        return self.lhs[0].shape

    def __len__(self):
        return len(self.lhs)

    def __getitem__(self, idx):
        lhs = self.lhs[idx]
        rhs = self.rhs[idx]
        if self.augment:
            lhs = self.augment(lhs)
            rhs = self.augment(rhs)

        if self.lhs_weights is not None:
            lhs_weight = self.lhs_weights[idx]
        else:
            lhs_weight = Tensor([])

        if self.rhs_weights is not None:
            rhs_weight = self.rhs_weights[idx]
        else:
            rhs_weight = Tensor([])

        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = Tensor([])

        if self.label_weights is not None:
            label_weight = self.label_weights[idx]
        else:
            label_weight = Tensor([])

        return (lhs, lhs_weight), (rhs, rhs_weight), (labels, label_weight)
