
from typing import Optional

import pandas as pd
import torch
from torch import Tensor

from caveat.label_encoding.label_weighting import (
    inverse_log_weight,
    inverse_weight,
    inverse_weights,
    log_inverse_weight,
    max_weight,
    product_weight,
    unit_weight,
    unit_weights,
)

# Registres des stratégies de pondération pour les labels
# 'unit' = poids uniforme (pas de rééquilibrage)
# 'inverse' = poids inversement proportionnel à la fréquence (les classes rares pèsent plus)
label_weights_library = {"unit": unit_weights, "inverse": inverse_weights}

# Pondération jointe : combine les poids de TOUS les attributs d'un individu
# en un seul scalaire. Utile pour équilibrer les combinaisons rares
# (ex: femme + retraitée + 0 véhicule = très rare dans l'enquête)
joint_label_weights_library = {
    "unit": unit_weight,
    "inverse": inverse_weight,
    "log_inverse": log_inverse_weight,
    "inverse_log": inverse_log_weight,
    "max_inverse": max_weight,
    "prod_inverse": product_weight,
}


class BaseLabelEncoder:
    """Classe de base pour l'encodage des attributs socio-démographiques.

    Les sous-classes (OneHotAttributeEncoder, TokenAttributeEncoder) implémentent
    encode() et decode() pour leur format spécifique. Le choix entre onehot et tokens
    dépend du modèle : les modèles joints (jrunners.py) nécessitent 'tokens',
    les modèles conditionnels classiques fonctionnent avec les deux.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise l'encodeur de labels avec sa configuration et ses stratégies de pondération."""
        self.config = config.copy()
        self.label_kwargs = {}
        self.weighting = kwargs.get("weighting", "unit")
        self.joint_weighting = kwargs.get("joint_weighting", "unit")

        self.label_weighter = label_weights_library.get(self.weighting, None)
        if self.label_weighter is None:
            raise ValueError(
                f"Unknown Label Encoder weighting: {self.label_weighting}, should be one of: {label_weights_library.keys()}"
            )

        self.joint_weighter = joint_label_weights_library.get(
            self.joint_weighting, None
        )
        if self.joint_weighter is None:
            raise ValueError(
                f"Unknown Label Encoder weighting: {self.joint_weighting}, should be one of: {joint_label_weights_library.keys()}"
            )

        print(
            f"""Label Encoder {self.__class__.__name__} initialised with:
            Label weighting: {self.weighting}
            Joint weighting: {self.joint_weighting}
            """
        )

    def encode(self, data: pd.DataFrame) -> Tensor:
        raise NotImplementedError

    def decode(self, data: Tensor) -> pd.DataFrame:
        raise NotImplementedError


def ordinal_encode(data: pd.Series, min, max) -> Tensor:
    """Normalise une variable continue (ex: âge) dans l'intervalle [0, 1].

    Pourquoi normaliser : les réseaux de neurones convergent mieux quand les entrées
    sont dans un intervalle borné. L'âge (0-100) et le nombre de véhicules (0-5)
    ont des échelles très différentes ; la normalisation les rend comparables.
    """
    encoded = Tensor(data.values).unsqueeze(-1)
    encoded -= min
    encoded /= max - min
    return encoded.float()


def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    """Encode une variable catégorielle en indices entiers (tokens).

    Si encodings est fourni (ex: depuis un checkpoint), on réutilise le même mapping
    pour garantir la cohérence entre entraînement et génération.
    Sinon, on construit le mapping depuis les catégories présentes dans les données.

    Retourne (tenseur d'indices, dictionnaire de mapping catégorie→indice).
    """
    if encodings:
        missing = set(data.unique()) - set(encodings.keys())
        if missing:
            raise UserWarning(
                f"""
                Categories in data do not match existing categories: {missing}.
                Please specify the new categories in the encoding.
                Your existing encodings are: {encodings}
"""
            )
        nominals = pd.Categorical(data, categories=encodings.keys())
    else:
        nominals = pd.Categorical(data)
        encodings = {e: i for i, e in enumerate(nominals.categories)}
    nominals = torch.tensor(nominals.codes).int()
    return nominals, encodings


def onehot_encode(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    nominals, encodings = tokenize(data, encodings)
    nominals = torch.nn.functional.one_hot(
        nominals.long(), num_classes=len(encodings)
    ).float()
    return nominals, encodings
