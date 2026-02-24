import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from caveat.models import Base


class LabelEncoder(nn.Module):
    """
    Encode les attributs en un vecteur partagé -> c'est à dire embedding puis hidden_size.
    Mode "sum" = somme des embeddings, mode "concat" = concaténation.
    """

    def __init__(self, label_cardinalities: List[int], hidden_size: int, mode: str = "sum"):
        super().__init__()
        if mode not in {"sum", "concat"}:
            raise ValueError("LabelEncoder mode must be 'sum' or 'concat'.")
        self.mode = mode

        # dans tous les cas on crée un embedding par attribut
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, hidden_size) for card in label_cardinalities]
        )
        # si on additionne, la dimension de sortie est hidden_size, sinon c'est hidden_size * n_attrs
        self.output_dim = (
            hidden_size if self.mode == "sum" else hidden_size * len(label_cardinalities)
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: [B, M, N_attrs]
        returns: [B, M, output_dim]
        """
        # on récupère l'attribut et la couche correspondante et on le fait passer dans cette couche
        embeds = [emb(labels[:, :, i]) for i, emb in enumerate(self.embeddings)]
        # si on somme, on empile puis somme, sinon on concatène
        if self.mode == "sum":
            return torch.stack(embeds, dim=-1).sum(dim=-1)
        return torch.cat(embeds, dim=-1)


class ChannelwiseLayerNorm2d(nn.Module):
    """LayerNorm sur la dimension des canaux [B, C, M, T]
    donc on normalise sur les canaux pour chaque pair (M, T)
    pas de contamination des membres fantômes"""

    def __init__(self, num_channels: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        # elementwise pour moduler la normalisation
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # [B, M, T, C]
        x = self.ln(x)
        # retour format image
        return x.permute(0, 3, 1, 2)  # [B, C, M, T]


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation
    y = gamma * x + beta, où gamma et beta sont appris depuis les attributs
    """

    def __init__(self, labels_dim: int, feature_dim: int):
        super().__init__()
        #labels_dim sort de label encoder
        # on projette vers la dimension des canaux pour moduler
        self.gamma_proj = nn.Linear(labels_dim, feature_dim)
        self.beta_proj = nn.Linear(labels_dim, feature_dim)

        # influence très faible au début, on apprend progressivement
        nn.init.normal_(self.gamma_proj.weight, std=0.02)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.normal_(self.beta_proj.weight, std=0.02)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, labels_hidden: torch.Tensor) -> torch.Tensor:
        B, C, M, T = x.shape
        # gamma et beta sont insensible au temps. on veut la même valeur partout
        gamma = self.gamma_proj(labels_hidden).permute(0, 2, 1).unsqueeze(-1)  # [B, C, M, 1]
        beta = self.beta_proj(labels_hidden).permute(0, 2, 1).unsqueeze(-1)

        return gamma * x + beta


class AddConditioner(nn.Module):
    """Conditioning par addition : x + proj(labels)."""

    def __init__(self, labels_dim: int, feature_dim: int):
        super().__init__()
        # pour additionner il faut que la dimension corresponde
        self.proj = nn.Linear(labels_dim, feature_dim)

    def forward(self, x: torch.Tensor, labels_hidden: torch.Tensor) -> torch.Tensor:
        B, C, M, T = x.shape
        # étire sur le temps
        cond = self.proj(labels_hidden).permute(0, 2, 1).unsqueeze(-1)  # [B, C, M, 1]

        return x + cond


class ConcatConditioner(nn.Module):
    """Conditioning par concaténation : [x, proj(labels)]."""

    def __init__(self, labels_dim: int, cond_hidden_size: int):
        super().__init__()
        # on choisit la taille de la projection à concatener
        self.cond_hidden_size = cond_hidden_size
        self.proj = nn.Linear(labels_dim, cond_hidden_size)

    def forward(self, x: torch.Tensor, labels_hidden: torch.Tensor) -> torch.Tensor:
        B, C, M, T = x.shape
        # étire sur le temps
        cond = self.proj(labels_hidden).permute(0, 2, 1).unsqueeze(-1)  # [B, cond_hidden, M, 1]
        cond = cond.expand(-1, -1, -1, T)  # [B, cond_hidden, M, T]

        return torch.cat([x, cond], dim=1)  # [B, C+cond_hidden, M, T]


class LatentAddConditioner(nn.Module):
    """Conditioning du latent par addition
    même principe que AddConditioner mais sur le latent (pas de dimension temps)"""

    def __init__(self, labels_dim: int, latent_dim: int):
        super().__init__()
        # les features sont le latent_dim
        self.proj = nn.Linear(labels_dim, latent_dim)

    def forward(self, z: torch.Tensor, labels_hidden: torch.Tensor) -> torch.Tensor:
        """z: [B, M, latent_dim], labels_hidden: [B, M, labels_dim]"""
        cond = self.proj(labels_hidden)  # [B, M, latent_dim]
        return z + cond


class LatentConcatConditioner(nn.Module):
    """Conditioning du latent par concaténation
    même principe que ConcatConditioner mais sur le latent (pas de dimension temps)"""

    def __init__(self, labels_dim: int, cond_hidden_size: int):
        super().__init__()
        self.cond_hidden_size = cond_hidden_size
        self.proj = nn.Linear(labels_dim, cond_hidden_size)

    def forward(self, z: torch.Tensor, labels_hidden: torch.Tensor) -> torch.Tensor:
        """z: [B, M, latent_dim], labels_hidden: [B, M, labels_dim]"""
        cond = self.proj(labels_hidden)  # [B, M, cond_hidden_size]
        return torch.cat([z, cond], dim=-1)  # [B, M, latent_dim + cond_hidden_size]


class GatedBottleneckSetAttentionCoordBlock(nn.Module):
    """
    coordination via self-attention entre les membres à chaque pas de temps
    avec un mécanisme de gate pour moduler l'importance de l'attention
    on les place dans l'encodeur et dans le decodeur
    """
    def __init__(
        self,
        channels: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout_attn: float = 0.1,
        dropout_ff: float = 0.1,
        dropout_resid: float = 0.0,
        gate_init: float = -4.0,
    ):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})."
            )
        # correspond à une linear pour atteindre la dimension d'attention
        # 512 parait trop élevé donc on réduit
        self.down = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)

        # idem pour remonter à la dimension originale après attention
        self.up = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)

        # multihead attention de pytorch avec dropout
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout_attn,
            batch_first=True,
        )
        # je ne sais pas encore si faut utiliser le modèle custom 
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        # si on veut dropout sur la sortie de l'attention et pas sur les poids
        self.drop_attn = nn.Dropout(dropout_resid) if dropout_resid > 0 else nn.Identity()

        # feedforward pour traiter les tokens après attention comme dans attention is all you need
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, ff_mult * attn_dim),
            nn.GELU(),
            nn.Dropout(dropout_ff),
            nn.Linear(ff_mult * attn_dim, attn_dim),
            nn.Dropout(dropout_ff),
        )
        # gate pour moduler l'importance de l'attention (== de chaque channels)
        self.gate = nn.Parameter(torch.full((channels,), float(gate_init)))
        self.last_attn = None # pour log
        self.last_kpm = None # pour log 

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> torch.Tensor:
        B, C, M, W = x.shape
        mask_bool = mask.bool()
        mask_4d = mask_bool.view(B, 1, M, 1).float()

        x = x * mask_4d # on masque les membres fantômes avant attention pour éviter la contamination meme si c'est gérer par key padding

        y = self.down(x)  # [B, d, M, W]
        d = y.shape[1]

        tokens = y.permute(0, 3, 2, 1).reshape(B * W, M, d)  # [B*W, M, d] -> un token par membre et par pas de temps, on traite les membres comme un set à chaque pas de temps
        kpm = (~mask_bool).unsqueeze(1).expand(-1, W, -1).reshape(B * W, M)  # [B*W, M] # 1 si fantome, 0 si réel comme le veut le kpm

        h = self.norm1(tokens) # on normalise avant attention comme dans les transformers 
        if need_weights:
            attn_out, attn_weights = self.mha(
                h,
                h,
                h,
                key_padding_mask=kpm, # ça met les logits d'attention à -inf pour les membres fantômes
                need_weights=True, # pour log
                average_attn_weights=average_attn_weights,
            )
            self.last_attn = attn_weights.detach()
            self.last_kpm = kpm.detach()
        else:
            attn_out, _ = self.mha(h, h, h, key_padding_mask=kpm, need_weights=False)
            self.last_attn = None
            self.last_kpm = None
        tokens = tokens + self.drop_attn(attn_out) # résiduel + attention (= qui)

        h = self.norm2(tokens) # on normalise avant feedforward comme dans les transformers
        tokens = tokens + self.ff(h) # résiduel + feedforward

        y = tokens.reshape(B, W, M, d).permute(0, 3, 2, 1) # retour format image
        y = y * mask_4d # au cas où

        y = self.up(y)  # [B, C, M, W] retourne à C pour le CNN

        alpha = torch.sigmoid(self.gate).view(1, -1, 1, 1) # sigmoid comme gate pour être entre 0 et 1
        out = (x + alpha * y) * mask_4d # controle l'apport de l'attention
        return out

class EncoderConvBlock(nn.Module):
    """Bloc de convolution de l'encodeur avec option d'attention et de conditionnement"""
    def __init__(
        self,
        base_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str,
        dropout: float,
        cond_type: str,
        labels_dim: int,
        cond_hidden_size: int,
        attn_cfg: Optional[dict],
        use_attention: bool,
    ):
        super().__init__()
        self.labels_dim = labels_dim
        self.cond_type = cond_type
        self.conditioner = None
        # bloc de conditionnement
        if cond_type != "none" and labels_dim > 0:
            if cond_type == "film":
                self.conditioner = FiLMConditioner(labels_dim, base_channels)
            elif cond_type == "add":
                self.conditioner = AddConditioner(labels_dim, base_channels)
            elif cond_type == "concat":
                self.conditioner = ConcatConditioner(labels_dim, cond_hidden_size)
                
            else:
                raise ValueError(f"Unknown encoder_conditionality: {cond_type}")
        # si concat on doit augmenter la taille pré et post latent
        in_channels = base_channels + (
            cond_hidden_size if (cond_type == "concat" and labels_dim > 0) else 0
        )
        # bloc d'attention
        self.attention = None
        if use_attention and attn_cfg is not None:
            self.attention = GatedBottleneckSetAttentionCoordBlock(
                channels=in_channels,
                attn_dim=attn_cfg["attn_dim"],
                num_heads=attn_cfg["num_heads"],
                ff_mult=attn_cfg["ff_mult"],
                dropout_attn=attn_cfg["dropout_attn"],
                dropout_ff=attn_cfg["dropout_ff"],
                dropout_resid=attn_cfg["dropout_resid"],
                gate_init=attn_cfg["gate_init"],
            )
        # bloc de convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            padding_mode=padding_mode,
            bias=False,
        )
        self.norm = ChannelwiseLayerNorm2d(out_channels)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        labels_hidden: Optional[torch.Tensor],
        need_attn_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> torch.Tensor:
        B, C, M, W = x.shape
        # bloc de conditionnement
        if self.conditioner is not None:
            if labels_hidden is None:
                labels_hidden = torch.zeros(
                    B, M, self.labels_dim, device=x.device, dtype=x.dtype
                )
            x = self.conditioner(x, labels_hidden)

        if mask is None:
            mask = torch.ones(B, M, device=x.device, dtype=torch.bool)
        mask_4d = mask.view(B, 1, M, 1).float()
        # on masque les membres fantômes post conditionnement 
        x = x * mask_4d
        # bloc d'attention
        if self.attention is not None:
            x = self.attention(
                x,
                mask,
                need_weights=need_attn_weights,
                average_attn_weights=average_attn_weights,
            )
        # bloc de conv
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DecoderConvBlock(nn.Module):
    def __init__(
        self,
        base_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        dropout: float,
        cond_type: str,
        labels_dim: int,
        cond_hidden_size: int,
        attn_cfg: Optional[dict],
        use_attention: bool,
        is_last: bool,
    ):
        """Bloc de convolution du décodeur avec option d'attention et de conditionnement"""
        super().__init__()
        self.labels_dim = labels_dim
        self.cond_type = cond_type
        self.conditioner = None
        # bloc de conditionnement
        if cond_type != "none" and labels_dim > 0:
            if cond_type == "film":
                self.conditioner = FiLMConditioner(labels_dim, base_channels)
            elif cond_type == "add":
                self.conditioner = AddConditioner(labels_dim, base_channels)
            elif cond_type == "concat":
                self.conditioner = ConcatConditioner(labels_dim, cond_hidden_size)
            else:
                raise ValueError(f"Unknown decoder_conditionality: {cond_type}")
        # si concat on doit augmenter la taille pré et post latent
        in_channels = base_channels + (
            cond_hidden_size if (cond_type == "concat" and labels_dim > 0) else 0
        )
        # bloc d'attention
        self.attention = None
        if use_attention and attn_cfg is not None:
            self.attention = GatedBottleneckSetAttentionCoordBlock(
                channels=in_channels,
                attn_dim=attn_cfg["attn_dim"],
                num_heads=attn_cfg["num_heads"],
                ff_mult=attn_cfg["ff_mult"],
                dropout_attn=attn_cfg["dropout_attn"],
                dropout_ff=attn_cfg["dropout_ff"],
                dropout_resid=attn_cfg["dropout_resid"],
                gate_init=attn_cfg["gate_init"],
            )
        # bloc de convolution
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            output_padding=(0, output_padding),
            bias=False,
        )
        self.norm = ChannelwiseLayerNorm2d(out_channels)
        self.act = nn.LeakyReLU() if not is_last else None
        self.drop = nn.Dropout(dropout) if not is_last else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        labels_hidden: Optional[torch.Tensor],
        need_attn_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> torch.Tensor:
        B, C, M, W = x.shape
        # bloc de conditionnement
        if self.conditioner is not None:
            if labels_hidden is None:
                labels_hidden = torch.zeros(
                    B, M, self.labels_dim, device=x.device, dtype=x.dtype
                )
            x = self.conditioner(x, labels_hidden)

        if mask is None:
            mask = torch.ones(B, M, device=x.device, dtype=torch.bool)
        mask_4d = mask.view(B, 1, M, 1).float()
        # on masque les membres fantômes post conditionnement
        x = x * mask_4d
        # bloc d'attention
        if self.attention is not None:
            x = self.attention(
                x,
                mask,
                need_weights=need_attn_weights,
                average_attn_weights=average_attn_weights,
            )
        # bloc de conv
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
            x = self.drop(x)
        return x

class HouseholdDiscCNNCVAEAttention(Base):
    def __init__(self, encoding_weights=None, *args, **kwargs):
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        if encoding_weights is not None:
            self.register_buffer("class_weights", encoding_weights)
        else:
            self.class_weights = None

        # KLD annealing
        self.kld_annealing_type = kwargs.get("kld_annealing_type", None)
        self.kld_annealing_epochs = kwargs.get("kld_annealing_epochs", 100)
        self.kld_start_weight = kwargs.get("kld_start_weight", 0.0)
        self.kld_max_weight = kwargs.get("kld_max_weight", None)

        self.kld_n_cycles = kwargs.get("kld_n_cycles", 4) # Nombre de cycles M
        self.kld_cycle_ratio = kwargs.get("kld_cycle_ratio", 0.5) # Ratio R (proportion de montée vs plateau)

        if self.kld_annealing_type:
            print(f"KLD Annealing: {self.kld_annealing_type}, Max={self.kld_max_weight}")
            if self.kld_annealing_type == "cyclical":
                print(f" -> Cycles: {self.kld_n_cycles}, Ratio: {self.kld_cycle_ratio}")
            self.scheduled_kld_weight = self.kld_start_weight # poids de départ (0 dans notre cas)

        self.automatic_optimization = True

    def build(self, **config):
        # dimensions principales
        # un peu plus grand car on rajoute les attributs
        self.latent_dim = config.get("latent_dim", 8)
        self.hidden_size = config.get("hidden_size", 512)
        # pour embedding
        self.num_activities = int(self._encodings_backup or config.get("encodings"))
        self.max_members = int(config.get("max_members", 5))
        self.time_steps = int(config.get("time_steps", 144))

        # patch ajout temporel
        # mais on prefere le padding
        self.use_monkey_patch = config.get("use_monkey_patch", False)
        self.monkey_patch_extra_steps = config.get("monkey_patch_extra_steps", 24)
        self.home_activity_index = config.get("home_activity_index", 0)

        self.original_time_steps = self.time_steps
        if self.use_monkey_patch:
            self.extended_time_steps = self.time_steps + self.monkey_patch_extra_steps
            print(
                f"Monkey patch enabled: {self.time_steps} -> {self.extended_time_steps} steps"
            )
        else:
            self.extended_time_steps = self.time_steps

        # paramètres CNN
        self.kernel_size = config.get("kernel_size", 4)
        self.stride = config.get("stride", 2)
        self.padding = config.get("padding", 0)
        self.dropout_rate = config.get("dropout", 0.1)
        self.num_layers = config.get("num_layers", 4)

        # Déterminer le padding mode
        self.use_replicate_padding = config.get("use_replicate_padding", False)
        self.use_circular_padding = config.get("use_circular_padding", False)
        if self.use_circular_padding:
            self.padding_mode = "circular"
        elif self.use_replicate_padding:
            self.padding_mode = "replicate"
        else:
            self.padding_mode = "zeros"

        # le script trouve automatiquement les deux
        self.hh_size_col_idx = config.get("hh_size_col_idx", None)
        hh_size_map = config.get("hh_size_map", None)
        if hh_size_map is not None:
            self.register_buffer(
                "hh_size_map_tensor", torch.tensor(hh_size_map, dtype=torch.long)
            )
        else:
            self.hh_size_map_tensor = None

        # besoin de la cardinalité pour l'embedding
        self.label_cardinalities = config.get("label_cardinalities", [])
        # active le conditionning
        self.use_conditioning = len(self.label_cardinalities) > 0

        self.conditioning_type = config.get("conditioning_type", "film")  # film, add, concat, none
        self.conditioning_hidden_size = config.get("conditioning_hidden_size", 64) # pour embedding concat (les autres c'est taille features)

        # Label encoder == comment on crée le vecteur d'attributs à partir de l'encoding brut
        # soit sum (comme dans actvae), soit concat
        self.label_encoder_mode = config.get("label_encoder_mode", "sum")
        self.labels_hidden_size = config.get(
            "labels_hidden_size", config.get("conditioning_embed_dim", 16)
        ) # taille embedding par attribut avant combinaison
        self.label_encoder = (
            LabelEncoder(
                self.label_cardinalities,
                hidden_size=self.labels_hidden_size,
                mode=self.label_encoder_mode,
            )
            if self.use_conditioning
            else None
        )
        # sortie de label encoder
        self.labels_condition_dim = (
            self.label_encoder.output_dim if self.label_encoder is not None else 0
        )

        # pour choisir le type de conditionnalité par endroit
        self.encoder_conditionality = config.get("encoder_conditionality", None)
        self.latent_conditionality = config.get("latent_conditionality", None)
        self.decoder_conditionality = config.get("decoder_conditionality", None)
        # sinon on prend le défaut
        if self.encoder_conditionality is None:
            self.encoder_conditionality = (
                self.conditioning_type if config.get("encoder_conditioning", True) else "none"
            )
        if self.decoder_conditionality is None:
            self.decoder_conditionality = (
                self.conditioning_type if config.get("decoder_conditioning", True) else "none"
            )

        # flags booléens pour la suite
        self.encoder_conditioning = self.encoder_conditionality != "none"
        self.decoder_conditioning = self.decoder_conditionality != "none"

        if config.get("latent_conditioning", False) or self.latent_conditionality not in (None, "none"):
            print("[CVAE-Attn] Latent conditioning is disabled in this attention model.")
        self.latent_conditionality = "none"
        self.latent_conditioning = False

        print(
            f"[CVAE-Attn] Conditioning: type={self.conditioning_type}, "
            f"encoder={self.encoder_conditionality} (per-layer), latent=none, "
            f"decoder={self.decoder_conditionality} (per-layer), label_mode={self.label_encoder_mode}"
        )

        # Attention config
        self.attn_dim = config.get("attn_dim", 128)
        self.attn_heads = config.get("attn_heads", 4)
        self.attn_ff_mult = config.get("attn_ff_mult", 2)
        self.attn_dropout_attn = config.get("attn_dropout_attn", 0.1)
        self.attn_dropout_ff = config.get("attn_dropout_ff", 0.1)
        self.attn_dropout_resid = config.get("attn_dropout_resid", 0.0)
        self.attn_gate_init = config.get("attn_gate_init", -4.0)
        self.attn_cfg = {
            "attn_dim": self.attn_dim,
            "num_heads": self.attn_heads,
            "ff_mult": self.attn_ff_mult,
            "dropout_attn": self.attn_dropout_attn,
            "dropout_ff": self.attn_dropout_ff,
            "dropout_resid": self.attn_dropout_resid,
            "gate_init": self.attn_gate_init,
        }

        # où l'activer ? c'est à dire à quelles couches de l'encodeur et du décodeur 
        self.enc_attn_indices = config.get("enc_attn_indices", [0, 1])
        self.dec_attn_indices = config.get("dec_attn_indices", [0, 1])
        self.use_enc_post_attn = config.get("enc_post_attn", True)
        self.use_dec_post_attn = config.get("dec_post_attn", True)
        # safe
        self.enc_attn_indices = [
            i for i in self.enc_attn_indices if 0 <= i < self.num_layers
        ]
        self.dec_attn_indices = [
            i for i in self.dec_attn_indices if 0 <= i < self.num_layers
        ]
        print(
            f"[CVAE-Attn] Attention: enc_pre={self.enc_attn_indices}, "
            f"enc_post={self.use_enc_post_attn}, dec_pre={self.dec_attn_indices}, "
            f"dec_post={self.use_dec_post_attn}"
        )
        # logging de l'attention et des gates pour analyse 
        self.log_attention = config.get("log_attention", False)
        self.log_attention_every_n_steps = config.get("log_attention_every_n_steps", 200)
        self.log_attention_average_heads = config.get("log_attention_average_heads", True)
        self.log_attention_hist = config.get("log_attention_hist", False)
        self.log_attention_images = config.get("log_attention_images", True)
        self.log_gates = config.get("log_gates", self.log_attention)
        self.log_gates_every_n_steps = config.get(
            "log_gates_every_n_steps", self.log_attention_every_n_steps
        )

        # embedding activité
        self.embedding = nn.Embedding(self.num_activities, self.hidden_size, padding_idx=0)

        # crée l'encodeur CNN
        self.encoder_blocks, self.enc_post_attn = self._build_encoder()

        # calculer dynamiquement la taille post encodeur
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.hidden_size, self.max_members, self.extended_time_steps
            )
            dummy_mask = torch.ones(1, self.max_members)
            dummy_labels = None
            if self.use_conditioning and self.encoder_conditioning:
                dummy_labels = torch.zeros(1, self.max_members, self.labels_condition_dim)
            enc_out = self._encode_features(dummy, dummy_mask, dummy_labels)
            self.c_out = enc_out.shape[1]
            self.h_out = enc_out.shape[2]
            self.w_out = enc_out.shape[3]
            print(f"Encoder output shape: [{self.c_out}, {self.h_out}, {self.w_out}]")
            print(f"Time compression: {self.extended_time_steps} -> {self.w_out}")

        self.flat_per_member = self.c_out * self.w_out

        self.dec_latent_norm = ChannelwiseLayerNorm2d(self.c_out)

        # projections latentes par membre (INDEPENDANTES)
        self.fc_mu = nn.Linear(self.flat_per_member, self.latent_dim)
        self.fc_var = nn.Linear(self.flat_per_member, self.latent_dim)
        # on reprojette du latent vers l'espace des features
        self.fc_decode = nn.Linear(self.latent_dim, self.c_out * self.w_out)

        # decodeur CNN
        self.decoder_blocks, self.dec_post_attn = self._build_decoder()

        # projette vers les logits puis log probs
        self.final_projection = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.num_activities, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

        print(f"Model built: latent_dim={self.latent_dim}, hidden_size={self.hidden_size}")
        print(f"2D Latent shape: [B, {self.max_members}, {self.latent_dim}]")

    def _build_encoder(self):
        # construit l'encodeur CNN
        # suffit de spécifier le nombre de couches et les paramètres
        # et si on veut de l'attention et à quelles couches
        blocks = nn.ModuleList()
        for i in range(self.num_layers):
            # toujours hidden_size en sortie (512 -> 512 -> ...)
            use_attn = i in self.enc_attn_indices
            blocks.append(
                EncoderConvBlock(
                    base_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                    dropout=self.dropout_rate,
                    cond_type=self.encoder_conditionality
                    if self.encoder_conditioning
                    else "none",
                    labels_dim=self.labels_condition_dim,
                    cond_hidden_size=self.conditioning_hidden_size,
                    attn_cfg=self.attn_cfg,
                    use_attention=use_attn,
                )
            )
        # attention de coordination post encodeur avant ff et latents
        post_attn = None
        if self.use_enc_post_attn:
            post_attn = GatedBottleneckSetAttentionCoordBlock(
                channels=self.hidden_size,
                attn_dim=self.attn_cfg["attn_dim"],
                num_heads=self.attn_cfg["num_heads"],
                ff_mult=self.attn_cfg["ff_mult"],
                dropout_attn=self.attn_cfg["dropout_attn"],
                dropout_ff=self.attn_cfg["dropout_ff"],
                dropout_resid=self.attn_cfg["dropout_resid"],
                gate_init=self.attn_cfg["gate_init"],
            )
        return blocks, post_attn

    def _build_decoder(self):
        blocks = nn.ModuleList()

        # calcul le chemin de la taille temporelle dans l'encodeur
        # en gros on veut calculer T' pour remonter à T
        sizes = [self.extended_time_steps]
        size = self.extended_time_steps
        for _ in range(self.num_layers):
            size = (size + 2 * self.padding - self.kernel_size) // self.stride + 1
            sizes.append(size)
        sizes.reverse()

        for i in range(self.num_layers):
            out_channels = self.hidden_size
            target_size = sizes[i + 1]
            current_size = sizes[i]
            # on calcule l'output padding nécessaire pour atteindre la taille cible
            # ConvTranspose: out = (in - 1) * stride - 2*padding + kernel + output_padding
            # padding = 1 ici
            expected = (current_size - 1) * self.stride - 2 * self.padding + self.kernel_size
            output_padding = max(0, target_size - expected)
            use_attn = i in self.dec_attn_indices
            # pas d'activation ni dropout à la dernière couche (éviter les biais et le bruit)
            blocks.append(
                DecoderConvBlock(
                    base_channels=self.hidden_size,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=output_padding,
                    dropout=self.dropout_rate,
                    cond_type=self.decoder_conditionality
                    if self.decoder_conditioning
                    else "none",
                    labels_dim=self.labels_condition_dim,
                    cond_hidden_size=self.conditioning_hidden_size,
                    attn_cfg=self.attn_cfg,
                    use_attention=use_attn,
                    is_last=(i == self.num_layers - 1),
                )
            )
        # attention de coordination post décodeur avant projection finale
        post_attn = None
        if self.use_dec_post_attn:
            post_attn = GatedBottleneckSetAttentionCoordBlock(
                channels=self.hidden_size,
                attn_dim=self.attn_cfg["attn_dim"],
                num_heads=self.attn_cfg["num_heads"],
                ff_mult=self.attn_cfg["ff_mult"],
                dropout_attn=self.attn_cfg["dropout_attn"],
                dropout_ff=self.attn_cfg["dropout_ff"],
                dropout_resid=self.attn_cfg["dropout_resid"],
                gate_init=self.attn_cfg["gate_init"],
            )
        return blocks, post_attn

    def _apply_monkey_patch(self, x: torch.Tensor) -> torch.Tensor:
        """étend la séquence temporelle avec des steps supplémentaires (activité home)"""
        B, M, T = x.shape
        extra = self.monkey_patch_extra_steps
        home_padding = torch.full(
            (B, M, extra),
            self.home_activity_index,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, home_padding], dim=2)

    def _encode_labels(
        self, labels: Optional[torch.Tensor], mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        # active le forward de LabelEncoder
        if labels is None or not self.use_conditioning or self.label_encoder is None:
            return None
        labels_hidden = self.label_encoder(labels.long())
        # on pousse l'embedding à 0 pour les membres absents
        if mask is not None:
            labels_hidden = labels_hidden * mask.unsqueeze(-1).float()
        return labels_hidden

    def _get_tb_writer(self):
        logger = getattr(self, "logger", None)
        if logger is None:
            return None
        exp = getattr(logger, "experiment", None)
        if exp is None:
            return None
        if not hasattr(exp, "add_scalar"):
            return None
        return exp

    def _should_log_attention(self):
        # est ce qu'on doit logger les stats d'attention à ce step d'entraînement
        if not self.log_attention:
            return False
        if self.log_attention_every_n_steps <= 0:
            return False
        return (self.global_step % self.log_attention_every_n_steps) == 0

    def _should_log_gates(self):
        # est ce qu'on doit logger les valeurs des gates à ce step d'entraînement
        if not self.log_gates:
            return False
        if self.log_gates_every_n_steps <= 0:
            return False
        return (self.global_step % self.log_gates_every_n_steps) == 0

    def _iter_attention_blocks(self):
        # Générateur utilitaire pour parcourir tous les blocs d'attention actifs du modèle
        # utile pour pas dupliquer les boucles de logging dans l'encodeur et le décodeur
        for i, block in enumerate(self.encoder_blocks):
            if block.attention is not None:
                yield f"enc_block_{i}", block.attention
        if self.enc_post_attn is not None:
            yield "enc_post_attn", self.enc_post_attn
        for i, block in enumerate(self.decoder_blocks):
            if block.attention is not None:
                yield f"dec_block_{i}", block.attention
        if self.dec_post_attn is not None:
            yield "dec_post_attn", self.dec_post_attn

    @staticmethod
    def _compute_attention_stats(weights, kpm, eps: float = 1e-8):
        # Calcule des statistiques sur les poids d'attention
        if weights is None or kpm is None:
            return None
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)
        if weights.dim() != 4:
            return None

        BW, H, M, _ = weights.shape
        if kpm.dim() != 2 or kpm.shape[0] != BW or kpm.shape[1] != M:
            return None
        # # ~kpm car kpm=True signifie "ignoré", donc active=True signifie "membre réel"
        active = ~kpm
        n_active = active.sum(dim=1)  # [BW] nombre de membres actifs par batch item
        active_any = n_active > 0
        ## Stats basiques sur la composition des ménages
        if active_any.any():
            active_members_mean = n_active[active_any].float().mean().item()
            frac_size1 = (n_active[active_any] == 1).float().mean().item()
        else:
            active_members_mean = 0.0
            frac_size1 = 0.0
        # calcule les stats d'attention que s'il y a au moins 2 membres
        valid_case = n_active >= 2
        valid_rows = active & valid_case.unsqueeze(1)  # [BW, M]
        if valid_rows.sum() == 0:
            return {
                "active_members_mean": active_members_mean,
                "frac_size1": frac_size1,
                "per_head": None,
                "mean": None,
            }
        #masquage pour ne garder que les membres réels dans le calcul des stats d'attention
        key_mask = active.view(BW, 1, 1, M)
        w = weights * key_mask
        # Calcul des métriques par tête et par Membre
        diag = w.diagonal(dim1=-2, dim2=-1)  # [BW, H, M] self attention
        rowmax = w.max(dim=-1).values # pic d'attention
        entropy = -(w * (w.clamp_min(eps).log())).sum(dim=-1) # dispersion de l'attention
        offdiag = 1.0 - diag # attention portée aux autres
        # Moyennes pondérées
        mask = valid_rows.unsqueeze(1)  # [BW, 1, M]
        denom = mask.sum(dim=(0, 2)).clamp(min=1)

        rowmax_mean_h = (rowmax * mask).sum(dim=(0, 2)) / denom
        entropy_mean_h = (entropy * mask).sum(dim=(0, 2)) / denom
        diag_mean_h = (diag * mask).sum(dim=(0, 2)) / denom
        offdiag_mean_h = (offdiag * mask).sum(dim=(0, 2)) / denom

        per_head = {
            "attn_rowmax_mean": rowmax_mean_h,
            "attn_entropy_mean": entropy_mean_h,
            "attn_diag_mass_mean": diag_mean_h,
            "attn_offdiag_mass_mean": offdiag_mean_h,
        }
        # Moyenne globale sur toutes les têtes
        mean = {k: v.mean().item() for k, v in per_head.items()}

        return {
            "active_members_mean": active_members_mean,
            "frac_size1": frac_size1,
            "per_head": per_head,
            "mean": mean,
        }

    def _log_attention_and_gates(self, attn_maps, stage: str):
        # Envoie les stats, histogrammes et images vers TensorBoard.
        writer = self._get_tb_writer()
        if writer is None:
            return
        step = int(self.global_step)

        if self.log_attention and attn_maps:
            for item in attn_maps:
                name = item["name"]
                weights = item.get("weights")
                kpm = item.get("kpm")
                stats = self._compute_attention_stats(weights, kpm)
                if stats is None:
                    continue

                writer.add_scalar(
                    f"{stage}/attn/{name}_active_members_mean",
                    stats["active_members_mean"],
                    step,
                )
                writer.add_scalar(
                    f"{stage}/attn/{name}_frac_size1",
                    stats["frac_size1"],
                    step,
                )

                per_head = stats["per_head"]
                mean = stats["mean"]
                if per_head is not None and mean is not None:
                    for metric, values in per_head.items():
                        for h, v in enumerate(values):
                            writer.add_scalar(
                                f"{stage}/attn/{name}_h{h}_{metric}",
                                v.item(),
                                step,
                            )
                        writer.add_scalar(
                            f"{stage}/attn/{name}_{metric}", mean[metric], step
                        )

                if weights is not None:
                    w = weights.float()
                    if self.log_attention_hist and hasattr(writer, "add_histogram"):
                        writer.add_histogram(f"{stage}/attn/{name}", w, step)

                    if self.log_attention_images and hasattr(writer, "add_image"):
                        if w.dim() == 4:
                            mean_map = w.mean(dim=(0, 1))
                        elif w.dim() == 3:
                            mean_map = w.mean(dim=0)
                        else:
                            mean_map = None
                        if mean_map is not None:
                            img = mean_map.unsqueeze(0)
                            writer.add_image(
                                f"{stage}/attn_map/{name}",
                                img,
                                step,
                                dataformats="CHW",
                            )

        if self.log_gates:
            for name, block in self._iter_attention_blocks():
                gate = torch.sigmoid(block.gate.detach())
                writer.add_scalar(
                    f"{stage}/gates/{name}_mean", gate.mean().item(), step
                )
                if self.log_attention_hist and hasattr(writer, "add_histogram"):
                    writer.add_histogram(f"{stage}/gates/{name}", gate, step)

    def _encode_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        labels_hidden: Optional[torch.Tensor],
        attn_maps: Optional[list] = None,
        average_attn_weights: bool = False,
    ) -> torch.Tensor:
        # forward de l'encodeur avec passage dans les blocs de convolution et d'attention
        features = x
        collect = attn_maps is not None
        for i, block in enumerate(self.encoder_blocks):
            features = block(
                features,
                mask,
                labels_hidden,
                need_attn_weights=collect,
                average_attn_weights=average_attn_weights,
            )
            if collect and block.attention is not None: # on log
                attn_maps.append(
                    {
                        "name": f"enc_block_{i}",
                        "weights": block.attention.last_attn,
                        "kpm": block.attention.last_kpm,
                    }
                )
        if self.enc_post_attn is not None: # idem pour la dernière couche d'attention de l'encodeur
            features = self.enc_post_attn(
                features,
                mask,
                need_weights=collect,
                average_attn_weights=average_attn_weights,
            )
            if collect:
                attn_maps.append(
                    {
                        "name": "enc_post_attn",
                        "weights": self.enc_post_attn.last_attn,
                        "kpm": self.enc_post_attn.last_kpm,
                    }
                )
        return features

    def forward(self, batch, **kwargs):
        x = batch["x"]  # [B, M, T]
        mask = batch["mask"]  # [B, M]
        labels = batch.get("labels", None)  # [B, M, N_attrs]

        B, M, T = x.shape
        return_attn = kwargs.get("return_attn", False)
        average_attn_weights = kwargs.get("average_attn_weights", False)
        attn_maps = [] if return_attn else None

        if self.use_monkey_patch:
            x = self._apply_monkey_patch(x)

        # embedding : [B, M, T] -> [B, M, T, C] -> [B, C, M, T] format image
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)  # [B, C, M, T]
        mask_4d = mask.view(B, 1, M, 1).float()
        x_emb = x_emb * mask_4d

        # conditionnement des labels
        labels_hidden = self._encode_labels(labels, mask)

        # encodeur : [B, C, M, T] -> [B, C, M, T']
        features = self._encode_features(
            x_emb,
            mask,
            labels_hidden,
            attn_maps=attn_maps,
            average_attn_weights=average_attn_weights,
        )

        # features: [B, C, M, T'] -> [B, M, C, T'] -> [B, M, C*T']
        # pour que les fc s'appliquent par ménage, par membre
        features = features.permute(0, 2, 1, 3)
        features_flat = features.reshape(B, M, -1)

        # projections latentes (par membre)
        mu = self.fc_mu(features_flat)
        logvar = self.fc_var(features_flat)

        # Reparameterize
        z = self._reparameterize(mu, logvar)
        # décodeur
        log_probs = self._decode(
            z,
            mask,
            labels_hidden,
            attn_maps=attn_maps,
            average_attn_weights=average_attn_weights,
        )

        if return_attn:
            return [log_probs, mu, logvar, z, mask, attn_maps]
        return [log_probs, mu, logvar, z, mask]

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _decode(
        self,
        z,
        mask,
        labels_hidden=None,
        attn_maps: Optional[list] = None,
        average_attn_weights: bool = False,
    ):
        B, M, _ = z.shape

        # reprojette du latent vers les features
        dec_features = self.fc_decode(z)
        dec_features = dec_features.view(B, M, self.c_out, self.w_out)
        dec_features = dec_features.permute(0, 2, 1, 3)

        # applique 0 à toutes les features des membres absents
        mask_4d = mask.view(B, 1, M, 1).float()
        dec_features = dec_features * mask_4d
        dec_features = self.dec_latent_norm(dec_features)

        collect = attn_maps is not None
        for i, block in enumerate(self.decoder_blocks):
            dec_features = block(
                dec_features,
                mask,
                labels_hidden,
                need_attn_weights=collect,
                average_attn_weights=average_attn_weights,
            )
            if collect and block.attention is not None:
                attn_maps.append(
                    {
                        "name": f"dec_block_{i}",
                        "weights": block.attention.last_attn,
                        "kpm": block.attention.last_kpm,
                    }
                )

        if self.dec_post_attn is not None:
            dec_features = self.dec_post_attn(
                dec_features,
                mask,
                need_weights=collect,
                average_attn_weights=average_attn_weights,
            )
            if collect:
                attn_maps.append(
                    {
                        "name": "dec_post_attn",
                        "weights": self.dec_post_attn.last_attn,
                        "kpm": self.dec_post_attn.last_kpm,
                    }
                )

        # decodeur CNN
        dec_out = dec_features

        # Trim si on utilise l'extension temporelle
        if self.use_monkey_patch:
            dec_out = dec_out[:, :, :, : self.original_time_steps]

        # logits -> log probs
        log_probs = self.final_projection(dec_out)
        log_probs = log_probs.permute(0, 2, 3, 1)
        return log_probs

    def loss_function(self, args, batch, **kwargs):
        log_probs = args[0]
        mu = args[1]
        logvar = args[2]
        z = args[3]
        mask = args[4]
        x = batch["x"].long()

        # format image pour NLL: [B, M, T, C] -> [B, C, M, T]
        log_probs_for_loss = log_probs.permute(0, 3, 1, 2)
        # poids
        weight = getattr(self, "class_weights", None)
        # log probs vs targets
        recon_loss = F.nll_loss(
            log_probs_for_loss,
            x,
            weight=weight,
            ignore_index=0,
            reduction="none",
        ) # [B, M, T] un par pixel

        # mask les membres absents
        # expand pour avoir [B, M, T]
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()
        denom = mask_expanded.sum().clamp(min=1.0)  # normalisation pour avoir la loss par pixel actif
        recon_loss = (recon_loss * mask_expanded).sum() / denom

        # on a un point latent par membre
        # d'abord on calcule le KLD par membre
        # KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dim = -1 est la dimension latent_dim -> on calcule par dimension latente la divergence et on somme
        kld_per_member = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )
        # mais les membres fantômes ont aussi un point latent
        # on mask et normalise
        # ici on divise par le nombre de membres actifs pas par les pixels
        kld_masked = kld_per_member * mask
        kld_loss = kld_masked.sum() / mask.sum().clamp(min=1.0)

        kld_weight = kwargs.get("kld_weight", 0.01)
        total_loss = recon_loss + kld_weight * kld_loss

        return {"loss": total_loss, "recon_loss": recon_loss, "KLD": kld_loss, "z": z}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            # Case 1: Validation/Test 
            log_probs, mu, logvar, z, _ = self.forward(batch)
            labels = batch.get("labels", torch.zeros_like(batch["mask"]))
            return batch["x"], log_probs, z, labels, mu, logvar
        else:
            # Case 2: Generation soit avec soit sans attributs 
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                z, labels = batch
            else:
                z = batch
                labels = None
            # problème si z est une liste de tenseurs et pas un seul tenseur

            if isinstance(z, list):
                z = (
                    torch.stack(z)
                    if all(isinstance(t, torch.Tensor) for t in z)
                    else torch.tensor(z)
                )
            if isinstance(labels, list) and labels and isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)

            B = z.shape[0]
            M = self.max_members

            # on doit construire le mask de génération via la colonne household_size
            if (
                labels is not None
                and self.hh_size_col_idx is not None
                and self.hh_size_map_tensor is not None
            ):
                # retrouve la taille réelle du ménage via le premier membre
                encoded_sizes = labels[:, 0, self.hh_size_col_idx].long()
                # mappe vers la taille réelle
                real_sizes = self.hh_size_map_tensor[encoded_sizes]
                # pour chaque membre on regarde si son index est < taille réelle pour savoir si c'est un fantome
                mask = (
                    torch.arange(M, device=z.device).expand(B, -1)
                    < real_sizes.unsqueeze(1)
                ).float()
            else:
                mask = torch.ones(B, M, device=z.device)

            # encode les labels pour le conditionnement du latent et décodeur
            labels_hidden = self._encode_labels(labels, mask)

            # decode
            log_probs = self._decode(z, mask, labels_hidden)

            # force génération à PAD pour les membres inactifs
            # et inversement interdit PAD pour les membres actifs
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs).bool()
            active_log_probs = log_probs.clone()
            active_log_probs[..., 0] = -1e9
            inactive_log_probs = torch.full_like(log_probs, -1e9)
            inactive_log_probs[..., 0] = 0.0

            final_log_probs = torch.where(mask_expanded, active_log_probs, inactive_log_probs)

            if labels is None:
                labels = torch.zeros(B, M, dtype=torch.long, device=z.device)

            return labels, final_log_probs, z

    def training_step(self, batch, batch_idx):
        if self._should_log_attention():
            outputs = self.forward(
                batch,
                return_attn=True,
                average_attn_weights=False,
            )
            attn_maps = outputs[-1]
            outputs = outputs[:-1]
            self._log_attention_and_gates(attn_maps, stage="train")
        else:
            outputs = self.forward(batch)

        if self._should_log_gates() and not self._should_log_attention():
            self._log_attention_and_gates(attn_maps=None, stage="train")
        kld_w = getattr(self, "scheduled_kld_weight", getattr(self, "kld_loss_weight", 0.01))
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)

        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("train_kld", loss_dict["KLD"], on_step=False, on_epoch=True)
        self.log("used_kld_weight", kld_w, on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        if self._should_log_attention() and batch_idx == 0:
            outputs = self.forward(
                batch,
                return_attn=True,
                average_attn_weights=False,
            )
            attn_maps = outputs[-1]
            outputs = outputs[:-1]
            self._log_attention_and_gates(attn_maps, stage="val")
        else:
            outputs = self.forward(batch)

        if self._should_log_gates() and not self._should_log_attention() and batch_idx == 0:
            self._log_attention_and_gates(attn_maps=None, stage="val")
        kld_w = getattr(self, "scheduled_kld_weight", getattr(self, "kld_loss_weight", 0.01))
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)

        self.log("val_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        kld_w = getattr(self, "scheduled_kld_weight", getattr(self, "kld_loss_weight", 0.01))
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        self.log("test_loss", loss_dict["loss"])
        return loss_dict["loss"]

    def on_train_epoch_start(self):
        # permet de modifier le poids KLD au début de chaque epoch
        if self.kld_annealing_type:
            epoch = self.current_epoch
            max_weight = (
                self.kld_max_weight if self.kld_max_weight is not None else self.kld_loss_weight
            )
            current_beta = max_weight
            # version linéaire
            if self.kld_annealing_type == "monotonic":
                if epoch < self.kld_annealing_epochs:
                    progress = epoch / self.kld_annealing_epochs
                    current_beta = self.kld_start_weight + progress * (
                        max_weight - self.kld_start_weight
                    )
                else:
                    current_beta = max_weight
            # vesion cyclique
            elif self.kld_annealing_type == "cyclical":
                total_epochs = self.trainer.max_epochs if self.trainer else 100
                # nombre de cycles voulus
                M = self.kld_n_cycles
                # çalcul de la longueur d'un cycle en epochs
                cycle_len = total_epochs / M

                # avancement du cycle [0,1]
                tau = ((epoch) % math.ceil(cycle_len)) / cycle_len
                # combien de temps ça monte / combien de temps ça reste à plat
                R = self.kld_cycle_ratio

                if tau <= R:
                    # on doit monter de façon linéaire
                    ratio = tau / R
                    current_beta = ratio * max_weight
                else:
                    # sinon plateau
                    current_beta = max_weight

            # active le poids calculé         
            self.scheduled_kld_weight = current_beta
            self.log(
                "kld_weight",
                self.scheduled_kld_weight,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def on_validation_end(self):
        pass
