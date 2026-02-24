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
        # labels_dim sort de label encoder
        # on projette vers la dimension des canaux pour moduler
        # il faut que ce soit de même taille que les canaux de features
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
        # c'est à dire que l'effet des attributs ne change pas en fonction du pas de temps
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
        # c'est l'intégration la moins chere
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


class DeltaProjection(nn.Module):
    """à partir des features, prédit delta_mu et delta_logvar qui sont les corrections
    utilisé pour les deux latents
    pour le z_coord il s'agit des mu et logvar classique
    pour le z_indiv ce sont des corrections ajoutés à la conditionnalité du latent coord"""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        # MLP simple: input_dim -> hidden_dim -> 2 * latent_dim
        # Le facteur 2 correspond à (mu, logvar).
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),     
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, features_flat: torch.Tensor):
        """
        [B, M, input_dim] ou poolé [B, input_dim]
        return delta_mu et delta_logvar
        """
        h = self.net(features_flat)
        delta_mu, delta_logvar = h.chunk(2, dim=-1)
        # stabilite tomczak: borne logvar a [-7, 2] pour éviter des variances extrêmes
        delta_logvar = F.hardtanh(delta_logvar, -7.0, 2.0)
        return delta_mu, delta_logvar


class ConditionalPriorNet(nn.Module):
    """Paramètre le prior conditionnel de z_indiv à partir de z_coord:
    q(z_indiv|x,z_coord) = N(mu_prior(z_coord)+delta_mu, exp(logvar_prior(z_coord)+delta_logvar))"""
    def __init__(self, coord_dim: int, indiv_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),     
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * indiv_dim),
        )
        self.indiv_dim = indiv_dim

    def forward(self, z_coord: torch.Tensor):
        """
        z_coord: [B, M, coord_dim] ou [B, coord_dim]
        """
        h = self.net(z_coord)
        mu_prior, logvar_prior = h.chunk(2, dim=-1)
        return mu_prior, logvar_prior

def time_sin_cos(W, device):
    t = torch.arange(W, device=device)
    frac = t.float() / W
    sin_feat = torch.sin(2 * math.pi * frac)
    cos_feat = torch.cos(2 * math.pi * frac)
    return torch.stack([sin_feat, cos_feat], dim=-1)
class AttentionBlockPairwiseFullDay(nn.Module):
    """
    chaque membre i interroge sa sequence temporelle complete et lit la
    sequence complete des autres membres j != i.
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
        use_gate: bool = True,
    ):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})."
            )
        # correspond à une linear pour atteindre la dimension d'attention
        # la dimension 512 parait trop elevee donc on reduit
        self.down = nn.Conv2d(channels + 2, attn_dim, kernel_size=1, bias=False)

        # idem pour remonter à la dimension originale après attention
        self.up = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)

        # MHA appliquee sur des sequences temporelles entieres (W tokens)
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout_attn,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.drop_attn = nn.Dropout(dropout_resid) if dropout_resid > 0 else nn.Identity()

        # feedforward pour traiter les tokens apres attention
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, ff_mult * attn_dim),
            nn.GELU(),
            nn.Dropout(dropout_ff),
            nn.Linear(ff_mult * attn_dim, attn_dim),
            nn.Dropout(dropout_ff),
        )
        # gate pour moduler l'importance de l'attention (== de chaque channels)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = nn.Parameter(torch.full((channels,), float(gate_init)))
            self.register_buffer("gate_fixed", torch.ones(channels))
        else:
            self.gate = None
            self.register_buffer("gate_fixed", torch.ones(channels))
        self.last_attn = None # pour log
        self.last_kpm = None # pour log 

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_4d: Optional[torch.Tensor] = None,
        kpm_base: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> torch.Tensor:
        B, C, M, W = x.shape
        if mask_4d is None:
            mask_4d = mask.bool().view(B, 1, M, 1).float()
        if kpm_base is None:
            kpm_base = ~mask.bool()
        time_feat = time_sin_cos(W, x.device)
        tf = time_feat.transpose(0, 1).view(1, 2, 1, W).expand(B, -1, M, -1) # [B, 2, M, W]
        x_with_time = torch.cat([x, tf], dim=1) # [B, C+2, M, W]
        y = self.down(x_with_time)  # [B, d, M, W]
        d = y.shape[1]
        tokens = y.permute(0, 2, 3, 1).contiguous()  # [B, M, W, d]
        mask_bool = ~kpm_base

        # torch.arrange(M) crée la liste de base [0,1,2...], unsqueeze passe en colonne, et repeat la copie M fois, flatten mets tout en ligne 
        # ça donne idx_i = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2...]
        # c'est un peu "qui regarde" -> les 5 premières pairs le  membre 0 regarde etc
        idx_i = torch.arange(M, device=x.device).unsqueeze(1).repeat(1, M).flatten()
        # idx_j : liste de base qui se répète en ligne -> [0,1,2,3,4,5, 0,1,2,3,4,5]
        # == qui est regardé
        # donc idx_i et idx_j crée les pairs element par element
        idx_j = torch.arange(M, device=x.device).repeat(M)
        # tokens = [B, M, W, d]
        # avec idx_i indexing ça donne [B, M*M, W, d] car on prend plusieurs fois le schedule de chaque membre
        # reshape pour avoir la forme attendue par MHA -> on fait l'attention sur la séquence du temps
        q = self.norm1(tokens[:, idx_i, :, :]).reshape(B * M * M, W, d)
        # idem pour j
        kv = self.norm1(tokens[:, idx_j, :, :]).reshape(B * M * M, W, d)

        # l'attention crée une matrice de scores [W,W]
        # i regarde toute la journée de j à chaque instant
        if need_weights:
            attn_out, _ = self.mha(
                q,
                kv,
                kv,
                need_weights=False,
                average_attn_weights=average_attn_weights,
            )
            self.last_attn = None
            self.last_kpm = None
        else:
            attn_out, _ = self.mha(q, kv, kv, need_weights=False)
            self.last_attn = None
            self.last_kpm = None
        # dimension 0 = B * M * M de scores d'interaction
        # on structure en ménage, membre i, membre j
        attn_out = self.drop_attn(attn_out).view(B, M, M, W, d)
        # on filtre l'attention qui nous intéresse
        pair_mask = (
            mask_bool[:, idx_i] # i doit exister
            & mask_bool[:, idx_j] # j aussi
            & (idx_i != idx_j) # pas de self
        ).view(B, M, M, 1, 1).float()

        msg_sum = (attn_out * pair_mask).sum(dim=2) # sommme sur l'axe des j
        pair_count = pair_mask.sum(dim=2) # combien de j valides
        has_pairs = (pair_count > 0).float() 
        tokens = msg_sum / pair_count.clamp(min=1.0) # moyenne par pairs valides
        h = self.norm2(tokens) 
        tokens = (tokens + self.ff(h)) * has_pairs # ajoute l'attention uniquement pour les familles

        y = tokens.permute(0, 3, 1, 2).contiguous()
        # on remonte
        y = self.up(y)  # [B, C, M, W]
        # gates
        if self.use_gate and self.gate is not None:
            alpha = torch.sigmoid(self.gate).view(1, -1, 1, 1)
        else:
            alpha = self.gate_fixed.view(1, -1, 1, 1)
        out = (x + alpha * y) * mask_4d
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
            self.attention = AttentionBlockPairwiseFullDay(
                channels=in_channels,
                attn_dim=attn_cfg["attn_dim"],
                num_heads=attn_cfg["num_heads"],
                ff_mult=attn_cfg["ff_mult"],
                dropout_attn=attn_cfg["dropout_attn"],
                dropout_ff=attn_cfg["dropout_ff"],
                dropout_resid=attn_cfg["dropout_resid"],
                gate_init=attn_cfg["gate_init"],
                use_gate=attn_cfg["use_gate"],
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
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        labels_hidden: Optional[torch.Tensor],
        mask_4d: Optional[torch.Tensor] = None,
        kpm_base: Optional[torch.Tensor] = None,
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

        if mask_4d is None:
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
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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
            self.attention = AttentionBlockPairwiseFullDay(
                channels=in_channels,
                attn_dim=attn_cfg["attn_dim"],
                num_heads=attn_cfg["num_heads"],
                ff_mult=attn_cfg["ff_mult"],
                dropout_attn=attn_cfg["dropout_attn"],
                dropout_ff=attn_cfg["dropout_ff"],
                dropout_resid=attn_cfg["dropout_resid"],
                gate_init=attn_cfg["gate_init"],
                use_gate=attn_cfg["use_gate"],
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
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU() if not is_last else None
        self.drop = nn.Dropout(dropout) if not is_last else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        labels_hidden: Optional[torch.Tensor],
        mask_4d: Optional[torch.Tensor] = None,
        kpm_base: Optional[torch.Tensor] = None,
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

        if mask_4d is None:
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
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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

class HouseholdDiscCNNCVAEFullAttentionInteractions(Base):
    def __init__(self, encoding_weights=None, *args, **kwargs):
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        if encoding_weights is not None:
            self.register_buffer("class_weights", encoding_weights)
        else:
            self.class_weights = None

        # kld annealing
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

        # poids specifique pour KL_coord (mode hierarchique)
        self.kld_coord_weight = kwargs.get("kld_coord_weight", None)
        self.kld_coord_annealing_type = kwargs.get("kld_coord_annealing_type", None)
        self.kld_coord_annealing_epochs = kwargs.get("kld_coord_annealing_epochs", 100)
        self.kld_coord_start_weight = kwargs.get("kld_coord_start_weight", 0.0)
        default_coord_weight = (
            self.kld_coord_weight
            if self.kld_coord_weight is not None
            else kwargs.get("kld_weight", 0.01)
        )
        self.kld_coord_max_weight = kwargs.get("kld_coord_max_weight", default_coord_weight)
        self.scheduled_kld_coord_weight = self.kld_coord_weight
        if self.kld_coord_annealing_type:
            print(
                "KLD_coord Annealing: "
                f"{self.kld_coord_annealing_type}, Max={self.kld_coord_max_weight}"
            )
            self.scheduled_kld_coord_weight = self.kld_coord_start_weight

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
        self.interaction_emb_dim = int(config.get("interaction_emb_dim", 8))
        self.social_loss_weight = float(config.get("social_loss_weight", 1.0))
        self.interaction_input_dropout = float(config.get("interaction_input_dropout", 0.3))
        self.social_inter_weight = float(config.get("social_inter_weight", 1.0))
        self.social_same_weight = float(config.get("social_same_weight", 1.0))
        self.auto_social_inter_pos_weight = bool(
            config.get("auto_social_inter_pos_weight", False)
        )
        self.social_inter_pos_weight = float(config.get("social_inter_pos_weight", 1.0))

        # configuration vae hierarchique (tomczak)
        self.use_hierarchical = config.get("use_hierarchical", False)
        # dim du latent ménage
        self.coord_latent_dim = config.get("coord_latent_dim", 4)
        # dim du latent indiv
        self.indiv_latent_dim = config.get("indiv_latent_dim", 8)
         # "2d" = un latent par membre, "1d" = un latent poolé
        self.coord_mode = config.get("coord_mode", "2d")
        # features -> hidden -> latent
        self.coord_prior_hidden = config.get("coord_prior_hidden", 64)
        self.hierarchical_delta_hidden = config.get("hierarchical_delta_hidden", 512)

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

        # determiner le padding mode
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

        # label encoder = comment on cree le vecteur d'attributs a partir de l'encoding brut
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

        # configuration attention
        self.attn_dim = config.get("attn_dim", 64)
        self.attn_heads = config.get("attn_heads", 2)
        self.attn_ff_mult = config.get("attn_ff_mult", 2)
        self.attn_dropout_attn = config.get("attn_dropout_attn", 0.1)
        self.attn_dropout_ff = config.get("attn_dropout_ff", 0.1)
        self.attn_dropout_resid = config.get("attn_dropout_resid", 0.0)
        self.attn_gate_init = config.get("attn_gate_init", -4.0)
        self.attn_use_gates = bool(config.get("attn_use_gates", True))
        self.attn_cfg = {
            "attn_dim": self.attn_dim,
            "num_heads": self.attn_heads,
            "ff_mult": self.attn_ff_mult,
            "dropout_attn": self.attn_dropout_attn,
            "dropout_ff": self.attn_dropout_ff,
            "dropout_resid": self.attn_dropout_resid,
            "gate_init": self.attn_gate_init,
            "use_gate": self.attn_use_gates,
        }

        # où l'activer ? c'est à dire à quelles couches de l'encodeur et du décodeur 
        self.enc_attn_indices = config.get("enc_attn_indices", [1])
        self.dec_attn_indices = config.get("dec_attn_indices", [0])
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
        print(f"[CVAE-Attn] Gates enabled: {self.attn_use_gates}")
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

        # embedding interactions
        self.interaction_embedding = nn.Embedding(2, self.interaction_emb_dim)

        # fusion des features d'activité et d'interaction
        self.interaction_fuse = nn.Conv2d(
            self.hidden_size + self.interaction_emb_dim,
            self.hidden_size,
            kernel_size=1,
            bias=False,
        )

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

        self.dec_latent_norm = nn.GroupNorm(1, self.c_out)

        if self.use_hierarchical:
            # couches vae hierarchiques 
            # chemin bottom-up: deux tetes de projection delta separees depuis les features encodeur
            self.delta_indiv_proj = DeltaProjection(
                self.flat_per_member, self.indiv_latent_dim, self.hierarchical_delta_hidden
            )
            self.delta_coord_proj = DeltaProjection(
                self.flat_per_member, self.coord_latent_dim, self.hierarchical_delta_hidden
            )
            # chemin top-down: prior conditionnel 
            self.cond_prior_net = ConditionalPriorNet(
                self.coord_latent_dim, self.indiv_latent_dim, self.coord_prior_hidden
            )
            # le decodeur projette depuis z_indiv uniquement 
            self.fc_decode = nn.Linear(self.indiv_latent_dim, self.c_out * self.w_out)
            # pour compatibilité runners
            self.latent_dim = self.coord_latent_dim
        else:
            # couches vae plates d'origine
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

        # projette vers le logit d'interaction
        self.interaction_head = nn.Conv2d(self.hidden_size, 1, kernel_size=1)

        if self.use_hierarchical:
            print(f"[HVAE] Hierarchical VAE enabled:")
            print(f"  z_coord: dim={self.coord_latent_dim}, mode={self.coord_mode}")
            print(f"  z_indiv: dim={self.indiv_latent_dim} (conditioned on z_coord)")
            print(f"  Hierarchical latent: z_coord=[B, {'M, ' if self.coord_mode == '2d' else ''}{self.coord_latent_dim}], "
                  f"z_indiv=[B, M, {self.indiv_latent_dim}]")
        else:
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
            post_attn = AttentionBlockPairwiseFullDay(
                channels=self.hidden_size,
                attn_dim=self.attn_cfg["attn_dim"],
                num_heads=self.attn_cfg["num_heads"],
                ff_mult=self.attn_cfg["ff_mult"],
                dropout_attn=self.attn_cfg["dropout_attn"],
                dropout_ff=self.attn_cfg["dropout_ff"],
                dropout_resid=self.attn_cfg["dropout_resid"],
                gate_init=self.attn_cfg["gate_init"],
                use_gate=self.attn_cfg["use_gate"],
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
            # convtranspose: out = (in - 1) * stride - 2*padding + kernel + output_padding
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
            post_attn = AttentionBlockPairwiseFullDay(
                channels=self.hidden_size,
                attn_dim=self.attn_cfg["attn_dim"],
                num_heads=self.attn_cfg["num_heads"],
                ff_mult=self.attn_cfg["ff_mult"],
                dropout_attn=self.attn_cfg["dropout_attn"],
                dropout_ff=self.attn_cfg["dropout_ff"],
                dropout_resid=self.attn_cfg["dropout_resid"],
                gate_init=self.attn_cfg["gate_init"],
                use_gate=self.attn_cfg["use_gate"],
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

    def _apply_interaction_patch(self, interactions: torch.Tensor) -> torch.Tensor:
        """étend la séquence temporelle des interactions avec des zéros"""
        B, M, T = interactions.shape
        extra = self.monkey_patch_extra_steps
        zero_padding = torch.zeros(
            (B, M, extra),
            dtype=interactions.dtype,
            device=interactions.device,
        )
        return torch.cat([interactions, zero_padding], dim=2)

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
        # generateur utilitaire pour parcourir tous les blocs d'attention actifs du modele
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
        # calcule des statistiques sur les poids d'attention
        if weights is None or kpm is None:
            return None
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)
        if weights.dim() != 4:
            return None

        BW, H, M, _ = weights.shape
        if kpm.dim() != 2 or kpm.shape[0] != BW or kpm.shape[1] != M:
            return None
        # utilise ~kpm car kpm=True signifie "ignore", donc active=True signifie "membre reel"
        active = ~kpm
        n_active = active.sum(dim=1)  # [BW] nombre de membres actifs par batch item
        active_any = n_active > 0
        # stats basiques sur la composition des menages
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
        # masquage pour ne garder que les membres réels dans le calcul des stats d'attention
        key_mask = active.view(BW, 1, 1, M)
        w = weights * key_mask
        # calcule les metriques par tete et par membre
        diag = w.diagonal(dim1=-2, dim2=-1)  # [BW, H, M] self attention
        rowmax = w.max(dim=-1).values # pic d'attention
        entropy = -(w * (w.clamp_min(eps).log())).sum(dim=-1) # dispersion de l'attention
        offdiag = 1.0 - diag # attention portée aux autres
        # moyennes ponderees
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
        # moyenne globale sur toutes les tetes
        mean = {k: v.mean().item() for k, v in per_head.items()}

        return {
            "active_members_mean": active_members_mean,
            "frac_size1": frac_size1,
            "per_head": per_head,
            "mean": mean,
        }

    def _log_attention_and_gates(self, attn_maps, stage: str):
        # envoie les stats, histogrammes et images vers tensorboard.
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
                if getattr(block, "use_gate", True) and getattr(block, "gate", None) is not None:
                    gate = torch.sigmoid(block.gate.detach())
                else:
                    gate = block.gate_fixed.detach()
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
        mask_4d: Optional[torch.Tensor] = None,
        kpm_base: Optional[torch.Tensor] = None,
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
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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
        interactions = batch.get("interactions", None)  # [B, M, T]
        B, M, T = x.shape
        return_attn = kwargs.get("return_attn", False)
        average_attn_weights = kwargs.get("average_attn_weights", False)
        attn_maps = [] if return_attn else None

        if self.use_monkey_patch:
            x = self._apply_monkey_patch(x)
            interactions = self._apply_interaction_patch(interactions)

        if self.training and self.interaction_input_dropout > 0.0:
            keep = torch.rand_like(interactions.float()) > self.interaction_input_dropout
            interactions = interactions * keep.to(interactions.dtype)

        # embedding : [B, M, T] -> [B, M, T, C] -> [B, C, M, T] format image
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)  # [B, C, M, T]
        interaction_emb = self.interaction_embedding(interactions.long()).permute(0, 3, 1, 2)
        x_emb = torch.cat([x_emb, interaction_emb], dim=1)
        x_emb = self.interaction_fuse(x_emb)
        # pre-calcule les tenseurs de masque une fois pour tout le forward
        mask_4d = mask.view(B, 1, M, 1).float()
        kpm_base = ~mask.bool()  # [B, M] True=phantom, for attention key_padding_mask
        x_emb = x_emb * mask_4d

        # conditionnement des labels
        labels_hidden = self._encode_labels(labels, mask)

        # encodeur : [B, C, M, T] -> [B, C, M, T']
        features = self._encode_features(
            x_emb,
            mask,
            labels_hidden,
            mask_4d=mask_4d,
            kpm_base=kpm_base,
            attn_maps=attn_maps,
            average_attn_weights=average_attn_weights,
        )

        # features: [B, C, M, T'] -> [B, M, C, T'] -> [B, M, C*T']
        # pour que les fc s'appliquent par ménage, par membre
        features = features.permute(0, 2, 1, 3)
        features_flat = features.reshape(B, M, -1)

        if self.use_hierarchical:
            # vae hierarchique (tomczak)
            # parametres delta pour z_indiv 
            delta_mu_indiv, delta_logvar_indiv = self.delta_indiv_proj(features_flat)
            # forme: [B, M, indiv_latent_dim]

            # parametres delta pour z_coord 
            if self.coord_mode == "2d":
                # latent de coordination par membre
                delta_mu_coord, delta_logvar_coord = self.delta_coord_proj(features_flat)
                # forme: [B, M, coord_latent_dim]
            else:
                # mode 1d: pooling moyenne masquee sur les membres 
                mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
                pooled_features = (features_flat * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
                # forme: [B, flat_per_member]
                delta_mu_coord, delta_logvar_coord = self.delta_coord_proj(
                    pooled_features.unsqueeze(1)
                )
                delta_mu_coord = delta_mu_coord.squeeze(1)      # [B, coord_latent_dim]
                delta_logvar_coord = delta_logvar_coord.squeeze(1)

            # echantillonne z_coord depuis les delta
            z_coord = self._reparameterize(delta_mu_coord, delta_logvar_coord)

            # calcule le prior conditionnel p(z_indiv | z_coord)
            if self.coord_mode == "1d":
                # etend z_coord de [B, coord_dim] vers [B, M, coord_dim] car on a un z_indiv en format 2d toujours
                # donc en gros on duplique le z_coord par ménage
                z_coord_for_prior = z_coord.unsqueeze(1).expand(-1, M, -1)
            else:
                #en 2d déjà la bonne forme 
                z_coord_for_prior = z_coord
            mu_prior_indiv, logvar_prior_indiv = self.cond_prior_net(z_coord_for_prior)
            # forme: [B, M, indiv_latent_dim]

            # échantillone z_indiv depuis la correction delta issu des données et le prior
            mu_indiv = mu_prior_indiv + delta_mu_indiv
            logvar_indiv = logvar_prior_indiv + delta_logvar_indiv
            z_indiv = self._reparameterize(mu_indiv, logvar_indiv)
            # forme: [B, M, indiv_latent_dim]

            # decode depuis z_indiv 
            log_probs, interaction_logits = self._decode(
                z_indiv, mask, labels_hidden,
                mask_4d=mask_4d, kpm_base=kpm_base,
                attn_maps=attn_maps, average_attn_weights=average_attn_weights,
            )

            if return_attn:
                return [log_probs, interaction_logits,
                        delta_mu_coord, delta_logvar_coord,
                        delta_mu_indiv, delta_logvar_indiv,
                        mu_prior_indiv, logvar_prior_indiv,
                        z_coord, z_indiv, mask, attn_maps]
            return [log_probs, interaction_logits,
                    delta_mu_coord, delta_logvar_coord,
                    delta_mu_indiv, delta_logvar_indiv,
                    mu_prior_indiv, logvar_prior_indiv,
                    z_coord, z_indiv, mask]
        else:
            # vae d'origine
            # projections latentes (par membre)
            mu = self.fc_mu(features_flat)
            logvar = self.fc_var(features_flat)

            # reparameterisation
            z = self._reparameterize(mu, logvar)
            # décodeur
            log_probs, interaction_logits = self._decode(
                z,
                mask,
                labels_hidden,
                mask_4d=mask_4d,
                kpm_base=kpm_base,
                attn_maps=attn_maps,
                average_attn_weights=average_attn_weights,
            )

            if return_attn:
                return [log_probs, interaction_logits, mu, logvar, z, mask, attn_maps]
            return [log_probs, interaction_logits, mu, logvar, z, mask]

    def _reparameterize(self, mu, logvar):
        # on garde le sampling stochastique du latent en train ET en eval/test,
        # c'est le comportement standard de CAVEAT (pas de mode déterministe)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(
        self,
        z,
        mask,
        labels_hidden=None,
        mask_4d: Optional[torch.Tensor] = None,
        kpm_base: Optional[torch.Tensor] = None,
        attn_maps: Optional[list] = None,
        average_attn_weights: bool = False,
    ):
        B, M, _ = z.shape

        # reprojette du latent vers les features
        dec_features = self.fc_decode(z)
        dec_features = dec_features.view(B, M, self.c_out, self.w_out)
        dec_features = dec_features.permute(0, 2, 1, 3)

        # applique 0 à toutes les features des membres absents
        if mask_4d is None:
            mask_4d = mask.view(B, 1, M, 1).float()
        dec_features = dec_features * mask_4d
        dec_features = self.dec_latent_norm(dec_features)

        collect = attn_maps is not None
        for i, block in enumerate(self.decoder_blocks):
            dec_features = block(
                dec_features,
                mask,
                labels_hidden,
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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
                mask_4d=mask_4d,
                kpm_base=kpm_base,
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

        # trim si on utilise l'extension temporelle
        if self.use_monkey_patch:
            dec_out = dec_out[:, :, :, : self.original_time_steps]

        interaction_logits = self.interaction_head(dec_out).squeeze(1)  # [B, M, T]

        # logits -> log probs
        log_probs = self.final_projection(dec_out)
        log_probs = log_probs.permute(0, 2, 3, 1)
        return log_probs, interaction_logits

    def loss_function(self, args, batch, **kwargs):
        if self.use_hierarchical:
            return self._hierarchical_loss_function(args, batch, **kwargs)
        return self._flat_loss_function(args, batch, **kwargs)

    def _compute_social_loss_full(self, interaction_logits, log_probs, batch, x_device, mask):
        """loss sociale"""
        social_weight = getattr(self, "social_loss_weight", 0.0)
        # initialize les loss
        social_loss = torch.tensor(0.0, device=x_device)
        social_inter_loss = torch.tensor(0.0, device=x_device)
        social_same_loss = torch.tensor(0.0, device=x_device)
        num_pairs_total = torch.tensor(0.0, device=x_device)
        num_pairs_valid = torch.tensor(0.0, device=x_device)
        num_pairs_pos = torch.tensor(0.0, device=x_device)

        # calcul les poids de classe d'interactions
        social_inter_pos_weight_t = torch.tensor(
            [self.social_inter_pos_weight], device=x_device
        )
        # == il y en a dans le batch
        interactions_present = batch.get("interactions_present", True)
        if isinstance(interactions_present, torch.Tensor):
            interactions_present = bool(interactions_present.any().item())
        # imagine binaire = target
        interactions = batch.get("interactions", None)
        # les pairs d'interactions
        interaction_pairs = batch.get("interaction_pairs", None)
        social_active = (
            social_weight > 0.0
            and interactions_present
            and interaction_logits is not None
            and interactions is not None
        )
        if social_active:
            # loss interaction par membre (binaire)
            # prend en compte les membres fantomes via le mask
            # et normalise sur tous les slots actifs (B * M_actifs * T)
            mask_3d = (
                mask.unsqueeze(-1).expand_as(interaction_logits).float()
            )
            pos_weight = torch.tensor(
                self.social_inter_pos_weight,
                device=x_device,
                dtype=interaction_logits.dtype,
            )
            social_inter_loss = F.binary_cross_entropy_with_logits(
                interaction_logits,
                interactions.float(),
                reduction="none",
                pos_weight=pos_weight,
            )
            social_inter_loss = (social_inter_loss * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

            # Must-Link
            if interaction_pairs is not None and interaction_pairs.numel() > 0:
                pairs = interaction_pairs.to(x_device, non_blocking=True)
                b = pairs[:, 0].long() # indice du ménage
                m1 = pairs[:, 1].long() # membre 1
                m2 = pairs[:, 2].long()# membre 2 qui interagit avec membre 1
                t = pairs[:, 3].long() # time step

                probs = log_probs.exp() 
                q_a = probs[b, m1, t, :] # proba du membre 1 au time step t
                q_b = probs[b, m2, t, :] # idem pour membre 2
                p_same = (q_a * q_b).sum(dim=-1).clamp(min=1e-8) # sous independance proba que l'un et l'autre interagissent
                social_same_loss = (-torch.log(p_same)).mean() # NLL 

                n_pairs = float(pairs.shape[0])
                num_pairs_total = torch.tensor(n_pairs, device=x_device)
                num_pairs_valid = torch.tensor(n_pairs, device=x_device)
                num_pairs_pos = torch.tensor(n_pairs, device=x_device)
            # applique les poids de loss
            social_loss = (
                self.social_inter_weight * social_inter_loss
                + self.social_same_weight * social_same_loss
            )

        return {
            "social_active": social_active,
            "social_weight": social_weight,
            "social_loss": social_loss,
            "social_inter_loss": social_inter_loss,
            "social_same_loss": social_same_loss,
            "num_pairs_total": num_pairs_total,
            "num_pairs_valid": num_pairs_valid,
            "num_pairs_pos": num_pairs_pos,
            "social_inter_pos_weight": social_inter_pos_weight_t,
        }

    def _compute_recon_loss(self, log_probs, x, mask):
        """loss de reconstruction"""
        log_probs_for_loss = log_probs.permute(0, 3, 1, 2)
        weight = getattr(self, "class_weights", None)
        recon_loss = F.nll_loss(
            log_probs_for_loss, x, weight=weight, ignore_index=0, reduction="none",
        )
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()
        denom = mask_expanded.sum().clamp(min=1.0)
        return (recon_loss * mask_expanded).sum() / denom

    def _flat_loss_function(self, args, batch, **kwargs):
        """loss d'origine"""
        log_probs = args[0]
        interaction_logits = args[1]
        mu = args[2]
        logvar = args[3]
        z = args[4]
        mask = args[5]
        x = batch["x"].long()

        recon_loss = self._compute_recon_loss(log_probs, x, mask)

        # kld par membre (standard: contre N(0,I))
        kld_per_member = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )
        kld_masked = kld_per_member * mask
        kld_loss = kld_masked.sum() / mask.sum().clamp(min=1.0)

        kld_weight = kwargs.get("kld_weight", 0.01)
        total_loss = recon_loss + kld_weight * kld_loss

        # perte sociale
        social = self._compute_social_loss_full(
            interaction_logits, log_probs, batch, x.device, mask
        )
        if social["social_active"] and social["social_weight"] > 0.0:
            total_loss = total_loss + social["social_weight"] * social["social_loss"]

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "KLD": kld_loss,
            "social_loss": social["social_loss"],
            "social_inter_loss": social["social_inter_loss"],
            "social_same_loss": social["social_same_loss"],
            "num_pairs_total": social["num_pairs_total"],
            "num_pairs_valid": social["num_pairs_valid"],
            "num_pairs_pos": social["num_pairs_pos"],
            "social_weight": torch.tensor([social["social_weight"]], device=x.device),
            "social_inter_weight": torch.tensor([self.social_inter_weight], device=x.device),
            "social_same_weight": torch.tensor([self.social_same_weight], device=x.device),
            "social_inter_pos_weight": social["social_inter_pos_weight"],
            "z": z,
        }

    def _hierarchical_loss_function(self, args, batch, **kwargs):
        """ KL_coord + KL_indiv + social + recons"""
        log_probs = args[0]
        interaction_logits = args[1]
        delta_mu_coord = args[2]
        delta_logvar_coord = args[3]
        delta_mu_indiv = args[4]
        delta_logvar_indiv = args[5]
        mu_prior_indiv = args[6]
        logvar_prior_indiv = args[7]
        z_coord = args[8]
        z_indiv = args[9]
        mask = args[10]
        x = batch["x"].long()

        recon_loss = self._compute_recon_loss(log_probs, x, mask)

        # kl pour z_coord: KL(q(z_coord|x) ||  N(0, I)) == KL classique
        if self.coord_mode == "2d":
            # forme: [B, M, coord_dim]
            kld_coord_per_member = -0.5 * torch.sum(
                1 + delta_logvar_coord - delta_mu_coord.pow(2) - delta_logvar_coord.exp(),
                dim=-1,
            )  # [B, M]
            kld_coord = (kld_coord_per_member * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            # forme: [B, coord_dim] - par menage 
            kld_coord = -0.5 * torch.sum(
                1 + delta_logvar_coord - delta_mu_coord.pow(2) - delta_logvar_coord.exp(),
                dim=-1,
            ).mean() # [B,1]

        # kl pour z_indiv: KL(q(z_indiv|x,z_coord) || p(z_indiv|z_coord))
        # formule tomczak du kl entre deux gaussiennes ou:
        # q a pour moyenne = mu_prior + delta_mu, logvar = logvar_prior + delta_logvar
        # p a pour moyenne = mu_prior, logvar = logvar_prior
        # kl = 0.5 * sum(delta_mu^2/exp(logvar_prior) + exp(delta_logvar) - delta_logvar - 1)
        kld_indiv_per_member = 0.5 * torch.sum(
            delta_mu_indiv.pow(2) / logvar_prior_indiv.exp().clamp(min=1e-8)
            + delta_logvar_indiv.exp()
            - delta_logvar_indiv
            - 1,
            dim=-1,
        )  # [B, M]
        kld_indiv = (kld_indiv_per_member * mask).sum() / mask.sum().clamp(min=1.0) # pour masque

        kld_weight = kwargs.get("kld_weight", 0.01)
        kld_coord_weight = kwargs.get("kld_coord_weight", kld_weight)
        weighted_kld_coord = kld_coord_weight * kld_coord
        weighted_kld_indiv = kld_weight * kld_indiv
        total_loss = recon_loss + weighted_kld_coord + weighted_kld_indiv
        # kl total non ponderee pour monitoring
        kld_loss = kld_coord + kld_indiv

        # perte sociale 
        social = self._compute_social_loss_full(
            interaction_logits, log_probs, batch, x.device, mask
        )
        if social["social_active"] and social["social_weight"] > 0.0:
            total_loss = total_loss + social["social_weight"] * social["social_loss"]

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "KLD": kld_loss,
            "KLD_coord": kld_coord.detach(),
            "KLD_indiv": kld_indiv.detach(),
            "social_loss": social["social_loss"],
            "social_inter_loss": social["social_inter_loss"],
            "social_same_loss": social["social_same_loss"],
            "num_pairs_total": social["num_pairs_total"],
            "num_pairs_valid": social["num_pairs_valid"],
            "num_pairs_pos": social["num_pairs_pos"],
            "social_weight": torch.tensor([social["social_weight"]], device=x.device),
            "social_inter_weight": torch.tensor([self.social_inter_weight], device=x.device),
            "social_same_weight": torch.tensor([self.social_same_weight], device=x.device),
            "social_inter_pos_weight": social["social_inter_pos_weight"],
            "z": z_indiv,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict) and "x" in batch:
            # cas 1: validation/test
            # si hierarchique on renvoie aussi les stats des deux priors en faisant passer le batch dans le fwd
            if self.use_hierarchical:
                outputs = self.forward(batch)
                log_probs = outputs[0]
                interaction_logits = outputs[1]
                mu_coord = outputs[2]
                logvar_coord = outputs[3]
                delta_mu_indiv = outputs[4]
                delta_logvar_indiv = outputs[5]
                mu_prior_indiv = outputs[6]
                logvar_prior_indiv = outputs[7]
                z_coord = outputs[8]
                z_indiv = outputs[9]
                mask = outputs[10]
                # recompose via delta + prior
                mu_indiv = mu_prior_indiv + delta_mu_indiv
                logvar_indiv = logvar_prior_indiv + delta_logvar_indiv
                labels = batch.get("labels", torch.zeros_like(batch["mask"]))
                interactions = batch.get("interactions", None)
                if interactions is None:
                    interactions = torch.zeros_like(batch["x"])
                return (
                    batch["x"],
                    log_probs,
                    z_indiv,
                    labels,
                    mu_coord,
                    logvar_coord,
                    interaction_logits,
                    interactions,
                    mask,
                    z_coord,
                    mu_indiv,
                    logvar_indiv,
                )
            # sinon qu'un seul latent 
            else:
                log_probs, interaction_logits, mu, logvar, z, mask = self.forward(batch)
                labels = batch.get("labels", torch.zeros_like(batch["mask"]))
                interactions = batch.get("interactions", None)
                if interactions is None:
                    interactions = torch.zeros_like(batch["x"])
                return (
                    batch["x"],
                    log_probs,
                    z,
                    labels,
                    mu,
                    logvar,
                    interaction_logits,
                    interactions,
                    mask,
                )
        else:
            # cas 2: generation avec ou sans attributs
            z_indiv_override = None
            if isinstance(batch, dict):
                if "z_indiv" in batch:
                    z_indiv_override = batch["z_indiv"]
                    z = None
                    labels = batch.get("labels", None)
                elif "z" in batch:
                    z = batch["z"]
                    labels = batch.get("labels", None)
                else:
                    raise ValueError(
                        "Unexpected generation batch dict format. Expected keys 'z_indiv' or 'z'."
                    )
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                z, labels = batch
            else:
                # sans 
                z = batch
                labels = None
            # bug de liste
            if isinstance(z, list):
                z = (
                    torch.stack(z)
                    if all(isinstance(t, torch.Tensor) for t in z)
                    else torch.tensor(z)
                )
            # bug de liste
            if isinstance(labels, list) and labels and isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)
            if isinstance(z_indiv_override, list):
                z_indiv_override = (
                    torch.stack(z_indiv_override)
                    if all(isinstance(t, torch.Tensor) for t in z_indiv_override)
                    else torch.tensor(z_indiv_override)
                )

            latent_ref = z_indiv_override if z_indiv_override is not None else z
            if latent_ref is None:
                raise ValueError("Generation batch did not provide a latent tensor.")

            B = latent_ref.shape[0]
            M = self.max_members
            device_ref = latent_ref.device

            # construire le mask de generation via la colonne household_size
            if (
                labels is not None
                and self.hh_size_col_idx is not None
                and self.hh_size_map_tensor is not None
            ):
                encoded_sizes = labels[:, 0, self.hh_size_col_idx].long()
                real_sizes = self.hh_size_map_tensor[encoded_sizes]
                mask = (
                    torch.arange(M, device=device_ref).expand(B, -1)
                    < real_sizes.unsqueeze(1)
                ).float()
            else:
                # sinon on génére 5 membres par ménage
                mask = torch.ones(B, M, device=device_ref)
            # encode et projette les attributs
            labels_hidden = self._encode_labels(labels, mask)

            if self.use_hierarchical:
                if z_indiv_override is not None:
                    # prior GMM : on sample z_indiv directement depuis le GMM fitté sur
                    # le vrai posterior — ça bypass la chaîne z_coord → cond_prior.
                    # sert à tester la capacité du decoder avec un prior plus riche
                    if z_indiv_override.ndim != 3:
                        raise ValueError(
                            f"Expected z_indiv override with shape [B, M, D], got {tuple(z_indiv_override.shape)}"
                        )
                    if z_indiv_override.shape[1] != M:
                        raise ValueError(
                            f"Expected z_indiv override second dim M={M}, got {z_indiv_override.shape[1]}"
                        )
                    if z_indiv_override.shape[2] != self.indiv_latent_dim:
                        raise ValueError(
                            f"Expected z_indiv override last dim {self.indiv_latent_dim}, "
                            f"got {z_indiv_override.shape[2]}"
                        )
                    z_indiv = z_indiv_override.to(device_ref)
                    z_return = z_indiv
                else:
                    # sampling hiérarchique standard : z == z_coord tiré de N(0,I)
                    if z is None:
                        raise ValueError("Hierarchical generation requires z or z_indiv.")
                    z_coord = z

                    # gere les formes de z en fonction du coord_mode
                    if self.coord_mode == "2d":
                        # au cas où mais ça doit pas arrivé
                        if z_coord.ndim == 2:
                            # forme: [B, D] -> [B, M, D] (etend a tous les membres)
                            z_coord = z_coord.unsqueeze(1).expand(-1, M, -1)
                        z_coord_for_prior = z_coord
                    else:  # "1d"
                        if z_coord.ndim == 3:
                            # au cas où mais ça doit pas arrivé
                            z_coord = z_coord[:, 0, :]
                        z_coord_for_prior = z_coord.unsqueeze(1).expand(-1, M, -1)
                    # récupère les stats prior pour sampler z_indiv
                    mu_prior, logvar_prior = self.cond_prior_net(z_coord_for_prior)

                    # echantillonne z_indiv depuis le prior conditionnel
                    # en mode génération
                    std = torch.exp(0.5 * logvar_prior)
                    z_indiv = mu_prior + std * torch.randn_like(std)
                    z_return = z

                log_probs, interaction_logits = self._decode(z_indiv, mask, labels_hidden)
            else:
                if z_indiv_override is not None:
                    raise ValueError("z_indiv override is only supported for hierarchical models.")
                log_probs, interaction_logits = self._decode(z, mask, labels_hidden)
                z_return = z

            # force pad pour les membres inactifs et interdit pad pour les actifs
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs).bool()
            active_log_probs = log_probs.clone()
            active_log_probs[..., 0] = -1e9
            inactive_log_probs = torch.full_like(log_probs, -1e9)
            inactive_log_probs[..., 0] = 0.0

            final_log_probs = torch.where(mask_expanded, active_log_probs, inactive_log_probs)

            if labels is None:
                labels = torch.zeros(B, M, dtype=torch.long, device=device_ref)

            return labels, final_log_probs, z_return, interaction_logits, mask

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
        kld_coord_w = getattr(self, "scheduled_kld_coord_weight", None)
        if kld_coord_w is None:
            kld_coord_w = kld_w
        loss_dict = self.loss_function(
            outputs, batch, kld_weight=kld_w, kld_coord_weight=kld_coord_w
        )

        bs = batch["mask"].shape[0]
        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_kld", loss_dict["KLD"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social", loss_dict["social_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_inter", loss_dict["social_inter_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_same", loss_dict["social_same_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pairs_total", loss_dict["num_pairs_total"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pairs_valid", loss_dict["num_pairs_valid"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pairs_pos", loss_dict["num_pairs_pos"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_weight", loss_dict["social_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_inter_weight", loss_dict["social_inter_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_same_weight", loss_dict["social_same_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_social_inter_pos_weight", loss_dict["social_inter_pos_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("used_kld_weight", kld_w, on_step=False, on_epoch=True, batch_size=bs)
        if self.use_hierarchical:
            self.log("used_kld_coord_weight", kld_coord_w, on_step=False, on_epoch=True, batch_size=bs)
        if self.use_hierarchical:
            self.log("train_kld_coord", loss_dict["KLD_coord"], on_step=False, on_epoch=True, batch_size=bs)
            self.log("train_kld_indiv", loss_dict["KLD_indiv"], on_step=False, on_epoch=True, batch_size=bs)
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
        kld_coord_w = getattr(self, "scheduled_kld_coord_weight", None)
        if kld_coord_w is None:
            kld_coord_w = kld_w
        loss_dict = self.loss_function(
            outputs, batch, kld_weight=kld_w, kld_coord_weight=kld_coord_w
        )

        bs = batch["mask"].shape[0]
        self.log("val_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social", loss_dict["social_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social_inter", loss_dict["social_inter_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social_same", loss_dict["social_same_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pairs_total", loss_dict["num_pairs_total"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pairs_valid", loss_dict["num_pairs_valid"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pairs_pos", loss_dict["num_pairs_pos"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social_inter_weight", loss_dict["social_inter_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social_same_weight", loss_dict["social_same_weight"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_social_inter_pos_weight", loss_dict["social_inter_pos_weight"], on_step=False, on_epoch=True, batch_size=bs)
        if self.use_hierarchical:
            self.log("val_kld_coord", loss_dict["KLD_coord"], on_step=False, on_epoch=True, batch_size=bs)
            self.log("val_kld_indiv", loss_dict["KLD_indiv"], on_step=False, on_epoch=True, batch_size=bs)

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        kld_w = getattr(self, "scheduled_kld_weight", getattr(self, "kld_loss_weight", 0.01))
        kld_coord_w = getattr(self, "scheduled_kld_coord_weight", None)
        if kld_coord_w is None:
            kld_coord_w = kld_w
        loss_dict = self.loss_function(
            outputs, batch, kld_weight=kld_w, kld_coord_weight=kld_coord_w
        )
        bs = batch["mask"].shape[0]
        self.log("test_loss", loss_dict["loss"], batch_size=bs)
        self.log("test_social", loss_dict["social_loss"], batch_size=bs)
        self.log("test_social_inter", loss_dict["social_inter_loss"], batch_size=bs)
        self.log("test_social_same", loss_dict["social_same_loss"], batch_size=bs)
        self.log("test_pairs_total", loss_dict["num_pairs_total"], batch_size=bs)
        self.log("test_pairs_valid", loss_dict["num_pairs_valid"], batch_size=bs)
        self.log("test_pairs_pos", loss_dict["num_pairs_pos"], batch_size=bs)
        self.log("test_social_inter_weight", loss_dict["social_inter_weight"], batch_size=bs)
        self.log("test_social_same_weight", loss_dict["social_same_weight"], batch_size=bs)
        self.log("test_social_inter_pos_weight", loss_dict["social_inter_pos_weight"], batch_size=bs)
        if self.use_hierarchical:
            self.log("test_kld_coord", loss_dict["KLD_coord"], batch_size=bs)
            self.log("test_kld_indiv", loss_dict["KLD_indiv"], batch_size=bs)
        return loss_dict["loss"]

    def _compute_social_inter_pos_weight_from_train(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        pos_count = 0.0
        total_count = 0.0
        found = False

        train_dataset = getattr(datamodule, "train_dataset", None)
        base_dataset = getattr(train_dataset, "dataset", train_dataset)
        subset_indices = getattr(train_dataset, "indices", None)
        base_interactions = getattr(base_dataset, "interaction_tensor", None)
        base_mask = getattr(base_dataset, "mask_tensor", None)

        if isinstance(base_interactions, torch.Tensor) and isinstance(base_mask, torch.Tensor):
            interactions = base_interactions
            mask = base_mask
            if subset_indices is not None:
                idx = torch.as_tensor(subset_indices, dtype=torch.long, device=interactions.device)
                interactions = interactions.index_select(0, idx)
                mask = mask.index_select(0, idx)
            if interactions.ndim == 3 and mask.ndim == 2 and interactions.shape[:2] == mask.shape:
                active = mask.bool().unsqueeze(-1).expand_as(interactions)
                pos_count = float(((interactions > 0) & active).sum().item())
                total_count = float(active.sum().item())
                found = True

        if not found:
            loader = datamodule.train_dataloader()
            if isinstance(loader, dict):
                loader = next(iter(loader.values()), None)
            if isinstance(loader, (list, tuple)):
                loader = loader[0] if loader else None
            if loader is None:
                return None

            with torch.no_grad():
                for batch in loader:
                    if not isinstance(batch, dict):
                        continue
                    interactions = batch.get("interactions", None)
                    mask = batch.get("mask", None)
                    if interactions is None or mask is None:
                        continue
                    if interactions.ndim != 3 or mask.ndim != 2:
                        continue
                    if interactions.shape[:2] != mask.shape:
                        continue
                    active = mask.bool().unsqueeze(-1).expand_as(interactions)
                    pos_count += float(((interactions > 0) & active).sum().item())
                    total_count += float(active.sum().item())
                    found = True

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            counts = torch.tensor(
                [pos_count, total_count], device=self.device, dtype=torch.float64
            )
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            pos_count = float(counts[0].item())
            total_count = float(counts[1].item())

        if (not found) or total_count <= 0.0:
            return None

        neg_count = total_count - pos_count
        if pos_count <= 0.0 or neg_count <= 0.0:
            return 1.0

        return neg_count / pos_count

    def on_fit_start(self):
        if not getattr(self, "auto_social_inter_pos_weight", False):
            return
        computed = self._compute_social_inter_pos_weight_from_train()
        if computed is None:
            self.print(
                "[SocialLoss] auto_social_inter_pos_weight=True but unable to compute from train data. "
                f"Keeping configured value={self.social_inter_pos_weight:.6f}"
            )
            return
        self.social_inter_pos_weight = float(computed)
        self.print(
            f"[SocialLoss] Computed masked train pos_weight={self.social_inter_pos_weight:.6f} "
            "(neg/pos over active member-time slots)."
        )

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
        # annealing monotonic pour KL_coord uniquement
        if self.use_hierarchical:
            coord_weight = self.kld_coord_weight
            if self.kld_coord_annealing_type:
                epoch = self.current_epoch
                max_weight = (
                    self.kld_coord_max_weight
                    if self.kld_coord_max_weight is not None
                    else getattr(self, "kld_loss_weight", 0.01)
                )
                if self.kld_coord_annealing_type == "monotonic":
                    if epoch < self.kld_coord_annealing_epochs:
                        progress = epoch / self.kld_coord_annealing_epochs
                        coord_weight = self.kld_coord_start_weight + progress * (
                            max_weight - self.kld_coord_start_weight
                        )
                    else:
                        coord_weight = max_weight
                else:
                    coord_weight = max_weight
            self.scheduled_kld_coord_weight = coord_weight
            if self.scheduled_kld_coord_weight is not None:
                self.log(
                    "kld_coord_weight",
                    self.scheduled_kld_coord_weight,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def on_validation_end(self):
        pass

