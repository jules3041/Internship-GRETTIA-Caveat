

import torch
from torch import nn
import torch.nn.functional as F
import math
from caveat.models import Base
from typing import List, Optional


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
        """
        x: [B, C, M, T]
        labels_hidden: [B, M, labels_dim]
        """
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


class HouseholdDiscCNNCVAE(Base):

    def __init__(self, encoding_weights=None, *args, **kwargs):
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        if encoding_weights is not None:
            self.register_buffer('class_weights', encoding_weights)
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
        self.max_members = int(config.get('max_members', 5))
        self.time_steps = int(config.get('time_steps', 144))

        # patch ajout temporel
        # mais on prefere le padding
        self.use_monkey_patch = config.get("use_monkey_patch", False)
        self.monkey_patch_extra_steps = config.get("monkey_patch_extra_steps", 24)
        self.home_activity_index = config.get("home_activity_index", 0)

        self.original_time_steps = self.time_steps
        if self.use_monkey_patch:
            self.extended_time_steps = self.time_steps + self.monkey_patch_extra_steps
            print(f"Monkey patch enabled: {self.time_steps} -> {self.extended_time_steps} steps")
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
            self.register_buffer("hh_size_map_tensor", torch.tensor(hh_size_map, dtype=torch.long))
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
        if self.latent_conditionality is None:
            self.latent_conditionality = (
                self.conditioning_type if config.get("latent_conditioning", False) else "none"
            )
        if self.decoder_conditionality is None:
            self.decoder_conditionality = (
                self.conditioning_type if config.get("decoder_conditioning", True) else "none"
            )
        # flags booléens pour la suite
        self.encoder_conditioning = self.encoder_conditionality != "none"
        self.latent_conditioning = self.latent_conditionality != "none"
        self.decoder_conditioning = self.decoder_conditionality != "none"

        print(
            f"[CVAE] Conditioning: type={self.conditioning_type}, "
            f"encoder={self.encoder_conditionality}, latent={self.latent_conditionality}, "
            f"decoder={self.decoder_conditionality}, label_mode={self.label_encoder_mode}"
        )


        # embedding activité
        self.embedding = nn.Embedding(self.num_activities, self.hidden_size, padding_idx=0)

        # crée l'encodeur CNN
        self.encoder_cnn = self._build_encoder()

        # calculer dynamiquement la taille post encodeur
        with torch.no_grad():
            dummy = torch.zeros(1, self.hidden_size, self.max_members, self.extended_time_steps)
            enc_out = self.encoder_cnn(dummy)
            self.c_out = enc_out.shape[1]
            self.h_out = enc_out.shape[2]
            self.w_out = enc_out.shape[3]
            print(f"Encoder output shape: [{self.c_out}, {self.h_out}, {self.w_out}]")
            print(f"Time compression: {self.extended_time_steps} -> {self.w_out}")

        self.flat_per_member = self.c_out * self.w_out

        # Coordination blocks
        self.use_coordination = config.get("use_coordination", False)
        self.n_coord_blocks = config.get("n_coord_blocks", 2)
        self.coord_kernel_size = tuple(config.get("coord_kernel_size", [3, 1]))
        self.coord_dilation = config.get("coord_dilation", 1)
        self.use_global_mix = config.get("use_global_mix", True)
        self.use_local_mix = config.get("use_local_mix", True)

        # coordination. toujours c_out en entrée (car juste avant/après latent)
        # dilation pour padding vertical
        if self.use_coordination:
            print(
                f"Coordination enabled: {self.n_coord_blocks} blocks, kernel={self.coord_kernel_size}, "
                f"dilation={self.coord_dilation})"
            )

            self.enc_coord_blocks = nn.ModuleList(
                [
                    CoordinationBlock(
                        channels=self.c_out,
                        dilation=self.coord_dilation,
                        use_global_mix=self.use_global_mix,
                        use_local_mix=self.use_local_mix,
                        kernel_size=self.coord_kernel_size,
                    )
                    for _ in range(self.n_coord_blocks)
                ]
            )
            self.dec_coord_blocks = nn.ModuleList(
                [
                    CoordinationBlock(
                        channels=self.c_out,
                        dilation=self.coord_dilation,
                        use_global_mix=self.use_global_mix,
                        use_local_mix=self.use_local_mix,
                        kernel_size=self.coord_kernel_size,
                    )
                    for _ in range(self.n_coord_blocks)
                ]
            )
        else:
            self.enc_coord_blocks = nn.ModuleList([])
            self.dec_coord_blocks = nn.ModuleList([])


        # crée les conditionneurs
        # un encodeur, un latent, un décodeur
        if self.use_conditioning and self.encoder_conditioning:
            if self.encoder_conditionality == "film":
                self.enc_conditioner = FiLMConditioner(
                    self.labels_condition_dim, self.c_out
                )
            elif self.encoder_conditionality == "add":
                self.enc_conditioner = AddConditioner(
                    self.labels_condition_dim, self.c_out
                )
            elif self.encoder_conditionality == "concat":
                self.enc_conditioner = ConcatConditioner(
                    self.labels_condition_dim, self.conditioning_hidden_size
                )
                # si concat on doit augmenter la taille pré et post latent
                self.flat_per_member = (self.c_out + self.conditioning_hidden_size) * self.w_out
            else:
                raise ValueError(f"Unknown encoder_conditionality: {self.encoder_conditionality}")
        else:
            self.enc_conditioner = None

        # Latent conditioning
        if self.use_conditioning and self.latent_conditioning:
            if self.latent_conditionality == "concat":
                self.latent_conditioner = LatentConcatConditioner(
                    self.labels_condition_dim, self.conditioning_hidden_size
                )
                # si concat ça augmente la taille. important pour les projections
                self.effective_latent_dim = self.latent_dim + self.conditioning_hidden_size

            elif self.latent_conditionality in {"add", "film"}:  # film -> on utilise add à la place
                # add = translation -> décale le point généré vers la région des latents correspondant aux attributs
                # alors que film = scaling -> peut déformer le latent et le rendre non-gaussien
                self.latent_conditioner = LatentAddConditioner(
                    self.labels_condition_dim, self.latent_dim
                )
                self.effective_latent_dim = self.latent_dim
            else:
                raise ValueError(f"Unknown latent_conditionality: {self.latent_conditionality}")
        else:
            self.latent_conditioner = None
            self.effective_latent_dim = self.latent_dim

        # decoder conditioning
        if self.use_conditioning and self.decoder_conditioning:
            if self.decoder_conditionality == "film":
                self.dec_conditioner = FiLMConditioner(
                    self.labels_condition_dim, self.c_out
                )
            elif self.decoder_conditionality == "add":
                self.dec_conditioner = AddConditioner(
                    self.labels_condition_dim, self.c_out
                )
            elif self.decoder_conditionality == "concat":
                self.dec_conditioner = ConcatConditioner(
                    self.labels_condition_dim, self.conditioning_hidden_size
                )
            else:
                raise ValueError(f"Unknown decoder_conditionality: {self.decoder_conditionality}")
        else:
            self.dec_conditioner = None

        # projections latentes par membre (INDEPENDANTES)
        self.fc_mu = nn.Linear(self.flat_per_member, self.latent_dim)
        self.fc_var = nn.Linear(self.flat_per_member, self.latent_dim)
        # on reprojette du latent vers l'espace des features
        self.fc_decode = nn.Linear(self.effective_latent_dim, self.c_out * self.w_out)

        # decodeur CNN
        self.decoder_cnn = self._build_decoder()

        # pour concat on doit agrandir l'input du décodeur 
        if self.use_conditioning and self.decoder_conditioning and self.decoder_conditionality == "concat":
            self.decoder_cnn = self._build_decoder(
                input_channels=self.c_out + self.conditioning_hidden_size
            )

        # projette vers les logits puis log probs
        self.final_projection = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.num_activities, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

        print(f"Model built: latent_dim={self.latent_dim}, hidden_size={self.hidden_size}")
        print(f"2D Latent shape: [B, {self.max_members}, {self.latent_dim}]")

    def _build_encoder(self):
        # construit l'encodeur CNN
        # suffit de spécifier le nombre de couches et les paramètres
        layers = []
        in_channels = self.hidden_size

        for i in range(self.num_layers):
            # toujours hidden_size en sortie (512 -> 512 -> ...)
            out_channels = self.hidden_size

            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                padding_mode=self.padding_mode,
                bias=False
            ))
            # normalisation par (M,T)
            layers.append(ChannelwiseLayerNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            # mise à jour du nombre de canaux (ça sert à rien ici mais bon)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_decoder(self, input_channels=None):
        layers = []

        # calcul le chemin de la taille temporelle dans l'encodeur
        # en gros on veut calculer T' pour remonter à T
        sizes = [self.extended_time_steps]
        size = self.extended_time_steps
        for _ in range(self.num_layers):
            size = (size + 2*self.padding - self.kernel_size) // self.stride + 1
            sizes.append(size)
        sizes.reverse()  # inverse pour le decodeur

        # gère le concat
        in_channels = input_channels if input_channels else self.hidden_size

        for i in range(self.num_layers):
            out_channels = self.hidden_size
            target_size = sizes[i + 1]
            current_size = sizes[i]
            # on calcule l'output padding nécessaire pour atteindre la taille cible
            # ConvTranspose: out = (in - 1) * stride - 2*padding + kernel + output_padding
            # padding = 1 ici
            expected = (current_size - 1) * self.stride - 2*self.padding + self.kernel_size
            output_padding = max(0, target_size - expected) # calcule la différence pour l'output padding

            layers.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                output_padding=(0, output_padding),
                bias=False
            ))
            layers.append(ChannelwiseLayerNorm2d(out_channels))
            # pas d'activation ni dropout à la dernière couche (éviter les biais et le bruit)
            if i < self.num_layers - 1:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(self.dropout_rate))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _apply_monkey_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Étend la séquence temporelle avec des steps supplémentaires (activité home)"""
        B, M, T = x.shape
        extra = self.monkey_patch_extra_steps

        home_padding = torch.full(
            (B, M, extra),
            self.home_activity_index,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, home_padding], dim=2)

    def _encode_labels(self, labels: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
        # active le forward de LabelEncoder
        if labels is None or not self.use_conditioning or self.label_encoder is None:
            return None
        labels_hidden = self.label_encoder(labels.long())
        # on pousse l'embedding à 0 pour les membres absents
        if mask is not None:
            labels_hidden = labels_hidden * mask.unsqueeze(-1).float()
        return labels_hidden

    def forward(self, batch, **kwargs):
        x = batch["x"]  # [B, M, T]
        mask = batch["mask"]  # [B, M]
        labels = batch.get("labels", None)  # [B, M, N_attrs]

        B, M, T = x.shape

       
        if self.use_monkey_patch:
            x = self._apply_monkey_patch(x)

        # embedding : [B, M, T] -> [B, M, T, C] -> [B, C, M, T] format image
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)  # [B, C, M, T]
        mask_4d = mask.view(B, 1, M, 1).float()
        x_emb = x_emb * mask_4d

        # encodeur : [B, C, M, T] -> [B, C, M, T']
        features = self.encoder_cnn(x_emb)  # [B, C, M, T']

        # Coordination blocks dans l'encodeur
        for block in self.enc_coord_blocks:
            features = block(features, mask)
        # conditionnement des labels
        labels_hidden = self._encode_labels(labels, mask)

        # conditionning à la fin de l'encodeur
        if self.enc_conditioner is not None and labels_hidden is not None:
            features = self.enc_conditioner(features, labels_hidden)

        # features: [B, C, M, T'] -> [B, M, C, T'] -> [B, M, C*T']
        # pour que les fc s'appliquent par ménage, par membre
        features = features.permute(0, 2, 1, 3)  # [B, M, C, T']
        features_flat = features.reshape(B, M, -1)  # [B, M, flat_per_member]

        # projections latentes (par membre)
        mu = self.fc_mu(features_flat)  # [B, M, latent_dim]
        logvar = self.fc_var(features_flat)

        # Reparameterize
        z = self._reparameterize(mu, logvar)

        # Latent conditioning
        if self.latent_conditioner is not None and labels_hidden is not None:
            z_cond = self.latent_conditioner(z, labels_hidden)
        else:
            z_cond = z

        # decode
        log_probs = self._decode(z_cond, mask, labels_hidden)

        return [log_probs, mu, logvar, z, mask]

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _decode(self, z, mask, labels_hidden=None):
        B, M, _ = z.shape

        # reprojette du latent vers les features
        dec_features = self.fc_decode(z)  # [B, M, c_out*T']
        dec_features = dec_features.view(B, M, self.c_out, self.w_out)
        dec_features = dec_features.permute(0, 2, 1, 3)  # [B, C, M, T']

        # applique 0 à toutes les features des membres absents
        mask_4d = mask.view(B, 1, M, 1).float()
        dec_features = dec_features * mask_4d

        # Coordination blocks dans le décodeur
        for block in self.dec_coord_blocks:
            dec_features = block(dec_features, mask)

        # Decoder conditioning
        if self.dec_conditioner is not None and labels_hidden is not None:
            dec_features = self.dec_conditioner(dec_features, labels_hidden)

        # decodeur CNN
        dec_out = self.decoder_cnn(dec_features)  # [B, C, M, T]

        # Trim si on utilise l'extension temporelle
        if self.use_monkey_patch:
            dec_out = dec_out[:, :, :, :self.original_time_steps]

        # logits -> log probs
        log_probs = self.final_projection(dec_out)  # [B, num_activities, M, T]
        log_probs = log_probs.permute(0, 2, 3, 1)  # [B, M, T, C]

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
            log_probs_for_loss, x,
            weight=weight,
            ignore_index=0, # Ignore PAD
            reduction='none'
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
        kld_per_member = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # mais les membres fantômes ont aussi un point latent
        # on mask et normalise
        # ici on divise par le nombre de membres actifs pas par les pixels
        kld_masked = kld_per_member * mask
        kld_loss = kld_masked.sum() / mask.sum().clamp(min=1.0)

        kld_weight = kwargs.get('kld_weight', 0.01)
        total_loss = recon_loss + kld_weight * kld_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "KLD": kld_loss,
            "z": z
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        # Case 1: Validation/Test 
        if isinstance(batch, dict):
            log_probs, mu, logvar, z, _ = self.forward(batch)
            labels = batch.get("labels", torch.zeros_like(batch["mask"]))
            return batch["x"], log_probs, z, labels, mu, logvar
        # Case 2: Generation soit avec soit sans attributs 
        else:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                z, labels = batch
            else:
                z = batch
                labels = None
            # problème si z est une liste de tenseurs et pas un seul tenseur

            if isinstance(z, list):
                z = torch.stack(z) if all(isinstance(t, torch.Tensor) for t in z) else torch.tensor(z)
            if isinstance(labels, list) and labels and isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)

            B = z.shape[0]
            M = self.max_members

            # on doit construire le mask de génération via la colonne household_size
            if labels is not None and self.hh_size_col_idx is not None and self.hh_size_map_tensor is not None:
                # retrouve la taille réelle du ménage via le premier membre
                encoded_sizes = labels[:, 0, self.hh_size_col_idx].long()
                # mappe vers la taille réelle
                real_sizes = self.hh_size_map_tensor[encoded_sizes]
                # pour chaque membre on regarde si son index est < taille réelle pour savoir si c'est un fantome
                mask = (torch.arange(M, device=z.device).expand(B, -1) < real_sizes.unsqueeze(1)).float()
            else:
                mask = torch.ones(B, M, device=z.device)
            # encode les labels pour le conditionnement du latent et décodeur
            labels_hidden = self._encode_labels(labels, mask)

            # conditionnement latent
            if self.latent_conditioner is not None and labels_hidden is not None:
                z_cond = self.latent_conditioner(z, labels_hidden)
            else:
                z_cond = z

            # decode
            log_probs = self._decode(z_cond, mask, labels_hidden)

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
        outputs = self.forward(batch)
        kld_w = getattr(self, "scheduled_kld_weight", getattr(self, "kld_loss_weight", 0.01))
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)

        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("train_kld", loss_dict["KLD"], on_step=False, on_epoch=True)
        self.log("used_kld_weight", kld_w, on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
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
            max_weight = self.kld_max_weight if self.kld_max_weight is not None else self.kld_loss_weight
            # version linéaire
            if self.kld_annealing_type == "monotonic":
                if epoch < self.kld_annealing_epochs:
                    progress = epoch / self.kld_annealing_epochs
                    self.scheduled_kld_weight = (
                        self.kld_start_weight + progress * (max_weight - self.kld_start_weight)
                    )
                else:
                    self.scheduled_kld_weight = max_weight
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

            self.log("kld_weight", self.scheduled_kld_weight, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_end(self):
        pass


class CoordinationBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation=1,
        use_global_mix=True,
        use_local_mix=True,
        kernel_size=(3, 1),
        global_mix_ratio=8,
    ):
        super().__init__()
        self.use_global_mix = use_global_mix
        self.use_local_mix = use_local_mix

        k_h, k_w = kernel_size
        pad_h = (k_h - 1) * dilation // 2
        pad_w = (k_w - 1) // 2

        if use_local_mix:
            self.local_mix = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=(dilation, 1),
                padding=(pad_h, pad_w),
                bias=False,
            )

        if self.use_global_mix:
            reduced_channels = max(1, channels // global_mix_ratio)
            self.global_mix = nn.Sequential(
                nn.Linear(channels, reduced_channels),
                nn.ReLU(),
                nn.Linear(reduced_channels, channels),
                nn.Sigmoid(),
            )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        B, C, M, W = x.shape
        mask_4d = mask.view(B, 1, M, 1).float()

        residual = x

        x = x * mask_4d
        if self.use_local_mix:
            out = self.local_mix(x)
        else:
            out = x

        if self.use_global_mix:
            sum_pooled = (out * mask_4d).sum(dim=(2, 3))
            denom = (mask.sum(dim=1) * W).clamp(min=1.0).view(B, 1)
            mean_pooled = sum_pooled / denom

            out_global = self.global_mix(mean_pooled).view(B, -1, 1, 1)
            out = out * out_global

        out = out + residual
        out = self.relu(self.norm(out))

        return out * mask_4d
