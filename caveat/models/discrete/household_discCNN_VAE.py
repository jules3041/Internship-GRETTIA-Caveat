

import torch
from torch import nn
import torch.nn.functional as F
from caveat.models import Base


class HouseholdDiscCNNVAE(Base):


    def __init__(self, encoding_weights=None, *args, **kwargs):
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        if encoding_weights is not None:
            self.register_buffer('class_weights', encoding_weights)
        else:
            self.class_weights = None

        self.kld_annealing_type = kwargs.get("kld_annealing_type", None) 
        self.kld_annealing_epochs = kwargs.get("kld_annealing_epochs", 100)
        # self.kld_annealing_cycles = kwargs.get("kld_annealing_cycles", 4) 
        self.kld_start_weight = kwargs.get("kld_start_weight", 0.0)
        self.kld_max_weight = kwargs.get("kld_max_weight", None) 
 
        if self.kld_annealing_type:
            print(f"KLD Annealing enabled: type={self.kld_annealing_type}, epochs={self.kld_annealing_epochs}")
            self.scheduled_kld_weight = self.kld_start_weight  
 
        self.automatic_optimization = True 
 

    def build(self, **config):

        self.latent_dim = config.get("latent_dim", 6)
        self.hidden_size = config.get("hidden_size", 512) 
        self.num_activities = int(self._encodings_backup or config.get("encodings"))
        self.max_members = int(config.get('max_members', 5))
        self.time_steps = int(config.get('time_steps', 144))

        # paramètres CNN
        self.kernel_size = config.get("kernel_size", 4)
        self.stride = config.get("stride", 2)
        self.padding = config.get("padding", 0)
        self.dropout = config.get("dropout", 0.1)
        self.num_layers = config.get("num_layers", 4)
        self.use_replicate_padding = config.get("use_replicate_padding", False)
        self.use_circular_padding = config.get("use_circular_padding", False)
        # Déterminer le padding mode
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

        # embedding activité
        # pas besoin des attributs car c'est un VAE simple
        self.embedding = nn.Embedding(self.num_activities, self.hidden_size, padding_idx=0)


        # crée l'encodeur CNN
        self.encoder_cnn = self._build_encoder()

        # calculer dynamiquement la taille post encodeur
        with torch.no_grad():
            dummy = torch.zeros(1, self.hidden_size, self.max_members, self.time_steps)
            enc_out = self.encoder_cnn(dummy)
            self.c_out = enc_out.shape[1]  # Channels
            self.h_out = enc_out.shape[2]  # Members (should stay same)
            self.w_out = enc_out.shape[3]  # Compressed time
            print(f"Encoder output shape: [{self.c_out}, {self.h_out}, {self.w_out}]")
            print(f"Time compression: {self.time_steps} -> {self.w_out}")

        # taille aplatie par membre après l'encodeur
        self.flat_per_member = self.c_out * self.w_out

        # Coordination blocks
        self.use_coordination = config.get("use_coordination", False)
        self.n_coord_blocks = config.get("n_coord_blocks", 2)
        self.coord_kernel_size = tuple(config.get("coord_kernel_size", [3, 1]))
        self.coord_dilation = config.get("coord_dilation", 1)
        self.use_global_mix = config.get("use_global_mix", True)
        self.use_local_mix = config.get("use_local_mix", True)

        if self.use_coordination:
            print(f"Coordination enabled: {self.n_coord_blocks} blocks, kernel={self.coord_kernel_size}, "
                  f"dilation={self.coord_dilation})")

            # Encodeur coordination
            self.enc_coord_blocks = nn.ModuleList([
                CoordinationBlock(
                    channels=self.c_out,
                    dilation=self.coord_dilation,
                    use_global_mix=self.use_global_mix,
                    use_local_mix=self.use_local_mix,
                    kernel_size=self.coord_kernel_size,
                ) for _ in range(self.n_coord_blocks)
            ])

            # Décodeur coordination
            self.dec_coord_blocks = nn.ModuleList([
                CoordinationBlock(
                    channels=self.c_out,
                    dilation=self.coord_dilation,
                    use_global_mix=self.use_global_mix,
                    use_local_mix=self.use_local_mix,
                    kernel_size=self.coord_kernel_size,
                ) for _ in range(self.n_coord_blocks)
            ])
        else:
            self.enc_coord_blocks = nn.ModuleList([])
            self.dec_coord_blocks = nn.ModuleList([])

        # projections latentes par membre (INDEPENDANTES)
        self.fc_mu = nn.Linear(self.flat_per_member, self.latent_dim)
        self.fc_var = nn.Linear(self.flat_per_member, self.latent_dim)

        # On reprojette du latent vers l'espace des features
        self.fc_decode = nn.Linear(self.latent_dim, self.flat_per_member)

        # decodeur CNN
        self.decoder_cnn = self._build_decoder()

        # projette vers les logits puis log probs
        self.final_projection = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.num_activities, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

        print(f"Model built: latent_dim={self.latent_dim}, hidden_size={self.hidden_size}")
        print(f"2D Latent shape: [B, {self.max_members}, {self.latent_dim}]")

    def _build_encoder(self):

        layers = []
        in_channels = self.hidden_size # toujours le même 

        # kernel=4, stride=2, padding=0: output = (input - 4) / 2 + 1 = (input - 2) / 2
        # 144 -> 71 -> 34 -> 16 -> 7 -> 2 (5 layers)

        for i in range(self.num_layers):
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
            layers.append(ChannelwiseLayerNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout))

            in_channels = out_channels # nécessaire pour la première couche

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []

        # calcul le chemin de la taille temporelle dans l'encodeur
        sizes = [self.time_steps]
        size = self.time_steps
        for _ in range(self.num_layers):
            size = (size + 2*self.padding - self.kernel_size) // self.stride + 1
            sizes.append(size) # 144, 71, 34, 16, 7, 2
        sizes.reverse()  # inverse pour le decodeur

        in_channels = self.hidden_size # toujours le même

        for i in range(self.num_layers):
            out_channels = self.hidden_size
            target_size = sizes[i + 1]
            current_size = sizes[i]

            # on calcule l'output padding nécessaire pour atteindre la taille cible
            # ConvTranspose: out = (in - 1) * stride - 2*padding + kernel + output_padding
            # padding = 0 ici
            expected = (current_size - 1) * self.stride - 2*self.padding + self.kernel_size

            output_padding = target_size - expected # calcule la différence pour l'output padding
            output_padding = max(0, output_padding) # doit être >= 0

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
                layers.append(nn.Dropout(self.dropout))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, batch, **kwargs):
        x = batch["x"]  # [B, M, T]
        mask = batch["mask"]  # [B, M]

        B, M, T = x.shape

        # embedding : [B, M, T] -> [B, M, T, C] -> [B, C, M, T] format image
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)
        mask_4d = mask.view(B, 1, M, 1).float()
        x_emb = x_emb * mask_4d
        # encodeur : [B, C, M, T] -> [B, C, M, T']
        features = self.encoder_cnn(x_emb)

        # Coordination blocks dans l'encodeur
        for block in self.enc_coord_blocks:
            features = block(features, mask)

        # features: [B, C, M, T'] -> [B, M, C, T'] -> [B, M, C*T']
        # pour que les fc s'appliquent par ménage, par membre
        features = features.permute(0, 2, 1, 3)  # [B, M, C, T']
        features_flat = features.reshape(B, M, -1)  # [B, M, flat_per_member]

        # projections latentes (par membre)
        mu = self.fc_mu(features_flat)  # [B, M, latent_dim]
        logvar = self.fc_var(features_flat)  # [B, M, latent_dim]

        # Reparameterize 
        z = self._reparameterize_2d(mu, logvar)  # [B, M, latent_dim]

        # decode
        log_probs = self._decode(z, mask)

        return [log_probs, mu, logvar, z, mask]

    def _reparameterize_2d(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z, mask):

        B, M, _ = z.shape

        # reprojette du latent vers les features
        dec_features = self.fc_decode(z)  # [B, M, flat_per_member]

        # reshape format image : [B, M, C*T'] -> [B, M, C, T'] -> [B, C, M, T']
        dec_features = dec_features.view(B, M, self.c_out, self.w_out)
        dec_features = dec_features.permute(0, 2, 1, 3)  # [B, C, M, T']

        # applique 0 à toutes les features des membres absents
        mask_4d = mask.view(B, 1, M, 1).float()
        dec_features = dec_features * mask_4d

        # Coordination blocks dans le décodeur
        for block in self.dec_coord_blocks:
            dec_features = block(dec_features, mask)

        # decodeur CNN
        dec_out = self.decoder_cnn(dec_features)  # [B, C, M, T]

        # logits -> log probs
        log_probs = self.final_projection(dec_out)  # [B, num_activities, M, T]
        log_probs = log_probs.permute(0, 2, 3, 1)

        return log_probs

    def loss_function(self, args, batch, **kwargs):
        log_probs = args[0]  # [B, M, T, Classes]
        mu = args[1]  # [B, M, latent_dim]
        logvar = args[2]  # [B, M, latent_dim]
        z = args[3]  # [B, M, latent_dim]
        mask = args[4]  # [B, M]
        x = batch["x"].long()  # [B, M, T]

        
        # format image pour NLL: [B, M, T, C] -> [B, C, M, T]
        log_probs_for_loss = log_probs.permute(0, 3, 1, 2)
        # poids
        weight = getattr(self, "class_weights", None)
        # log probs vs targets
        recon_loss = F.nll_loss(
            log_probs_for_loss,
            x,
            weight=weight,
            ignore_index=0,  # Ignore PAD
            reduction='none'
        )  # [B, M, T] un par pixel

        # mask les membres absents
        # expand pour avoir [B, M, T]
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()
        denom = mask_expanded.sum().clamp(min=1.0) # normalisation pour avoir la loss par pixel actif
        recon_loss = (recon_loss * mask_expanded).sum() / denom

        # on a un point latent par membre 
        # d'abord on calcule le KLD par membre
        # KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dim = -1 est la dimension latent_dim -> on calcule par dimension latente la divergence et on somme 
        kld_per_member = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, M] 

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
            outputs = self.forward(batch)
            log_probs = outputs[0]
            mu = outputs[1]     
            logvar = outputs[2]  
            z = outputs[3]
            
           
            return batch["x"], log_probs, z, batch.get("labels", torch.zeros_like(batch["mask"])), mu, logvar

        # Case 2: Generation soit avec soit sans attributs (juste descriptif pas CVAE)
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

            # z = [B, M, latent_dim] 

            # on doit construire le mask de génération via la colonne household_size
            if labels is not None and self.hh_size_col_idx is not None and self.hh_size_map_tensor is not None:
                encoded_sizes = labels[:, 0, self.hh_size_col_idx].long()

                real_sizes = self.hh_size_map_tensor[encoded_sizes]
                mask = (torch.arange(M, device=z.device).expand(B, -1) < real_sizes.unsqueeze(1)).float()
            else:
                mask = torch.ones(B, M, device=z.device)

            # decode
            log_probs = self._decode(z, mask)

            # force génération à PAD pour les membres inactifs
            # et inversement interdit PAD pour les membres actifs
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs).bool()
            active_log_probs = log_probs.clone()
            active_log_probs[..., 0] = -1e9
            inactive_log_probs = torch.full_like(log_probs, -1e9)
            inactive_log_probs[..., 0] = 0.0

            final_log_probs = torch.where(mask_expanded, active_log_probs, inactive_log_probs)

            # pour runners: (labels, log_probs, z)
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
        if self.kld_annealing_type:
            epoch = self.current_epoch
            max_weight = self.kld_max_weight if self.kld_max_weight is not None else self.kld_loss_weight
        
            if self.kld_annealing_type == "monotonic":
                if epoch < self.kld_annealing_epochs:
                    progress = epoch / self.kld_annealing_epochs
                    self.scheduled_kld_weight = self.kld_start_weight + progress * (max_weight - self.kld_start_weight)
                else:
                    self.scheduled_kld_weight = max_weight
            else:
                # fixed
                self.scheduled_kld_weight = max_weight
            # Log the KLD weight
            self.log("kld_weight", self.scheduled_kld_weight, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_end(self):
        pass

class ChannelwiseLayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, M, T] -> [B, M, T, C]
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # [B, M, T, C] -> [B, C, M, T]
        return x.permute(0, 3, 1, 2)




class CoordinationBlock(nn.Module):

    def __init__(self, channels, dilation=1, use_global_mix=True, use_local_mix=True,
                 kernel_size=(3, 1), global_mix_ratio=8,):
        super().__init__()
        self.use_global_mix = use_global_mix
        self.use_local_mix = use_local_mix

        # Calcul du padding pour garder la taille constante
        k_h, k_w = kernel_size
        pad_h = (k_h - 1) * dilation // 2
        pad_w = (k_w - 1) // 2

        if use_local_mix:

            self.local_mix = nn.Conv2d(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=(dilation, 1),
                    padding=(pad_h, pad_w),
                    bias=False
                )

        # Global mix 
        if self.use_global_mix:
            reduced_channels = max(1, channels // global_mix_ratio)
            self.global_mix = nn.Sequential(
                nn.Linear(channels, reduced_channels),
                nn.ReLU(),
                nn.Linear(reduced_channels, channels),
                nn.Sigmoid()
            )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):

        B, C, M, W = x.shape
        mask_4d = mask.view(B, 1, M, 1).float()

        # Connexion résiduelle
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