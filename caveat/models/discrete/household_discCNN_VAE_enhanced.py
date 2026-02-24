

import torch
from torch import nn
import torch.nn.functional as F
from caveat.models import Base



class HouseholdDiscCNNVAEEnhanced(Base):


    def __init__(self, encoding_weights=None, *args, **kwargs):
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        if encoding_weights is not None:
            self.register_buffer('class_weights', encoding_weights)
        else:
            self.class_weights = None

        self.kld_annealing_type = kwargs.get("kld_annealing_type", None)
        self.kld_annealing_epochs = kwargs.get("kld_annealing_epochs", 100)
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


        self.use_monkey_patch = config.get("use_monkey_patch", False)
        self.monkey_patch_extra_steps = config.get("monkey_patch_extra_steps", 24)
        self.home_activity_index = config.get("home_activity_index", 0)

        self.use_replicate_padding = config.get("use_replicate_padding", False)
        self.use_circular_padding = config.get("use_circular_padding", False)

        # On stocke les dimensions originales et étendues
        self.original_time_steps = self.time_steps
        if self.use_monkey_patch:
            self.extended_time_steps = self.time_steps + self.monkey_patch_extra_steps
            print(f"Monkey patch enabled: {self.time_steps} -> {self.extended_time_steps} steps")
        else:
            self.extended_time_steps = self.time_steps

        # Déterminer le padding mode
        if self.use_circular_padding:
            self.padding_mode = "circular"
        elif self.use_replicate_padding:
            self.padding_mode = "replicate"
        else:
            self.padding_mode = "zeros"

        print(f"Padding mode: {self.padding_mode}")

        # le script trouve automatiquement les deux
        self.hh_size_col_idx = config.get("hh_size_col_idx", None)
        hh_size_map = config.get("hh_size_map", None)
        if hh_size_map is not None:
            self.register_buffer("hh_size_map_tensor", torch.tensor(hh_size_map, dtype=torch.long))
        else:
            self.hh_size_map_tensor = None

        # embedding activité
        self.embedding = nn.Embedding(self.num_activities, self.hidden_size, padding_idx=0)

        # crée l'encodeur CNN (avec extended_time_steps si monkey patch)
        self.encoder_cnn = self._build_encoder()

        # calculer dynamiquement la taille post encodeur
        with torch.no_grad():
            dummy = torch.zeros(1, self.hidden_size, self.max_members, self.extended_time_steps)
            enc_out = self.encoder_cnn(dummy)
            self.c_out = enc_out.shape[1]  # Channels
            self.h_out = enc_out.shape[2]  # Members (should stay same)
            self.w_out = enc_out.shape[3]  # Compressed time
            print(f"Encoder output shape: [{self.c_out}, {self.h_out}, {self.w_out}]")
            print(f"Time compression: {self.extended_time_steps} -> {self.w_out}")

        # taille aplatie par membre après l'encodeur
        self.flat_per_member = self.c_out * self.w_out

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
        in_channels = self.hidden_size

        for i in range(self.num_layers):
            out_channels = self.hidden_size

            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                padding_mode=self.padding_mode,  # Utilise le padding mode choisi
                bias=False
            ))
            layers.append(ChannelwiseLayerNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []

        # calcul le chemin de la taille temporelle dans l'encodeur
        sizes = [self.extended_time_steps]
        size = self.extended_time_steps
        for _ in range(self.num_layers):
            size = (size + 2*self.padding - self.kernel_size) // self.stride + 1

            sizes.append(size)
        sizes.reverse()

        in_channels = self.hidden_size

        for i in range(self.num_layers):
            out_channels = self.hidden_size
            target_size = sizes[i + 1]
            current_size = sizes[i]

            # on calcule l'output padding nécessaire pour atteindre la taille cible
            expected = (current_size - 1) * self.stride - 2*self.padding + self.kernel_size
            output_padding = target_size - expected

            output_padding = max(0, output_padding)

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
            if i < self.num_layers - 1:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(self.dropout))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _apply_monkey_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Étend la séquence temporelle avec des steps supplémentaires (activité home).

        Args:
            x: [B, M, T] tensor

        Returns:
            extended: [B, M, T+extra] tensor
        """
        B, M, T = x.shape
        extra = self.monkey_patch_extra_steps

        # Crée le padding avec l'activité "home"
        home_padding = torch.full(
            (B, M, extra),
            self.home_activity_index,
            dtype=x.dtype,
            device=x.device,
        )
        extended = torch.cat([x, home_padding], dim=2)  # concat sur la dimension temporelle

        return extended

    def forward(self, batch, **kwargs):
        x = batch["x"]  # [B, M, T]
        mask = batch["mask"]  # [B, M]

        B, M, T = x.shape

        # Applique le monkey patch si activé
        if self.use_monkey_patch:
            x = self._apply_monkey_patch(x)  # [B, M, T+extra]
            T_extended = x.shape[2]
        else:
            T_extended = T

        # embedding : [B, M, T'] -> [B, M, T', C] -> [B, C, M, T'] format image
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)  # [B, C, M, T']


        mask_4d = mask.view(B, 1, M, 1).float()
        x_emb = x_emb * mask_4d

        # encodeur : [B, C, M, T'] -> [B, C, M, T'']
        features = self.encoder_cnn(x_emb)

        # features: [B, C, M, T''] -> [B, M, C, T''] -> [B, M, C*T'']
        features = features.permute(0, 2, 1, 3)  # [B, M, C, T'']
        features_flat = features.reshape(B, M, -1)  # [B, M, flat_per_member]

        # projections latentes (par membre)
        mu = self.fc_mu(features_flat)  # [B, M, latent_dim]
        logvar = self.fc_var(features_flat)  # [B, M, latent_dim]

        # Reparameterize
        z = self._reparameterize_2d(mu, logvar)  # [B, M, latent_dim]

        # decode
        log_probs = self._decode(z, mask)  # [B, M, T', num_activities]

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

        # decodeur CNN
        dec_out = self.decoder_cnn(dec_features)  # [B, C, M, T_extended]

        # Si monkey patch, on trim au temps original pour la loss
        if self.use_monkey_patch:
            dec_out = dec_out[:, :, :, :self.original_time_steps]  # [B, C, M, T_original]

        # logits -> log probs
        log_probs = self.final_projection(dec_out)  # [B, num_activities, M, T]
        log_probs = log_probs.permute(0, 2, 3, 1)  # [B, M, T, num_activities]

        return log_probs

    def loss_function(self, args, batch, **kwargs):
        log_probs = args[0]  # [B, M, T, Classes]
        mu = args[1]  # [B, M, latent_dim]
        logvar = args[2]  # [B, M, latent_dim]
        z = args[3]  # [B, M, latent_dim]
        mask = args[4]  # [B, M]
        x = batch["x"].long()  # [B, M, T_original]

        # Note: x est de taille T_original, log_probs aussi (trimé dans _decode)

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
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()
        denom = mask_expanded.sum().clamp(min=1.0)
        recon_loss = (recon_loss * mask_expanded).sum() / denom

        # KLD par membre
        kld_per_member = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, M]

        # mask et normalise
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
            z = outputs[3]
            return batch["x"], log_probs, z, batch.get("labels", torch.zeros_like(batch["mask"]))

        # Case 2: Generation
        else:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                z, labels = batch
            else:
                z = batch
                labels = None

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
