from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from caveat.models import utils
from caveat.models.base import Base
from caveat.models.embed import PositionalEmbedding
from caveat.models.utils import calc_output_padding_1d, conv1d_size
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

class VAEDiscCNN1D(Base):
    def __init__(self, *args, **kwargs):
        """Convolution based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        hidden_layers = list
        latent_dim = int
        dropout = Optional[float]
        kernel_size = Optional[int]
        stride = Optional[int]
        padding = Optional[int]

        encoded_size = config.get("embed_size", self.encodings)
        hidden_layers = hidden_layers = utils.build_hidden_layers(config)
        latent_dim = config["latent_dim"]
        dropout = config.get("dropout", 0)
        kernel_size = config.get("kernel_size", 2)
        stride = config.get("stride", 2)
        padding = config.get("padding", 0)

        
        self.use_positional_embedding = config.get("use_positional_embedding", False)
        self.use_replicate_padding = config.get("use_replicate_padding", False)

        # Monkey patch pour extension de durée (en steps)
        self.use_monkey_patch = config.get("use_monkey_patch", False)
        self.monkey_patch_extra_steps = config.get("monkey_patch_extra_steps", 24)
        self.home_activity_index = config.get("home_activity_index", 0)

        # on stocke la longueur originale pour masker le reste dans la loss
        # in_shape = schedules[0].shape -> c'est le nombre de time steps originaux
        self.original_length = self.in_shape[0]
        in_shape_tuple = (self.in_shape[0], 1)
        
        if self.use_monkey_patch:
            extended_length = self.original_length + self.monkey_patch_extra_steps
            modified_in_shape = (extended_length, in_shape_tuple[1])
        else:
            modified_in_shape = in_shape_tuple

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_encoding=self.encodings,
            encoded_size=encoded_size,
            in_shape=modified_in_shape,
            hidden_layers=hidden_layers,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_positional_embedding=self.use_positional_embedding,
            use_replicate_padding=self.use_replicate_padding,
        )
        self.decoder = Decoder(
            encoded_size=encoded_size,
            target_shapes=self.encoder.shapes,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.fc_mu = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, self.encoder.flat_size)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def _apply_monkey_patch(self, x: Tensor) -> Tensor:
        # rajoute les steps supplémentaires avec l'activité "home"
        B, L = x.shape
        extra = self.monkey_patch_extra_steps

        # Create home padding tensor
        home_padding = torch.full(
            (B, extra),
            self.home_activity_index,
            dtype=x.dtype,
            device=x.device,
        )
        extended = torch.cat([x, home_padding], dim=1)

        return extended

    def encode(self, input: Tensor, labels: Optional[Tensor] = None) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # Apply monkey patch if enabled
        if self.use_monkey_patch:
            input = self._apply_monkey_patch(input)

        hidden = self.encoder(input)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return [mu, log_var]

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        hidden = self.fc_hidden(z)
        hidden = hidden.view(self.encoder.shape_before_flattening)
        log_probs = self.decoder(hidden)

        # If monkey patch enabled, trim output to original length for loss
        if self.use_monkey_patch:
            log_probs = log_probs[:, : self.original_length, :]

        return log_probs

    def loss_function(
        self, log_probs, mu, log_var, target, weights, *args, **kwargs
    ) -> dict:
        return self.discretized_loss(
            log_probs, mu, log_var, target, weights, *args, **kwargs
        )

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        """Given samples from latent space, return decoder output.

        If monkey patch is enabled, returns full extended sequence.

        Args:
            z (tensor): [N, latent_dims].
            device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        hidden = self.fc_hidden(z)
        hidden = hidden.view(self.encoder.shape_before_flattening)
        log_probs = self.decoder(hidden)

        # For prediction, trim to original length
        if self.use_monkey_patch:
            log_probs = log_probs[:, : self.original_length, :]

        prob_samples = torch.exp(log_probs)
        return prob_samples


class Encoder(nn.Module):
    def __init__(
        self,
        input_encoding: int,
        encoded_size: int,
        in_shape: tuple,
        hidden_layers: list,
        dropout: float = 0.1,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 1,
        use_positional_embedding: bool = False,
        use_replicate_padding: bool = False,
    ):
        """1d Convolutions Encoder.

        Args:
            input_encoding (int): number of encoding classes.
            encoded_size (int): number of encoding classes and hidden size.
            in_shape (tuple[int, int, int]): [C, time_step, activity_encoding].
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (int): kernel size. Defaults to 2.
            stride (int): stride. Defaults to 2.
            padding (int): padding. Defaults to 1.
            use_positional_embedding (bool): Whether to use positional embedding.
            use_replicate_padding (bool): Whether to use replicate padding mode.
        """
        super(Encoder, self).__init__()
        length = in_shape[0]
        channels = encoded_size
        self.shapes = []
        modules = []

        self.embedding = nn.Embedding(input_encoding, encoded_size)

        # Positional embedding
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.positional_embedding = PositionalEmbedding(
                max_len=length, d_model=encoded_size
            )

        # Padding mode
        padding_mode = "replicate" if use_replicate_padding else "zeros"

        for hidden_channels in hidden_layers:
            self.shapes.append((channels, length))
            if length + padding < kernel_size:
                print("Skipping convolution:", length, kernel_size)
                break
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                        padding_mode=padding_mode,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
            )
            length = conv1d_size(
                length=length,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            channels = hidden_channels
        self.shapes.append((channels, length))

        self.encoder = nn.Sequential(*modules)
        self.shape_before_flattening = (-1, channels, length)
        self.flat_size = int(channels * length)

    def forward(self, x):
        y = self.embedding(x.int())
        y = y.permute(0, 2, 1)  # [B, C, T]

        if self.use_positional_embedding:
            y = self.positional_embedding(y)

        y = self.encoder(y)
        y = y.flatten(start_dim=1)
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        encoded_size: int,
        target_shapes: list,
        dropout: float = 0.1,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
    ):
        """1d Conv Decoder.

        Args:
            encoded_size (int): number of encoding classes and hidden size.
            target_shapes (list): list of target shapes.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (int): kernel size. Defaults to 3.
            stride (int): stride. Defaults to 2.
            padding (int): padding. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.hidden_size = encoded_size
        modules = []
        target_shapes.reverse()

        for i in range(len(target_shapes) - 1):
            c_in, l_in = target_shapes[i]
            c_out, l_out = target_shapes[i + 1]
            if c_in == c_out and l_in == l_out:
                print(
                    "Skipping transpose convolution:", c_in, l_in, c_out, l_out
                )
                continue
            in_padding, out_padding = calc_output_padding_1d(
                length=l_in,
                target=l_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            block = [
                nn.ConvTranspose1d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=in_padding,
                    output_padding=out_padding,
                    bias=False,
                ),
                nn.BatchNorm1d(c_out),
            ]
            if i < len(target_shapes) - 2:
                block.append(nn.LeakyReLU())
                block.append(nn.Dropout(dropout))
            modules.append(nn.Sequential(*block))

        self.decoder = nn.Sequential(*modules)
        self.logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, hidden, **kwargs):
        y = self.decoder(hidden)
        y = y.permute(0, 2, 1)
        return self.logprob_activation(y)
