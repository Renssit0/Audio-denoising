"""
Wave-U-Net and U-Net Implementation with Modular Building Blocks
================================================================

This module provides:
- ConvBlock: Basic convolutional building block
- EncoderBlock: Encoder path block with pooling
- DecoderBlock: Decoder path block with upsampling and skip connections
- FCBlock: Fully connected block for bottleneck
- UNet_FlattenBottleneck: 2D U-Net without FC bottleneck (proof of concept)
- UNet_FCBottleneck: 2D U-Net with FC bottleneck (1000 neurons)
- WaveUNet_FlattenBottleneck: 1D Wave-U-Net without FC bottleneck
- WaveUNet_FCBottleneck: 1D Wave-U-Net with FC bottleneck
- Training and testing loops with early stopping + plateau-triggered LR spikes

Author: Generated for speech enhancement experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ConvBlock(nn.Module):
    """
    Basic convolutional block with customizable parameters.
    Includes: Conv -> BatchNorm -> Activation -> Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        weight_decay: float = 1e-4,
        conv_type: str = '2d'
    ):
        super().__init__()
        self.conv_type = conv_type

        if conv_type == '1d':
            ConvLayer = nn.Conv1d
            BatchNormLayer = nn.BatchNorm1d
        else:
            ConvLayer = nn.Conv2d
            BatchNormLayer = nn.BatchNorm2d

        self.conv = ConvLayer(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn = BatchNormLayer(out_channels) if use_batchnorm else nn.Identity()

        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        self.activation = activations.get(activation, nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: Double ConvBlock + MaxPooling (optional)
    Returns both the feature map (for skip connections) and pooled output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        pool_size: int = 2,
        apply_pooling: bool = True,
        activation: str = 'relu',
        conv_type: str = '2d',
        weight_decay: float = 1e-4
    ):
        super().__init__()

        self.apply_pooling = apply_pooling
        self.conv_type = conv_type

        self.conv1 = ConvBlock(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
            dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
            activation=activation, conv_type=conv_type,
            weight_decay=weight_decay
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
            dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
            activation=activation, conv_type=conv_type,
            weight_decay=weight_decay
        )

        if apply_pooling:
            if conv_type == '1d':
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            else:
                self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + Concatenate skip + Double ConvBlock
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        scale_factor: int = 2,
        upsample_mode: str = 'transpose',
        activation: str = 'relu',
        conv_type: str = '2d',
        weight_decay: float = 1e-4
    ):
        super().__init__()

        self.conv_type = conv_type
        self.upsample_mode = upsample_mode

        if upsample_mode == 'transpose':
            if conv_type == '1d':
                self.upsample = nn.ConvTranspose1d(
                    in_channels, in_channels,
                    kernel_size=scale_factor, stride=scale_factor
                )
            else:
                self.upsample = nn.ConvTranspose2d(
                    in_channels, in_channels,
                    kernel_size=scale_factor, stride=scale_factor
                )
        else:
            self.upsample = nn.Upsample(
                scale_factor=scale_factor, 
                mode='bilinear' if conv_type == '2d' else 'linear',
                align_corners=True
            )

        self.conv1 = ConvBlock(
            in_channels + skip_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
            dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
            activation=activation, conv_type=conv_type,
            weight_decay=weight_decay
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
            dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
            activation=activation, conv_type=conv_type,
            weight_decay=weight_decay
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        if self.conv_type == '2d':
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        else:
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, [diff // 2, diff - diff // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FCBlock(nn.Module):
    """
    Fully Connected Neural Network block for bottleneck
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: List[int] = [1000, 512],
        out_features: int = None,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batchnorm: bool = True,
        weight_decay: float = 1e-4
    ):
        super().__init__()

        if out_features is None:
            out_features = in_features

        layers = []
        prev_features = in_features

        for hidden_dim in hidden_features:
            layers.append(nn.Linear(prev_features, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_features = hidden_dim

        layers.append(nn.Linear(prev_features, out_features))
        self.fc = nn.Sequential(*layers)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# U-NET ARCHITECTURES (2D - for spectrograms)
# ============================================================================

class UNet_FlattenBottleneck(nn.Module):
    """
    U-Net with simple flatten connection between encoder and decoder.
    Proof of concept - no FC layers in bottleneck.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        activation: str = 'relu',
        weight_decay: float = 1e-4
    ):
        super().__init__()

        self.depth = depth
        padding = kernel_size // 2

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(
                EncoderBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    pool_size=pool_size, apply_pooling=(i < depth - 1),
                    activation=activation, conv_type='2d',
                    weight_decay=weight_decay
                )
            )
            in_ch = out_ch

        self.bottleneck_channels = base_filters * (2 ** (depth - 1))

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = base_filters * (2 ** (i + 1))
            skip_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** i)
            self.decoders.append(
                DecoderBlock(
                    in_ch, skip_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    scale_factor=pool_size, activation=activation,
                    conv_type='2d', weight_decay=weight_decay
                )
            )

        self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            features, x = encoder(x)
            if i < self.depth - 1:
                skips.append(features)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_conv(x)


class UNet_FCBottleneck(nn.Module):
    """
    U-Net with Fully Connected bottleneck between encoder and decoder.
    Proof of concept with 1000 neurons in first FC layer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        activation: str = 'relu',
        weight_decay: float = 1e-4,
        input_size: Tuple[int, int] = (128, 128),
        fc_hidden: List[int] = [1000, 512]
    ):
        super().__init__()

        self.depth = depth
        self.input_size = input_size
        padding = kernel_size // 2

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(
                EncoderBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    pool_size=pool_size, apply_pooling=(i < depth - 1),
                    activation=activation, conv_type='2d',
                    weight_decay=weight_decay
                )
            )
            in_ch = out_ch

        self.bottleneck_channels = base_filters * (2 ** (depth - 1))
        bottleneck_h = input_size[0] // (pool_size ** (depth - 1))
        bottleneck_w = input_size[1] // (pool_size ** (depth - 1))
        self.bottleneck_spatial = (bottleneck_h, bottleneck_w)
        self.flatten_size = self.bottleneck_channels * bottleneck_h * bottleneck_w

        self.fc_bottleneck = FCBlock(
            in_features=self.flatten_size,
            hidden_features=fc_hidden,
            out_features=self.flatten_size,
            dropout_rate=dropout_rate * 2,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = base_filters * (2 ** (i + 1))
            skip_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** i)
            self.decoders.append(
                DecoderBlock(
                    in_ch, skip_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    scale_factor=pool_size, activation=activation,
                    conv_type='2d', weight_decay=weight_decay
                )
            )

        self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)

        skips = []
        for i, encoder in enumerate(self.encoders):
            features, x = encoder(x)
            if i < self.depth - 1:
                skips.append(features)

        x_flat = x.view(batch_size, -1)
        x_flat = self.fc_bottleneck(x_flat)
        x = x_flat.view(batch_size, self.bottleneck_channels, 
                        self.bottleneck_spatial[0], self.bottleneck_spatial[1])

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_conv(x)


# ============================================================================
# WAVE-U-NET ARCHITECTURES (1D - for raw waveforms)
# ============================================================================

class WaveUNet_FlattenBottleneck(nn.Module):
    """
    Wave-U-Net with simple flatten connection (proof of concept).
    Uses 1D convolutions for raw audio processing.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 24,
        depth: int = 6,
        kernel_size: int = 15,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        activation: str = 'leaky_relu',
        weight_decay: float = 1e-4
    ):
        super().__init__()

        self.depth = depth
        padding = kernel_size // 2

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (i + 1)
            self.encoders.append(
                EncoderBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    pool_size=pool_size, apply_pooling=(i < depth - 1),
                    activation=activation, conv_type='1d',
                    weight_decay=weight_decay
                )
            )
            in_ch = out_ch

        self.bottleneck_channels = base_filters * depth

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = base_filters * (i + 2)
            skip_ch = base_filters * (i + 1)
            out_ch = base_filters * (i + 1)
            self.decoders.append(
                DecoderBlock(
                    in_ch, skip_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    scale_factor=pool_size, activation=activation,
                    conv_type='1d', weight_decay=weight_decay
                )
            )

        self.output_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            features, x = encoder(x)
            if i < self.depth - 1:
                skips.append(features)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_activation(self.output_conv(x))


class WaveUNet_FCBottleneck(nn.Module):
    """
    Wave-U-Net with FC bottleneck (proof of concept with 1000 neurons).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 24,
        depth: int = 6,
        kernel_size: int = 15,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        activation: str = 'leaky_relu',
        weight_decay: float = 1e-4,
        input_length: int = 16384,
        fc_hidden: List[int] = [1000, 512]
    ):
        super().__init__()

        self.depth = depth
        self.input_length = input_length
        padding = kernel_size // 2

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (i + 1)
            self.encoders.append(
                EncoderBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    pool_size=pool_size, apply_pooling=(i < depth - 1),
                    activation=activation, conv_type='1d',
                    weight_decay=weight_decay
                )
            )
            in_ch = out_ch

        self.bottleneck_channels = base_filters * depth
        bottleneck_len = input_length // (pool_size ** (depth - 1))
        self.bottleneck_length = bottleneck_len
        self.flatten_size = self.bottleneck_channels * bottleneck_len

        self.fc_bottleneck = FCBlock(
            in_features=self.flatten_size,
            hidden_features=fc_hidden,
            out_features=self.flatten_size,
            dropout_rate=dropout_rate * 2,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = base_filters * (i + 2)
            skip_ch = base_filters * (i + 1)
            out_ch = base_filters * (i + 1)
            self.decoders.append(
                DecoderBlock(
                    in_ch, skip_ch, out_ch,
                    kernel_size=kernel_size, padding=padding,
                    dropout_rate=dropout_rate, use_batchnorm=use_batchnorm,
                    scale_factor=pool_size, activation=activation,
                    conv_type='1d', weight_decay=weight_decay
                )
            )

        self.output_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)

        skips = []
        for i, encoder in enumerate(self.encoders):
            features, x = encoder(x)
            if i < self.depth - 1:
                skips.append(features)

        x_flat = x.view(batch_size, -1)
        x_flat = self.fc_bottleneck(x_flat)
        x = x_flat.view(batch_size, self.bottleneck_channels, self.bottleneck_length)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_activation(self.output_conv(x))


# ============================================================================
# TRAINING AND TESTING LOOPS
# ============================================================================

def get_optimizer_with_weight_decay(model, base_lr=1e-4, weight_decay=1e-4):
    """Create optimizer with L2 regularization (weight decay)."""
    return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (noisy, clean) in enumerate(train_loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)

        # Handle potential size mismatch
        if output.shape != clean.shape:
            min_len = min(output.shape[-1], clean.shape[-1])
            if len(output.shape) == 4:
                min_h = min(output.shape[2], clean.shape[2])
                min_w = min(output.shape[3], clean.shape[3])
                output = output[:, :, :min_h, :min_w]
                clean = clean[:, :, :min_h, :min_w]
            else:
                output = output[:, :, :min_len]
                clean = clean[:, :, :min_len]

        loss = criterion(output, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            if output.shape != clean.shape:
                min_len = min(output.shape[-1], clean.shape[-1])
                if len(output.shape) == 4:
                    min_h = min(output.shape[2], clean.shape[2])
                    min_w = min(output.shape[3], clean.shape[3])
                    output = output[:, :, :min_h, :min_w]
                    clean = clean[:, :, :min_h, :min_w]
                else:
                    output = output[:, :, :min_len]
                    clean = clean[:, :, :min_len]

            loss = criterion(output, clean)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Test the model and return metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            if output.shape != clean.shape:
                min_len = min(output.shape[-1], clean.shape[-1])
                if len(output.shape) == 4:
                    min_h = min(output.shape[2], clean.shape[2])
                    min_w = min(output.shape[3], clean.shape[3])
                    output = output[:, :, :min_h, :min_w]
                    clean = clean[:, :, :min_h, :min_w]
                else:
                    output = output[:, :, :min_len]
                    clean = clean[:, :, :min_len]

            loss = criterion(output, clean)
            total_loss += loss.item()
            num_batches += 1

            all_outputs.append(output.cpu())
            all_targets.append(clean.cpu())

    return {
        'test_loss': total_loss / num_batches,
        'outputs': torch.cat(all_outputs, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }


class EarlyStopperWithPlateau:
    """Early stopping with plateau-triggered LR spike."""
    def __init__(
        self,
        patience_plateau: int = 10,
        patience_after_spike: int = 15,
        spike_cycles: int = 6,
        min_delta: float = 1e-4,
        verbose: bool = True
    ):
        self.patience_plateau = patience_plateau
        self.patience_after_spike = patience_after_spike
        self.spike_cycles = spike_cycles
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.spike_triggered = False
        self.epochs_after_spike = 0

    def __call__(self, val_loss):
        improved = val_loss < (self.best_loss - self.min_delta)

        if improved:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            self.spike_triggered = False
            self.epochs_after_spike = 0
            if self.verbose:
                print(f"  [EarlyStopper] New best: {val_loss:.6f}")
            return False, False

        self.epochs_no_improve += 1

        if self.spike_triggered:
            self.epochs_after_spike += 1
            if self.epochs_after_spike >= self.patience_after_spike:
                if self.verbose:
                    print(f"  [EarlyStopper] No improvement after spike. Stopping.")
                return True, False

        if not self.spike_triggered and self.epochs_no_improve >= self.patience_plateau:
            if self.verbose:
                print(f"  [EarlyStopper] Plateau detected. Triggering LR spike cycle.")
            self.spike_triggered = True
            return False, True

        return False, False


def full_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience_plateau: int = 10,
    patience_after_spike: int = 15,
    spike_max_lr_factor: float = 5.0,
    spike_cycles: int = 6,
    verbose: bool = True
):
    """Full training loop with early stopping and plateau-triggered LR spikes."""
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    early_stopper = EarlyStopperWithPlateau(
        patience_plateau=patience_plateau,
        patience_after_spike=patience_after_spike,
        spike_cycles=spike_cycles,
        verbose=verbose
    )

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        should_stop, trigger_spike = early_stopper(val_loss)

        if trigger_spike:
            spike_lr = base_lr * spike_max_lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = spike_lr
            if verbose:
                print(f"  [Spike] LR increased to {spike_lr:.2e} for exploration")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=spike_cycles, eta_min=base_lr
            )

        if should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Example 1: U-Net (2D) with Flatten Bottleneck
    print("\n" + "="*60)
    print("U-Net (2D) - Flatten Bottleneck")
    print("="*60)

    unet_flatten = UNet_FlattenBottleneck(
        in_channels=1, out_channels=1, base_filters=64, depth=4,
        kernel_size=3, pool_size=2, dropout_rate=0.1
    )

    dummy_spec = torch.randn(2, 1, 128, 128)
    output = unet_flatten(dummy_spec)
    print(f"Input shape:  {dummy_spec.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in unet_flatten.parameters()):,}")

    # Example 2: U-Net (2D) with FC Bottleneck
    print("\n" + "="*60)
    print("U-Net (2D) - FC Bottleneck (1000 neurons)")
    print("="*60)

    unet_fc = UNet_FCBottleneck(
        in_channels=1, out_channels=1, base_filters=64, depth=4,
        input_size=(128, 128), fc_hidden=[1000, 512]
    )

    output = unet_fc(dummy_spec)
    print(f"Input shape:  {dummy_spec.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in unet_fc.parameters()):,}")

    # Example 3: Wave-U-Net (1D) with Flatten Bottleneck
    print("\n" + "="*60)
    print("Wave-U-Net (1D) - Flatten Bottleneck")
    print("="*60)

    wave_unet_flatten = WaveUNet_FlattenBottleneck(
        in_channels=1, out_channels=1, base_filters=24, depth=6,
        kernel_size=15, pool_size=2, dropout_rate=0.1
    )

    dummy_audio = torch.randn(2, 1, 16384)
    output = wave_unet_flatten(dummy_audio)
    print(f"Input shape:  {dummy_audio.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in wave_unet_flatten.parameters()):,}")

    # Example 4: Wave-U-Net (1D) with FC Bottleneck
    print("\n" + "="*60)
    print("Wave-U-Net (1D) - FC Bottleneck (1000 neurons)")
    print("="*60)

    wave_unet_fc = WaveUNet_FCBottleneck(
        in_channels=1, out_channels=1, base_filters=24, depth=6,
        kernel_size=15, input_length=16384, fc_hidden=[1000, 512]
    )

    output = wave_unet_fc(dummy_audio)
    print(f"Input shape:  {dummy_audio.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in wave_unet_fc.parameters()):,}")

    print("\n" + "="*60)
    print("All models created successfully!")
    print("="*60)
