import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Small UNet-like backbone for diffusion on CIFAR-10 (32x32x3) ----

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by an MLP projection."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) int64 or float
        if t.dtype != torch.float32:
            t = t.float()
        device = t.device
        half_dim = self.dim // 2
        max_period = 10000.0
        exponents = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponents = -math.log(max_period) * exponents / (half_dim - 1)
        freqs = torch.exp(exponents)  # (half_dim,)
        args = t[:, None] * freqs[None, :]  # (batch, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
        return self.proj(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        # x: (B, C, H, W), t_emb: (B, time_emb_dim)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # add time embedding
        temb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + temb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SmallUNet(nn.Module):
    """A small UNet-style network for DDPM on CIFAR-10 sized images."""
    def __init__(self, time_emb_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)

        # encoder
        self.conv_in = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

        # bottleneck
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # decoder
        self.up1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        self.conv_out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x, t):
        # x: (B, 3, 32, 32), t: (B,) integer timesteps
        t_emb = self.time_mlp(t)

        # encoder
        x1 = self.conv_in(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.pool(x2)
        x4 = self.down2(x3, t_emb)
        x5 = self.pool(x4)

        # bottleneck
        h = self.bot1(x5, t_emb)

        # decoder (simple upsample + residual)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(h, t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(h, t_emb)
        out = self.conv_out(h)
        return out  # predict noise eps(x_t, t)


# ---- Energy-Based Model for CIFAR-10 ----

class EnergyCNN(nn.Module):
    """Simple CNN that outputs a scalar energy per image."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        energy = self.fc(h)
        return energy  # (B, 1)

def get_model(name: str):
    name = name.lower()
    if name == "diffusion":
        return SmallUNet()
    elif name == "energy":
        return EnergyCNN()
    else:
        raise ValueError(f"Unknown model name: {name}")
