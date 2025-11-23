import math
from dataclasses import dataclass
from typing import Tuple

import torch

@dataclass
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DiffusionProcess:
    """Utility class that implements forward and reverse diffusion."""
    def __init__(self, cfg: DiffusionConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._build()

    def _build(self):
        T = self.cfg.timesteps
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: sample x_t given x_0 and timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One reverse diffusion step p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        ac_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        ac_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)

        # model predicts noise eps_theta(x_t, t)
        eps_theta = model(x_t, t)

        # equation for mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alphas_t * (x_t - betas_t / torch.sqrt(1.0 - ac_t) * eps_theta)

        # add noise except for t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        sample = mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return sample

    def p_sample_loop(self, model, shape: Tuple[int, int, int, int], device: str = "cpu") -> torch.Tensor:
        """Generate samples starting from pure noise x_T ~ N(0, I)."""
        model.to(device)
        model.eval()
        img = torch.randn(shape, device=device)
        T = self.cfg.timesteps
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            with torch.no_grad():
                img = self.p_sample(model, img, t)
        return img
