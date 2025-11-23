import os
from typing import Tuple

import torch
from torchvision.utils import save_image

from .diffusion import DiffusionConfig, DiffusionProcess
from .trainers import langevin_sample


def generate_diffusion_samples(model,
                               device: str = "cpu",
                               num_samples: int = 16,
                               timesteps: int = 200,
                               out_dir: str = "outputs",
                               filename_prefix: str = "diffusion") -> str:
    """Generate images using the reverse diffusion process.

    Returns the path to the saved grid image (PNG).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    cfg = DiffusionConfig(timesteps=timesteps)
    diffusion = DiffusionProcess(cfg, device=device)

    shape: Tuple[int, int, int, int] = (num_samples, 3, 32, 32)
    samples = diffusion.p_sample_loop(model.to(device), shape, device=device)
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0  # back to [0, 1]

    out_path = os.path.join(out_dir, f"{filename_prefix}_samples.png")
    save_image(samples, out_path, nrow=int(num_samples ** 0.5))
    return out_path


def generate_energy_samples(model,
                            device: str = "cpu",
                            num_samples: int = 16,
                            langevin_steps: int = 60,
                            out_dir: str = "outputs",
                            filename_prefix: str = "energy") -> str:
    """Generate samples by running Langevin dynamics on noise inputs.

    Returns the path to the saved grid image (PNG).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    init_x = torch.randn(num_samples, 3, 32, 32, device=device)
    samples = langevin_sample(model.to(device), init_x,
                              steps=langevin_steps,
                              step_size=0.1,
                              noise_scale=0.01)
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0

    out_path = os.path.join(out_dir, f"{filename_prefix}_samples.png")
    save_image(samples, out_path, nrow=int(num_samples ** 0.5))
    return out_path
