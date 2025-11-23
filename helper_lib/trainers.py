from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

from .diffusion import DiffusionConfig, DiffusionProcess


def train_diffusion(model: nn.Module,
                    dataloader: DataLoader,
                    device: str = "cpu",
                    epochs: int = 1,
                    lr: float = 2e-4,
                    timesteps: int = 200,
                    max_batches_per_epoch: Optional[int] = 100) -> nn.Module:
    """Train a DDPM-style diffusion model on CIFAR-10.

    This implementation is intentionally lightweight so it can run on CPU
    for a few epochs just to demonstrate correctness for the assignment.
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    cfg = DiffusionConfig(timesteps=timesteps)
    diffusion = DiffusionProcess(cfg, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            x = x.to(device) * 2.0 - 1.0  # scale to [-1, 1]
            b = x.size(0)
            t = torch.randint(0, timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)

            # model predicts noise
            eps_theta = model(x_t, t)
            loss = mse(eps_theta, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"[Diffusion] Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}")

    return model


def langevin_sample(model: nn.Module,
                    init_x: torch.Tensor,
                    steps: int = 30,
                    step_size: float = 0.1,
                    noise_scale: float = 0.01) -> torch.Tensor:
    """Run Langevin dynamics on inputs to move them towards low-energy regions.

    Gradients are taken w.r.t. the inputs (init_x), not the model parameters.
    """
    x = init_x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        energy = model(x).mean()
        grad_x, = torch.autograd.grad(energy, x)
        x = x - step_size * grad_x + noise_scale * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
    return x.detach()


def train_energy(model: nn.Module,
                 dataloader: DataLoader,
                 device: str = "cpu",
                 epochs: int = 1,
                 lr: float = 1e-4,
                 n_negative_steps: int = 20,
                 max_batches_per_epoch: Optional[int] = 100) -> nn.Module:
    """Train an Energy-Based Model using simple contrastive divergence.

    Positive samples: real CIFAR-10 images.
    Negative samples: initialized from noise and refined via Langevin dynamics.
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            x = x.to(device) * 2.0 - 1.0  # scale to [-1, 1]

            # Positive energy
            energy_pos = model(x).mean()

            # Negative samples via Langevin dynamics starting from Gaussian noise
            x_neg_init = torch.randn_like(x)
            x_neg = langevin_sample(model, x_neg_init,
                                    steps=n_negative_steps,
                                    step_size=0.1,
                                    noise_scale=0.01)
            energy_neg = model(x_neg).mean()

            # We want E(real) < E(fake)
            loss = energy_pos - energy_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"[EBM] Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f} "
                      f"(E_pos={energy_pos.item():.3f}, E_neg={energy_neg.item():.3f})")

    return model
