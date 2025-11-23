from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib import get_model, train_diffusion, train_energy
from helper_lib import generate_diffusion_samples, generate_energy_samples

app = FastAPI(title="Assignment 4 - Advanced Image Generation API")


class TrainRequest(BaseModel):
    model_type: str  # "diffusion" or "energy"
    epochs: int = 1
    max_batches_per_epoch: int = 100


class GenerateRequest(BaseModel):
    model_type: str  # "diffusion" or "energy"
    num_samples: int = 16


def _get_cifar10_loader(batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train_endpoint(req: TrainRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = _get_cifar10_loader()

    if req.model_type.lower() == "diffusion":
        model = get_model("diffusion")
        model = train_diffusion(model, dl, device=device,
                                epochs=req.epochs,
                                max_batches_per_epoch=req.max_batches_per_epoch)
        torch.save(model.state_dict(), "diffusion_cifar10.pt")
        return {"message": "Diffusion model trained and saved to diffusion_cifar10.pt"}
    elif req.model_type.lower() == "energy":
        model = get_model("energy")
        model = train_energy(model, dl, device=device,
                             epochs=req.epochs,
                             max_batches_per_epoch=req.max_batches_per_epoch)
        torch.save(model.state_dict(), "energy_cifar10.pt")
        return {"message": "Energy model trained and saved to energy_cifar10.pt"}
    else:
        return {"error": "model_type must be 'diffusion' or 'energy'"}


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    """Generate samples and return JSON with path to saved PNG image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if req.model_type.lower() == "diffusion":
        model = get_model("diffusion")
        try:
            state = torch.load("diffusion_cifar10.pt", map_location=device)
            model.load_state_dict(state)
        except FileNotFoundError:
            return {"error": "diffusion_cifar10.pt not found. Train the model first via /train."}

        out_path = generate_diffusion_samples(model, device=device,
                                              num_samples=req.num_samples,
                                              filename_prefix="diffusion")
        return {"message": "Diffusion samples generated.", "image_path": out_path}

    elif req.model_type.lower() == "energy":
        model = get_model("energy")
        try:
            state = torch.load("energy_cifar10.pt", map_location=device)
            model.load_state_dict(state)
        except FileNotFoundError:
            return {"error": "energy_cifar10.pt not found. Train the model first via /train."}

        out_path = generate_energy_samples(model, device=device,
                                           num_samples=req.num_samples,
                                           filename_prefix="energy")
        return {"message": "Energy-based samples generated.", "image_path": out_path}
    else:
        return {"error": "model_type must be 'diffusion' or 'energy'"}


@app.post("/generate_image")
def generate_image_endpoint(req: GenerateRequest):
    """Generate samples and directly return a PNG image (not just JSON)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if req.model_type.lower() == "diffusion":
        model = get_model("diffusion")
        try:
            state = torch.load("diffusion_cifar10.pt", map_location=device)
            model.load_state_dict(state)
        except FileNotFoundError:
            return {"error": "diffusion_cifar10.pt not found. Train the model first via /train."}

        out_path = generate_diffusion_samples(model, device=device,
                                              num_samples=req.num_samples,
                                              filename_prefix="diffusion")
        return FileResponse(out_path, media_type="image/png")

    elif req.model_type.lower() == "energy":
        model = get_model("energy")
        try:
            state = torch.load("energy_cifar10.pt", map_location=device)
            model.load_state_dict(state)
        except FileNotFoundError:
            return {"error": "energy_cifar10.pt not found. Train the model first via /train."}

        out_path = generate_energy_samples(model, device=device,
                                           num_samples=req.num_samples,
                                           filename_prefix="energy")
        return FileResponse(out_path, media_type="image/png")
    else:
        return {"error": "model_type must be 'diffusion' or 'energy'"}
