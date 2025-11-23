# Assignment 4 — Advanced Image Generation (Diffusion + Energy-Based Models)

This repository implements the **Practice: Model Deployment** portion of Assignment 4.\
It trains a **Diffusion Model** and an **Energy-Based Model (EBM)** on CIFAR-10, exposes
both models through a **FastAPI** server, and provides a **Docker** deployment.

The **theory questions** (diffusion and energy-based model building blocks) are answered
in `theory_answers.md`.

## Project Layout

```text
assignment4_advanced_image_generation/
  app/
    main.py                # FastAPI application with /train, /generate, /generate_image endpoints
  helper_lib/
    __init__.py
    models.py              # Diffusion UNet + Energy CNN
    diffusion.py           # DiffusionProcess class (forward + reverse diffusion)
    trainers.py            # train_diffusion, train_energy, Langevin sampling
    generator.py           # Image generation helpers (save PNG grids)
  scripts/
    train_diffusion_cifar10.py
    train_energy_cifar10.py
  Dockerfile
  requirements.txt
  README.md
  theory_answers.md
```

## 1. Local Setup (MacBook Intel / CPU-friendly)

```bash
cd assignment4_advanced_image_generation

python -m venv .venv
source .venv/bin/activate      # on macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On an Intel Mac this will run on **CPU**. Training for 1 epoch with
> `max_batches_per_epoch=100` is enough to demonstrate the pipeline.

## 2. Train the Models (CLI helper scripts)

You can quickly train both models using the helper scripts (recommended first run):

```bash
# From project root
python scripts/train_diffusion_cifar10.py
python scripts/train_energy_cifar10.py
```

Each script will download CIFAR-10 automatically and save weights as:

- `diffusion_cifar10.pt`
- `energy_cifar10.pt`

## 3. Run the FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

Test health:

```bash
curl http://127.0.0.1:8000/health
```

### 3.1 Train via API

Example: train diffusion for 1 epoch and 50 batches (for speed):

```bash
curl -X POST "http://127.0.0.1:8000/train" \         -H "Content-Type: application/json" \         -d '{"model_type": "diffusion", "epochs": 1, "max_batches_per_epoch": 50}'
```

Train energy-based model:

```bash
curl -X POST "http://127.0.0.1:8000/train" \         -H "Content-Type: application/json" \         -d '{"model_type": "energy", "epochs": 1, "max_batches_per_epoch": 50}'
```

This will create `diffusion_cifar10.pt` or `energy_cifar10.pt` in the working directory.

### 3.2 Generate Images via API (JSON + image path)

Once a model is trained and its weights are saved, you can generate images and get the
path to a PNG file saved under `outputs/`:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \         -H "Content-Type: application/json" \         -d '{"model_type": "diffusion", "num_samples": 16}'
```

```bash
curl -X POST "http://127.0.0.1:8000/generate" \         -H "Content-Type: application/json" \         -d '{"model_type": "energy", "num_samples": 16}'
```

The response will include the path to a PNG grid, for example:

```json
{
  "message": "Diffusion samples generated.",
  "image_path": "outputs/diffusion_samples.png"
}
```

### 3.3 Generate and Download Image Directly (PNG response)

To avoid the previous issue ("only encoding text"), this project also exposes a
`/generate_image` endpoint that **returns a real PNG image** instead of only JSON.

Example: diffusion model, save the image as `diffusion.png`:

```bash
curl -o diffusion.png -X POST "http://127.0.0.1:8000/generate_image" \         -H "Content-Type: application/json" \         -d '{"model_type": "diffusion", "num_samples": 16}'
```

Example: energy-based model, save as `energy.png`:

```bash
curl -o energy.png -X POST "http://127.0.0.1:8000/generate_image" \         -H "Content-Type: application/json" \         -d '{"model_type": "energy", "num_samples": 16}'
```

This guarantees that the output is **a real PNG image**, not just a text encoding.
Images are also saved inside the container under the `outputs/` directory.

## 4. Docker Deployment (as required by the rubric)

Build the image:

```bash
docker build -t assignment4-image-gen .
```

Run the container:

```bash
docker run --rm -p 8000:8000 assignment4-image-gen
```

Then, from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/train" \         -H "Content-Type: application/json" \         -d '{"model_type": "diffusion", "epochs": 1, "max_batches_per_epoch": 50}'
curl -X POST "http://127.0.0.1:8000/generate_image" \         -H "Content-Type: application/json" \         -d '{"model_type": "diffusion", "num_samples": 16}'
```

This satisfies the rubric requirements:

1. Code committed to GitHub (you can push this folder as-is).
2. Docker deployment that runs a FastAPI server with added endpoints.
3. API can be queried to train and run the diffusion / EBM models.
4. Code is modular and well organized in `helper_lib` + `app`.
5. Theory questions are answered in `theory_answers.md`.
6. Output **saves real PNG images** and `/generate_image` returns a real `image/png` response.
