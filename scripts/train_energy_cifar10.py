import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib import get_model, train_energy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)

    model = get_model("energy")
    model = train_energy(model, dl, device=device, epochs=1, max_batches_per_epoch=100)
    torch.save(model.state_dict(), "energy_cifar10.pt")
    print("Saved energy_cifar10.pt")


if __name__ == "__main__":
    main()
