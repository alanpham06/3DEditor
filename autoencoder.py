from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


###############################################################################
# 1. Configuration
###############################################################################

DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 128
LATENT_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# 2. Data Loading
###############################################################################

def build_dataloaders():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


###############################################################################
# 3. Model Definition
###############################################################################

class Autoencoder(nn.Module):
    def __init__(self, input_dim=28 * 28, latent_dim=LATENT_DIM):
        super().__init__()

        # The encoder compresses a 784-value image into a smaller latent vector.
        self.encoder = nn.Linear(input_dim, latent_dim)

        # The decoder expands that latent vector back into image-sized output.
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


###############################################################################
# 4. Training Step
###############################################################################

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, _ in dataloader:
        images = images.view(images.size(0), -1).to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


###############################################################################
# 5. Evaluation / Encoding Demo
###############################################################################

@torch.no_grad()
def inspect_reconstruction(model, dataloader):
    model.eval()

    images, _ = next(iter(dataloader))
    images = images.to(DEVICE)

    flat_images = images.view(images.size(0), -1)
    latent_vectors = model.encode(flat_images)
    reconstructed = model.decode(latent_vectors)

    print(f"Input image shape: {images.shape}")
    print(f"Flattened image shape: {flat_images.shape}")
    print(f"Latent vector shape: {latent_vectors.shape}")
    print(f"Reconstructed output shape: {reconstructed.shape}")

    original = images[0].cpu().squeeze(0)
    rebuilt = reconstructed[0].cpu().view(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(rebuilt, cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


###############################################################################
# 6. Main Script
###############################################################################

def main():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader = build_dataloaders()

    model = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch}/{EPOCHS} - training loss: {train_loss:.6f}")

    inspect_reconstruction(model, test_loader)


if __name__ == "__main__":
    main()