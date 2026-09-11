import os
import torch
from torch.utils.data import DataLoader

from data.dataset import MNISTDigitDataset
from flow.paths import sample_conditional_path, conditional_vector_field
from models.vector_field_unet import VectorFieldUNet
from config import DIGIT, SIGMA, DIM, BASE_CHANNELS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_PATH

def train_unet(
    digit: int = DIGIT,
    sigma: float = SIGMA,
    dim: int = DIM,
    base_channels: int = BASE_CHANNELS,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    checkpoint_path: str = CHECKPOINT_PATH,
):
    dataset = MNISTDigitDataset(digit=digit, train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VectorFieldUNet(dim=dim, base_channels=base_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for x in loader:
            t = torch.rand(x.size(0), 1)
            y, _ = sample_conditional_path(x, t, sigma)
            xi_target = conditional_vector_field(y, t, x, sigma)

            xi_pred = model(y, t)
            loss = ((xi_pred - xi_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}  |  Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved in {checkpoint_path}")

    return model


if __name__ == "__main__":
    train_unet()