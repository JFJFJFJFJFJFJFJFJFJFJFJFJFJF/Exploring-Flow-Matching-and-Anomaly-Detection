import torch
import torch.nn as nn

class VectorFieldNN(nn.Module):

    def __init__(self, dim: int = 784, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),   #y (dim) + t (1)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),        # Ausgabe: wieder dim
        )

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        yt = torch.cat([y, t], dim=1)   # (batch, dim+1)
        return self.net(yt)

if __name__ == "__main__":
    from flow.paths import sample_conditional_path, conditional_vector_field

    torch.manual_seed(0)

    batch_size = 4
    dim = 784
    sigma = 0.01

    x = torch.randn(batch_size, dim)
    t = torch.rand(batch_size, 1)

    y, _ = sample_conditional_path(x, t, sigma)
    xi_target = conditional_vector_field(y, t, x, sigma)

    model = VectorFieldNN(dim=dim, hidden_dim=256)
    xi_pred = model(y, t)

    print(f"xi_pred shape: {xi_pred.shape}")
    print(f"xi_target shape: {xi_target.shape}")

    loss = ((xi_pred - xi_target) ** 2).mean()
    print(loss.item())