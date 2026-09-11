import torch
import torch.nn as nn
from config import DIM, BASE_CHANNELS

class VectorFieldUNet(nn.Module):

    def __init__(self, dim: int = DIM, base_channels: int = BASE_CHANNELS):
        super().__init__()
        self.image_size = int(dim ** 0.5)

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.enc1 = nn.Sequential(
            nn.Conv2d(1 + 1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=3, padding=1)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = y.size(0)

        #Reshape flat vectors (batch,784) back to image format and inject t (--> shape (batch,2,28,28))
        y_img = y.view(batch, 1, self.image_size, self.image_size)
        t_img = t.view(batch, 1, 1, 1).expand(batch, 1, self.image_size, self.image_size)
        yt = torch.cat([y_img, t_img], dim=1)

        #Acutal forward calculation
        e1 = self.enc1(yt)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1)

        out = self.out_conv(u1)
        return out.view(batch, -1)
