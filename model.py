import torch
import torch.nn as nn

# === Denoising Generator ===
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 64, 4, stride=2, padding=1),  # (B, 64, 512)
            nn.ReLU(True),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),  # (B, 128, 256)
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),  # (B, 64, 512)
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 2, 4, stride=2, padding=1),    # (B, 2, 1024)
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)