
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VCFDeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            R3D_18_Weights.DEFAULT if pretrained else None
        )
        self.backbone = r3d_18(weights=weights)

        # R3D-18's original classification layer is replaced.
        self.backbone.fc = nn.Identity()

        # R3D-18 produces a 512-dimensional feature vector.
        self.classifier = nn.Linear(512, 1)

    def forward(self, video):
        """
        video: (B, C, T, H, W)
        B = batch size
        C = 3 RGB channels
        T = number of frames
        H, W = frame dimensions
        """
        features = self.backbone(video)       # (B, 512)
        logits = self.classifier(features)    # (B, 1)

        return logits, features


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VCFDeepfakeDetector(pretrained=True).to(device)

    # One video, 8 RGB frames, 112 x 112 resolution
    sample = torch.randn(1, 3, 8, 112, 112).to(device)

    with torch.no_grad():
        logits, features = model(sample)

    print("Device:", device)
    print("Logits shape:", logits.shape)
    print("Feature shape:", features.shape)
    print("Fake probability:", torch.sigmoid(logits).item())

