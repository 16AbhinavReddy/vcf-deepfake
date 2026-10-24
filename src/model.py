import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VCFDeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Pretrained R3D-18 backbone
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)

        # Remove the original R3D-18 classification layer.
        # The backbone outputs a 512-dimensional feature vector.
        self.backbone.fc = nn.Identity()

        # Original linear classification head: 512 features -> 1 logit.
        # No 256-dimensional hidden layer, ReLU, or dropout.
        self.classifier = nn.Linear(512, 1)

    def forward(self, video):
        """
        video:    (B, C, T, H, W)
        features: (B, 512)
        logits:   (B, 1)
        """
        features = self.backbone(video)
        logits = self.classifier(features)
        return logits, features


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VCFDeepfakeDetector(pretrained=True).to(device)
    model.eval()

    sample = torch.randn(1, 3, 8, 112, 112, device=device)

    with torch.no_grad():
        logits, features = model(sample)

    print("Device:", device)
    print("Logits shape:", tuple(logits.shape))
    print("Feature shape:", tuple(features.shape))
    print("Fake probability:", torch.sigmoid(logits).item())

