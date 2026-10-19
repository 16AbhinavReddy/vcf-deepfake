from pathlib import Path

import torch

from dataset import VCFDataset
from model import VCFDeepfakeDetector
from losses import vcf_loss


def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 1. Load one real dataset sample
    print("\n[1/5] Loading dataset...")
    dataset = VCFDataset(
        root=root / "vcf" / "vcf",
        num_frames=8,
        image_size=112,
    )
    raw, c23, c40, label, group_id = dataset[0]

    print("Dataset size:", len(dataset))
    print("Sample label:", label.item(), "(0=real, 1=fake)")
    print("Group ID:", group_id)
    print("Tensor shapes:", raw.shape, c23.shape, c40.shape)

    assert raw.shape == (3, 8, 112, 112)
    assert c23.shape == raw.shape
    assert c40.shape == raw.shape
    assert torch.isfinite(raw).all()
    assert torch.isfinite(c23).all()
    assert torch.isfinite(c40).all()

    # 2. Initialize the model
    print("\n[2/5] Initializing R3D-18...")
    model = VCFDeepfakeDetector(pretrained=True).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    raw = raw.unsqueeze(0).to(device)
    c23 = c23.unsqueeze(0).to(device)
    c40 = c40.unsqueeze(0).to(device)
    labels = label.reshape(1, 1).to(device)

    # 3. Forward pass and calculate losses
    print("\n[3/5] Running forward pass and loss...")
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits_raw, features_raw = model(raw)
        logits_c23, features_c23 = model(c23)
        logits_c40, features_c40 = model(c40)

        loss, parts = vcf_loss(
            logits_raw,
            logits_c23,
            logits_c40,
            features_raw,
            features_c23,
            features_c40,
            labels,
            lambda_cos=0.1,
        )

    print("Logits shape:", logits_raw.shape)
    print("Feature shape:", features_raw.shape)
    print("BCE loss:", parts["bce"].item())
    print("Cosine loss:", parts["cosine"].item())
    print("Total loss:", loss.item())

    assert torch.isfinite(loss).item(), "Loss is not finite!"

    # 4. Backward pass and optimizer update
    print("\n[4/5] Running backward pass and optimizer step...")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print("Backward pass and optimizer step succeeded.")

    # 5. Check GPU memory and probability output
    with torch.no_grad():
        probability = torch.sigmoid(logits_raw.float()).item()

    print("\n[5/5] Final checks")
    print(f"Raw-video fake score (untrained): {probability:.4f}")

    if device.type == "cuda":
        print(
            "Peak allocated GPU memory:",
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        )

    print("\nSMOKE TEST PASSED!")
    print("Dataset, model, loss, backward pass, and optimizer all ran.")


if __name__ == "__main__":
    main()
