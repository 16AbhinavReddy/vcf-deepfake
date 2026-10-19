from collections import Counter
from pathlib import Path

from dataset import VCFDataset


dataset = VCFDataset(root="vcf/vcf", num_frames=8, image_size=112)

print("Total samples:", len(dataset))
print("Label counts:", Counter(r["label"] for r in dataset.records))
print("Unique source groups:", len(set(r["group_id"] for r in dataset.records)))

for i in [0, len(dataset) // 2, len(dataset) - 1]:
    record = dataset.records[i]
    print("\nExample:", record["relative_path"])
    print("Label:", record["label"], "(real=0, fake=1)")
    print("Group ID:", record["group_id"])

print("\nReading one sample from disk...")
raw, c23, c40, label, group_id = dataset[0]

print("Raw shape:", tuple(raw.shape))
print("c23 shape:", tuple(c23.shape))
print("c40 shape:", tuple(c40.shape))
print("Label:", label.item())
print("Group ID:", group_id)
print("All tensors finite:",
      raw.isfinite().all().item()
      and c23.isfinite().all().item()
      and c40.isfinite().all().item())
