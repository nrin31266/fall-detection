from pathlib import Path

import matplotlib.pyplot as plt

from video_dataset import FallVideoDataset


OUTPUT_DIR = Path("results/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = "data/video_index_16.csv"
IMAGE_SIZE = 224


def save_clip_grid(sample, output_path: Path):
    clip = sample["clip"]  # [T, C, H, W]
    label = int(sample["label"].item())
    label_name = "Fall" if label == 1 else "Normal"

    sample_id = sample["sample_id"]
    subtype = sample["subtype"]

    frames = clip.permute(0, 2, 3, 1).numpy()

    num_frames = frames.shape[0]
    cols = 4
    rows = (num_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")

        if i < num_frames:
            axes[i].imshow(frames[i])
            axes[i].set_title(f"Frame {i}", fontsize=9)

    fig.suptitle(
        f"{label_name} | subtype={subtype}\n{sample_id}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def find_first_samples(dataset, label_value: int, count: int = 5):
    samples = []

    for i in range(len(dataset)):
        sample = dataset[i]

        if int(sample["label"].item()) == label_value:
            samples.append(sample)

        if len(samples) >= count:
            break

    return samples


def main():
    train_dataset = FallVideoDataset(
        index_csv=INDEX_CSV,
        split="train",
        image_size=IMAGE_SIZE,
    )

    print("Train dataset size:", len(train_dataset))

    fall_samples = find_first_samples(train_dataset, label_value=1, count=5)
    normal_samples = find_first_samples(train_dataset, label_value=0, count=5)

    print("Saving Fall samples...")
    for idx, sample in enumerate(fall_samples):
        output_path = OUTPUT_DIR / f"fall_uniform_sample_{idx}.png"
        save_clip_grid(sample, output_path)
        print("Saved:", output_path)

    print("Saving Normal / hard negative samples...")
    for idx, sample in enumerate(normal_samples):
        output_path = OUTPUT_DIR / f"normal_hard_uniform_sample_{idx}.png"
        save_clip_grid(sample, output_path)
        print("Saved:", output_path)

    print("[DONE] Visualization completed.")


if __name__ == "__main__":
    main()