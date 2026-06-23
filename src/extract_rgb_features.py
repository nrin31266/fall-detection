from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from video_dataset import FallVideoDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index-csv", default="data/video_index_16.csv")
    parser.add_argument("--output-dir", default="data/features/rgb_efficientnet_lite0")
    parser.add_argument("--model-name", default="tf_efficientnet_lite0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)

    return parser.parse_args()


class EfficientNetFeatureExtractor(torch.nn.Module):
    """
    Extract frame-level features from EfficientNet-Lite0.

    Input:
        clips: [B, T, C, H, W], pixel range [0, 1]

    Output:
        features: [B, T, D]
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim = self.backbone.num_features

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, clips):
        b, t, c, h, w = clips.shape

        x = clips.view(b * t, c, h, w)
        x = (x - self.mean) / self.std

        features = self.backbone(x)
        features = features.view(b, t, -1)

        return features


def extract_split(args, split: str, device: str):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{split}_features.pt"

    if output_path.exists() and not args.force:
        print(f"[SKIP] {output_path} already exists. Use --force to overwrite.")
        return

    dataset = FallVideoDataset(
        index_csv=args.index_csv,
        split=split,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    model = EfficientNetFeatureExtractor(
        model_name=args.model_name,
        pretrained=bool(args.pretrained),
    ).to(device)

    model.eval()

    all_features = []
    all_labels = []
    all_sample_ids = []
    all_video_paths = []
    all_subtypes = []

    print()
    print(f"[INFO] Extracting split: {split}")
    print(f"[INFO] Samples: {len(dataset)}")
    print(f"[INFO] Output: {output_path}")

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"Extract {split}")

        for batch_idx, batch in enumerate(progress):
            clips = batch["clip"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                features = model(clips)

            features = features.detach().cpu().half()

            all_features.append(features)
            all_labels.append(batch["label"].cpu())
            all_sample_ids.extend(list(batch["sample_id"]))
            all_video_paths.extend(list(batch["video_path"]))
            all_subtypes.extend(list(batch["subtype"]))

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                print(f"[DEBUG] Stop early at max_batches={args.max_batches}")
                break

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0).long()

    payload = {
        "features": features_tensor,          # [N, 16, D], float16
        "labels": labels_tensor,              # [N]
        "sample_ids": all_sample_ids,
        "video_paths": all_video_paths,
        "subtypes": all_subtypes,
        "split": split,
        "model_name": args.model_name,
        "feature_dim": features_tensor.shape[-1],
        "clip_length": features_tensor.shape[1],
        "image_size": args.image_size,
        "pretrained": bool(args.pretrained),
    }

    torch.save(payload, output_path)

    print(f"[DONE] Saved {split} features to {output_path}")
    print("Feature shape:", features_tensor.shape)
    print("Labels shape:", labels_tensor.shape)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Model:", args.model_name)
    print("Pretrained:", bool(args.pretrained))
    print("Output dir:", args.output_dir)

    for split in args.splits:
        extract_split(args, split, device)


if __name__ == "__main__":
    main()