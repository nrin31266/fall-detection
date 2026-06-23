from pathlib import Path
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", default="data/features/rgb_efficientnet_lite0")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-dir", default="results/metrics")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    return parser.parse_args()


class FeatureDataset(Dataset):
    def __init__(self, feature_path: str | Path):
        payload = torch.load(feature_path, map_location="cpu")

        self.features = payload["features"].float()   # [N, T, D]
        self.labels = payload["labels"].long()         # [N]

        self.sample_ids = payload.get("sample_ids", None)
        self.split = payload.get("split", "unknown")
        self.feature_dim = int(payload["feature_dim"])
        self.clip_length = int(payload["clip_length"])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "label": self.labels[idx],
        }

        if self.sample_ids is not None:
            item["sample_id"] = self.sample_ids[idx]

        return item


class GRUFeatureClassifier(nn.Module):
    """
    GRU classifier on precomputed EfficientNet-Lite0 features.

    Input:
        features: [B, T, D]

    Output:
        logits: [B, 2]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features):
        gru_out, _ = self.gru(features)
        last_hidden = gru_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


def compute_class_weights(dataset: FeatureDataset):
    labels = dataset.labels

    normal_count = int((labels == 0).sum().item())
    fall_count = int((labels == 1).sum().item())
    total = len(dataset)

    weight_normal = total / (2 * normal_count)
    weight_fall = total / (2 * fall_count)

    weights = torch.tensor([weight_normal, weight_fall], dtype=torch.float32)

    print("Class counts:")
    print(f"  Normal: {normal_count}")
    print(f"  Fall:   {fall_count}")
    print("Class weights:")
    print(f"  Normal: {weight_normal:.4f}")
    print(f"  Fall:   {weight_fall:.4f}")

    return weights


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    for batch in progress:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(features)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_fall": precision,
        "recall_fall": recall,
        "f1_fall": f1,
        "confusion_matrix": cm,
    }


def save_checkpoint(model, optimizer, epoch, metrics, args, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def main():
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_path = feature_dir / "train_features.pt"
    val_path = feature_dir / "val_features.pt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train feature file: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Missing val feature file: {val_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Feature dir:", feature_dir)

    train_dataset = FeatureDataset(train_path)
    val_dataset = FeatureDataset(val_path)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Feature dim:", train_dataset.feature_dim)
    print("Clip length:", train_dataset.clip_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GRUFeatureClassifier(
        feature_dim=train_dataset.feature_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Epoch {epoch}/{args.epochs}")

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        result = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(result)

        print(
            f"Train | "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"precision_fall={train_metrics['precision_fall']:.4f} "
            f"recall_fall={train_metrics['recall_fall']:.4f} "
            f"f1_fall={train_metrics['f1_fall']:.4f}"
        )

        print(
            f"Val   | "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"precision_fall={val_metrics['precision_fall']:.4f} "
            f"recall_fall={val_metrics['recall_fall']:.4f} "
            f"f1_fall={val_metrics['f1_fall']:.4f}"
        )

        print("Val confusion matrix [[TN, FP], [FN, TP]]:")
        print(val_metrics["confusion_matrix"])

        current_f1 = val_metrics["f1_fall"]

        latest_path = checkpoint_dir / "latest_gru_on_rgb_features.pt"
        save_checkpoint(model, optimizer, epoch, val_metrics, args, latest_path)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_path = checkpoint_dir / "best_gru_on_rgb_features.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, args, best_path)
            print(f"[SAVE] Best checkpoint saved to {best_path}")

        history_path = metrics_dir / "gru_on_rgb_features_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print()
    print("[DONE] Training completed.")
    print(f"Best val F1 Fall: {best_f1:.4f}")
    print(f"Total time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()