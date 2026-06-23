from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from video_dataset import FallVideoDataset
from model_efficientnet_lite_gru import EfficientNetLiteGRU


# =========================
# Config
# =========================

INDEX_CSV = "data/video_index_16.csv"

CHECKPOINT_DIR = Path("checkpoints")
METRICS_DIR = Path("results/metrics")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tf_efficientnet_lite0"

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 2

EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 128
DROPOUT = 0.3

FREEZE_BACKBONE = True
PRETRAINED = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utility
# =========================

def compute_class_weights(dataset: FallVideoDataset):
    """
    Tính class weight từ train set.
    Normal = 0
    Fall = 1
    """
    labels = dataset.df["label"].map({"Normal": 0, "Fall": 1}).values

    normal_count = (labels == 0).sum()
    fall_count = (labels == 1).sum()
    total = len(labels)

    # weight càng lớn nếu class càng ít
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


def run_one_epoch(model, dataloader, criterion, optimizer=None):
    """
    Nếu optimizer is None => validation mode.
    Nếu optimizer != None => train mode.
    """
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress = tqdm(
        dataloader,
        desc="Train" if is_train else "Val",
        leave=False,
    )

    for batch in progress:
        clips = batch["clip"].to(DEVICE)      # [B, 16, 3, 224, 224]
        labels = batch["label"].to(DEVICE)    # [B]

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(clips)             # [B, 2]
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_fall": precision,
        "recall_fall": recall,
        "f1_fall": f1,
        "confusion_matrix": cm,
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "freeze_backbone": FREEZE_BACKBONE,
                "pretrained": PRETRAINED,
            },
        },
        path,
    )


# =========================
# Main
# =========================

def main():
    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Freeze backbone:", FREEZE_BACKBONE)
    print("Pretrained:", PRETRAINED)

    train_dataset = FallVideoDataset(
        index_csv=INDEX_CSV,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = FallVideoDataset(
        index_csv=INDEX_CSV,
        split="val",
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    class_weights = compute_class_weights(train_dataset).to(DEVICE)

    model = EfficientNetLiteGRU(
        model_name=MODEL_NAME,
        num_classes=2,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
        )

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(epoch_result)

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

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_path = CHECKPOINT_DIR / "best_efficientnet_lite_gru.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=best_path,
            )

            print(f"[SAVE] Best checkpoint saved to {best_path}")

        latest_path = CHECKPOINT_DIR / "latest_efficientnet_lite_gru.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            path=latest_path,
        )

        history_path = METRICS_DIR / "efficientnet_lite_gru_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print()
    print("[DONE] Training completed.")
    print(f"Best val F1 Fall: {best_f1:.4f}")
    print(f"Total time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()