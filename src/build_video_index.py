from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INVENTORY_CSV = Path("data/video_inventory.csv")
OUTPUT_CSV = Path("data/video_index_16.csv")

CLIP_LENGTH = 16
RANDOM_STATE = 42


def split_by_video(df: pd.DataFrame) -> pd.DataFrame:
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        stratify=train_val["label"],
        random_state=RANDOM_STATE,
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)


def main():
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"Missing file: {INVENTORY_CSV}")

    df = pd.read_csv(INVENTORY_CSV)
    df = df[df["readable"] == True].copy()

    # Giữ lại toàn bộ video đọc được.
    # Dataset này video ngắn, nên 1 video = 1 sample là hợp lý.
    df = split_by_video(df)

    rows = []

    for _, row in df.iterrows():
        video_stem = Path(row["video_path"]).stem

        rows.append({
            "sample_id": f"{video_stem}_uniform16",
            "video_path": row["video_path"],
            "filename": row["filename"],
            "label": row["label"],
            "subtype": row["subtype"],
            "dataset": row["dataset"],
            "split": row["split"],
            "fps": row["fps"],
            "frames": row["frames"],
            "duration_sec": row["duration_sec"],
            "width": row["width"],
            "height": row["height"],
            "clip_length": CLIP_LENGTH,
            "sampling_strategy": "uniform_full_video",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Saved video index to: {OUTPUT_CSV}")
    print("Total samples:", len(out_df))

    print()
    print("Samples by split:")
    print(out_df.groupby(["split", "label", "subtype"]).size())


if __name__ == "__main__":
    main()