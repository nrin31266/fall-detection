from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
OUTPUT_CSV = Path("data/video_inventory.csv")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def inspect_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "readable": False,
            "width": None,
            "height": None,
            "fps": None,
            "frames": None,
            "duration_sec": None,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    if fps is None or fps <= 0:
        duration = None
    else:
        duration = frames / fps

    return {
        "readable": True,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "fps": round(float(fps), 3) if fps else None,
        "frames": int(frames) if frames else None,
        "duration_sec": round(float(duration), 3) if duration else None,
    }


def collect_videos():
    rows = []

    folders = [
        ("fall", "Fall", "fall"),
        ("hard_negative", "Normal", "hard_negative"),
        ("normal", "Normal", "normal"),
    ]

    for folder_name, label, subtype in folders:
        folder = RAW_DIR / folder_name

        if not folder.exists():
            continue

        video_paths = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]

        for video_path in tqdm(video_paths, desc=f"Inspecting {folder_name}"):
            info = inspect_video(video_path)

            rows.append({
                "video_path": str(video_path),
                "label": label,
                "subtype": subtype,
                "dataset": "fall_video_dataset",
                "filename": video_path.name,
                **info,
            })

    return rows


def main():
    rows = collect_videos()
    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Saved inventory to: {OUTPUT_CSV}")
    print()
    print("Total videos:", len(df))

    if len(df) > 0:
        print()
        print("By label:")
        print(df.groupby(["label", "subtype"]).size())

        print()
        print("Readable:")
        print(df["readable"].value_counts())

        print()
        print("Duration summary:")
        print(df.groupby(["label", "subtype"])["duration_sec"].describe())

        print()
        print("FPS summary:")
        print(df.groupby(["label", "subtype"])["fps"].describe())


if __name__ == "__main__":
    main()