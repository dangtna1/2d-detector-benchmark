import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_eval(model_name: str, cfg: dict) -> dict:
    model = YOLO(model_name)
    start = time.time()
    train_results = model.train(
        data=cfg["dataset"],
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        device=cfg.get("device"),
        seed=cfg["seed"],
        project=cfg["project"],
        name=f'{cfg["name"]}/{Path(model_name).stem}',
        verbose=False,
    )
    train_time = time.time() - start

    val_results = model.val(
        data=cfg["dataset"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device"),
        project=cfg["project"],
        name=f'{cfg["name"]}/{Path(model_name).stem}',
        verbose=False,
    )

    metrics = {
        "model": model_name,
        "train_time_sec": round(train_time, 2),
        "map50": float(val_results.box.map50),
        "map50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "dataset": cfg["dataset"],
    }
    metrics["train_results_dir"] = str(train_results.save_dir)
    metrics["val_results_dir"] = str(val_results.save_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 vs YOLOv11 benchmark.")
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark config.",
    )
    parser.add_argument(
        "--out",
        default="runs/benchmarks/summary.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in cfg["models"]:
        rows.append(train_and_eval(model_name, cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
