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
        verbose=True,
    )
    train_time = time.time() - start

    eval_split = cfg.get("eval_split", "val")
    eval_results = model.val(
        data=cfg["dataset"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device"),
        split=eval_split,
        project=cfg["project"],
        name=f'{cfg["name"]}/{Path(model_name).stem}/{eval_split}',
        verbose=True,
    )

    box = eval_results.box
    f1 = 0.0
    if (box.mp + box.mr) > 0:
        f1 = (2 * box.mp * box.mr) / (box.mp + box.mr)

    def to_list(values):
        if values is None:
            return None
        try:
            return [float(v) for v in values]
        except TypeError:
            return None

    names = eval_results.names or {}
    ap = to_list(getattr(box, "ap", None))
    ap50 = to_list(getattr(box, "ap50", None))
    per_class_p = to_list(getattr(box, "p", None))
    per_class_r = to_list(getattr(box, "r", None))
    per_class_f1 = None
    if per_class_p is not None and per_class_r is not None:
        per_class_f1 = [
            (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            for p, r in zip(per_class_p, per_class_r)
        ]

    def by_name(values):
        if values is None or not names:
            return None
        return {names[i]: values[i] for i in range(len(values)) if i in names}

    metrics = {
        "model": model_name,
        "train_time_sec": round(train_time, 2),
        "eval_split": eval_split,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": float(f1),
        "per_class_ap50_95": by_name(ap),
        "per_class_ap50": by_name(ap50),
        "per_class_precision": by_name(per_class_p),
        "per_class_recall": by_name(per_class_r),
        "per_class_f1": by_name(per_class_f1),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "dataset": cfg["dataset"],
    }
    metrics["train_results_dir"] = str(train_results.save_dir)
    metrics["eval_results_dir"] = str(eval_results.save_dir)
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
