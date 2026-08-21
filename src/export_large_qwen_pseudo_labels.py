import csv
import json
import os

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
REASONING_DIR = os.path.join(DATA_DIR, "GPT_Reasoning")
OUT_DIR = os.path.join(DATA_DIR, "features")

LABEL_TO_ID = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABELS = ["C0", "C1", "C2", "C3", "C4"]

SPLITS = {
    "train": os.path.join(
        REASONING_DIR,
        "qwen_reasoning_lora_large_infer_train_S1L2_C0_C1_C2_C3_C4.jsonl",
    ),
    "test": os.path.join(
        REASONING_DIR,
        "qwen_reasoning_lora_large_infer_test_S1L2_C0_C1_C2_C3_C4.jsonl",
    ),
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(true_ids, pred_ids):
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    invalid = 0
    for true_id, pred_id in zip(true_ids, pred_ids):
        if 0 <= true_id < len(LABELS) and 0 <= pred_id < len(LABELS):
            cm[true_id, pred_id] += 1
        else:
            invalid += 1

    total = int(len(true_ids))
    correct = int(np.trace(cm))
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(LABELS)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_list.append(float(precision))
        recall_list.append(float(recall))
        f1_list.append(float(f1))

    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(np.mean(precision_list)),
        "macro_recall": float(np.mean(recall_list)),
        "macro_f1": float(np.mean(f1_list)),
        "confusion_matrix": cm,
    }


def export_split(split_name, jsonl_path):
    rows = read_jsonl(jsonl_path)
    true_ids = []
    pseudo_ids = []
    csv_rows = []

    for idx, row in enumerate(rows):
        true_label = str(row.get("fault_id", "")).upper()
        pred_label = str(row.get("pred_label", "")).upper()
        true_id = LABEL_TO_ID.get(true_label, -1)
        pseudo_id = LABEL_TO_ID.get(pred_label, -1)
        true_ids.append(true_id)
        pseudo_ids.append(pseudo_id)
        csv_rows.append({
            "index": idx,
            "sample_no": row.get("sample_no", ""),
            "array_index": row.get("array_index", ""),
            "folder": row.get("folder", ""),
            "condition": row.get("condition", ""),
            "source_path": row.get("source_path", ""),
            "true_label": true_label,
            "true_id": true_id,
            "pseudo_label": pred_label,
            "pseudo_id": pseudo_id,
            "is_correct": int(true_id == pseudo_id),
        })

    true_ids = np.asarray(true_ids, dtype=np.int64)
    pseudo_ids = np.asarray(pseudo_ids, dtype=np.int64)
    metrics = compute_metrics(true_ids, pseudo_ids)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"qwen_large_{split_name}_pseudo_labels.npy"), pseudo_ids)
    np.save(os.path.join(OUT_DIR, f"qwen_large_{split_name}_true_labels_for_eval.npy"), true_ids)

    csv_path = os.path.join(REASONING_DIR, f"qwen_large_{split_name}_pseudo_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path, metrics


def main():
    summary_rows = []
    print("=" * 80)
    print("Export Qwen large-sample pseudo labels")
    print(f"Reasoning dir: {REASONING_DIR}")
    print(f"Output feature dir: {OUT_DIR}")
    print("=" * 80)

    for split_name, jsonl_path in SPLITS.items():
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Missing inference result jsonl: {jsonl_path}")

        csv_path, metrics = export_split(split_name, jsonl_path)
        print("-" * 80)
        print(f"Split: {split_name}")
        print(f"Input: {jsonl_path}")
        print(f"CSV: {csv_path}")
        print(f"Pseudo labels npy: {os.path.join(OUT_DIR, f'qwen_large_{split_name}_pseudo_labels.npy')}")
        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['macro_precision']:.4f}, "
            f"recall={metrics['macro_recall']:.4f}, "
            f"f1={metrics['macro_f1']:.4f}, "
            f"invalid={metrics['invalid']}"
        )
        print(f"Confusion matrix rows=true, cols=pseudo, labels={LABELS}:")
        for row in metrics["confusion_matrix"].tolist():
            print(row)

        summary_rows.append({
            "split": split_name,
            "total": metrics["total"],
            "correct": metrics["correct"],
            "invalid": metrics["invalid"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "pseudo_npy": os.path.join(OUT_DIR, f"qwen_large_{split_name}_pseudo_labels.npy"),
            "true_eval_npy": os.path.join(OUT_DIR, f"qwen_large_{split_name}_true_labels_for_eval.npy"),
            "csv": csv_path,
        })

    summary_path = os.path.join(REASONING_DIR, "qwen_large_pseudo_label_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 80)
    print(f"Summary saved to: {summary_path}")
    print("Done")


if __name__ == "__main__":
    main()
