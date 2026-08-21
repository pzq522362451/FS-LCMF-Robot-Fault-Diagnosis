import csv
import importlib
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from tqdm import tqdm

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", torch.uint8)

from transformers import AutoProcessor, AutoTokenizer

sys.path.append(os.path.dirname(__file__))

from simple_lora import inject_lora


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data1")
REASONING_DIR = os.path.join(DATA_DIR, "GPT_Reasoning")

MODEL_PATH = os.path.join(ROOT_DIR, "MainFunc", "Qwen2.5-Omni-7B")
WEIGHT_PATH = os.path.join(DATA_DIR, "qwen_reasoning_lora_checkpoints_S2S3L2", "final", "lora_weights.pt")

EVAL_SETS = [
    ("train", os.path.join(REASONING_DIR, "train_reasoning_chains.jsonl")),
    ("test", os.path.join(REASONING_DIR, "test_reasoning_chains.jsonl")),
]
EVAL_CONDITIONS = ["S1L2"]
EVAL_CLASSES = ["C0", "C1", "C2", "C3", "C4"]
ACTIVE_FAULT_IDS = ["C0", "C1", "C2", "C3", "C4"]

SUMMARY_CSV = os.path.join(REASONING_DIR, "qwen_reasoning_lora_eval_summary.csv")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_NEW_TOKENS = 96

FAULT_NAME = {
    "C0": "正常工况",
    "C1": "第2轴RV40E-121行星轮齿裂纹故障",
    "C2": "第2轴RV40E-121行星轮齿断齿故障，全断齿",
    "C3": "第3轴RV20E-121太阳轮齿点蚀故障",
    "C4": "第3轴RV20E-121太阳轮齿断齿故障，全断齿",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_records(records, conditions=None, classes=None):
    conditions = set(conditions or [])
    classes = set(classes or [])
    filtered = []
    for item in records:
        if conditions and item.get("condition") not in conditions:
            continue
        if classes and str(item.get("fault_id", "")).upper() not in classes:
            continue
        filtered.append(item)
    return filtered


def load_processor(model_path):
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return processor, True
    except Exception as exc:
        print(f"AutoProcessor failed, use AutoTokenizer instead: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, False


def load_model(model_path):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if has_accelerate:
        load_kwargs["device_map"] = "auto"

    omni_module = importlib.import_module("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
    model_cls = getattr(omni_module, "Qwen2_5OmniThinkerForConditionalGeneration")
    print("Loading Qwen2.5-Omni Thinker model")
    model = model_cls.from_pretrained(model_path, **load_kwargs)
    if not has_accelerate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    return model


def load_lora_weights(model, weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"LoRA weights not found: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded LoRA weights: {weight_path}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def resolve_image_path(item):
    path = item.get("source_path") or item.get("image_path")
    if path is None:
        raise KeyError("Missing source_path or image_path in jsonl item")
    if os.path.isabs(path):
        return path
    return os.path.join(DATA_DIR, path)


def build_prompt(item, processor):
    image_path = resolve_image_path(item)
    copied_name = item.get("copied_name", os.path.basename(image_path))
    condition = item.get("condition", "未知工况")
    rms = item.get("rms", "缺失")
    kurtosis = item.get("kurtosis", "缺失")
    image_token = getattr(processor, "image_token", "<image>")

    system_prompt = "你是一个资深的机械故障诊断专家，能够根据RV减速器STFT时频图和统计特征判断故障类别。"
    candidate_text = "\n".join(f"{cid} {FAULT_NAME[cid]}" for cid in ACTIVE_FAULT_IDS)
    user_msg = (
        f"{image_token}\n"
        "请根据该样本进行故障诊断。注意：不能使用真实标签，只能根据图像和特征推理。\n"
        f"文件名：{copied_name}\n"
        f"工况：{condition}\n"
        f"统计特征：RMS={rms}，Kurtosis={kurtosis}\n"
        f"候选类别包括：\n{candidate_text}\n"
        "请严格按以下格式输出：\n"
        "第一行只写 A: C?，例如 A: C0。\n"
        "第二行开始再用 <think>...</think> 给出简短推理过程。"
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def encode_inputs(processor, supports_image, prompt_text, image):
    if supports_image:
        try:
            return processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
        except TypeError:
            return processor([prompt_text], images=[image], padding=True, return_tensors="pt")
    try:
        return processor(text=[prompt_text], padding=True, return_tensors="pt")
    except TypeError:
        return processor([prompt_text], padding=True, return_tensors="pt")


def get_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def move_to_device(inputs, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def decode_output(processor, generated_ids, input_len):
    tokenizer = getattr(processor, "tokenizer", processor)
    new_tokens = generated_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_pred_label(text):
    patterns = [
        r"^\s*A\s*[:：]\s*(C[0-4])",
        r"A\s*[:：]\s*(C[0-4])",
        r"答案\s*[:：]\s*(C[0-4])",
        r"故障类别\s*[:：]\s*(C[0-4])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    for cid, cname in FAULT_NAME.items():
        if cid in text or cname in text:
            return cid
    return ""


def compute_metrics(rows):
    labels = EVAL_CLASSES if EVAL_CLASSES else ["C0", "C1", "C2", "C3", "C4"]
    idx = {name: i for i, name in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    total = 0
    correct = 0
    invalid = 0

    for row in rows:
        true_label = str(row.get("fault_id", "")).upper()
        pred_label = str(row.get("pred_label", "")).upper()
        if true_label not in idx:
            continue
        total += 1
        if pred_label in idx:
            cm[idx[true_label]][idx[pred_label]] += 1
        else:
            invalid += 1
        correct += int(true_label == pred_label)

    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(labels)):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "labels": labels,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precision_list) / len(precision_list),
        "macro_recall": sum(recall_list) / len(recall_list),
        "macro_f1": sum(f1_list) / len(f1_list),
        "confusion_matrix": cm,
    }


def output_path(split_name):
    condition_tag = "all_conditions" if not EVAL_CONDITIONS else "_".join(EVAL_CONDITIONS)
    class_tag = "all_classes" if not EVAL_CLASSES else "_".join(EVAL_CLASSES)
    return os.path.join(REASONING_DIR, f"qwen_reasoning_lora_infer_{split_name}_{condition_tag}_{class_tag}.jsonl")


def run_eval_set(split_name, jsonl_path, processor, supports_image, model, device):
    records_all = read_jsonl(jsonl_path)
    records = filter_records(records_all, EVAL_CONDITIONS, EVAL_CLASSES)
    if len(records) == 0:
        raise RuntimeError(f"No samples after filtering for {split_name}.")

    print("-" * 80)
    print(f"Eval split: {split_name}")
    print(f"Sample filter: {len(records_all)} -> {len(records)}")

    results = []
    start = time.perf_counter()
    with torch.no_grad():
        for item in tqdm(records, desc=f"Infer {split_name}"):
            image = Image.open(resolve_image_path(item)).convert("RGB")
            prompt_text = build_prompt(item, processor)
            inputs = encode_inputs(processor, supports_image, prompt_text, image)
            inputs = move_to_device(inputs, device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            output_text = decode_output(processor, generated_ids, input_len)
            pred_label = extract_pred_label(output_text)

            out_item = dict(item)
            out_item["prompt_text"] = prompt_text
            out_item["qwen_output"] = output_text
            out_item["pred_label"] = pred_label
            out_item["pred_label_name"] = FAULT_NAME.get(pred_label, "")
            results.append(out_item)

    elapsed = time.perf_counter() - start
    out_jsonl = output_path(split_name)
    write_jsonl(out_jsonl, results)
    metrics = compute_metrics(results)

    print(f"Output: {out_jsonl}")
    print(f"Time: {elapsed:.2f}s")
    print(
        f"{split_name} accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['macro_precision']:.4f}, "
        f"recall={metrics['macro_recall']:.4f}, "
        f"f1={metrics['macro_f1']:.4f}, "
        f"invalid={metrics['invalid']}"
    )
    print(f"Confusion matrix rows=true, cols=pred, labels={metrics['labels']}:")
    for row in metrics["confusion_matrix"]:
        print(row)

    return {
        "split": split_name,
        "jsonl": jsonl_path,
        "output_jsonl": out_jsonl,
        "conditions": "|".join(EVAL_CONDITIONS) if EVAL_CONDITIONS else "all",
        "classes": "|".join(EVAL_CLASSES) if EVAL_CLASSES else "all",
        "total": metrics["total"],
        "correct": metrics["correct"],
        "invalid": metrics["invalid"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "elapsed_sec": elapsed,
    }


def save_summary(rows):
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    fieldnames = [
        "split",
        "jsonl",
        "output_jsonl",
        "conditions",
        "classes",
        "total",
        "correct",
        "invalid",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "elapsed_sec",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 80)
    print("Qwen reasoning-chain evaluation with plain LoRA")
    print(f"Model path: {MODEL_PATH}")
    print(f"LoRA weights: {WEIGHT_PATH}")
    print(f"Eval conditions: {EVAL_CONDITIONS if EVAL_CONDITIONS else 'all'}")
    print(f"Eval classes: {EVAL_CLASSES if EVAL_CLASSES else 'all'}")
    print("=" * 80)

    processor, supports_image = load_processor(MODEL_PATH)
    print(f"Using image processor: {supports_image}")

    model = load_model(MODEL_PATH)
    inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )
    load_lora_weights(model, WEIGHT_PATH)
    model.eval()

    device = get_device(model)
    print(f"Inference device: {device}")

    summaries = []
    for split_name, jsonl_path in EVAL_SETS:
        summaries.append(run_eval_set(split_name, jsonl_path, processor, supports_image, model, device))

    save_summary(summaries)
    print("=" * 80)
    print(f"Summary saved to: {SUMMARY_CSV}")
    print("Evaluation finished")


if __name__ == "__main__":
    main()
