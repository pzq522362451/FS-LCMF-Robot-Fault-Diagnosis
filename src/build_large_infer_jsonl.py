# 1. 读取 D:\MaBing\Data\images\test
# 2. 只筛选 S3L2：
#    C0S3L2 / C1S3L2 / C2S3L2 / C3S3L2 / C4S3L2
#
# 3. 按 batch_size=10 分批
# 4. 复制图片到：
#    D:\MaBing\Data\GPT_Image_Batches\test\C0S3L2\batch_0001\...
#
# 5. 文件命名和之前一样：
#    01_原始文件名.png
#    02_原始文件名.png
#    ...
#
# 6. 每个 batch 生成：
#    prompt.txt
#    metadata.json
#
# 7. 再合并生成：
#    D:\MaBing\Data\GPT_Reasoning\large_test_infer_images.jsonl

import json
import os
import pickle
import re
import shutil

import numpy as np
from scipy.stats import kurtosis


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BATCH_ROOT = os.path.join(DATA_DIR, "GPT_Image_Batches")
OUT_DIR = os.path.join(DATA_DIR, "GPT_Reasoning")

BATCH_SIZE = 10
SPLITS = ["train", "test"]
CONDITIONS = ["S1L2"]
CLASSES = ["C0", "C1", "C2", "C3", "C4"]
MAX_IMAGES_PER_SPLIT = {
    "train": 800,
    "test": 200,
}

# Same file naming style as batch_images_by_condition.py:
# copied_name = 01_original_name.png
# Set True only for strict no-leakage experiments.
ANONYMIZE_COPIED_NAME = False

FAULT_TEXT = {
    "C0": "正常工况",
    "C1": "第2轴RV40E-121行星轮齿裂纹故障",
    "C2": "第2轴RV40E-121行星轮齿断齿故障，全断齿",
    "C3": "第3轴RV20E-121太阳轮齿点蚀故障",
    "C4": "第3轴RV20E-121太阳轮齿断齿故障，全断齿",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_folder_name(name):
    match = re.match(r"^(C\d+)(S\d+L\d+)$", name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def list_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def load_processed_signals(folder_name, split):
    path = os.path.join(PROCESSED_DIR, f"{folder_name}_{split}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return np.asarray(payload["X"], dtype=np.float32)


def signal_features(signal):
    if signal is None:
        return None, None
    flat = np.asarray(signal, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(flat ** 2)))
    kurt = float(kurtosis(flat))
    return round(rms, 4), round(kurt, 4)


def prompt_header():
    return (
        "你是一个资深的机械故障诊断专家。我上传了多张STFT时频图，"
        "它们的文件名分别与下面的编号对应。\n"
        "这些样本用于大样本推理阶段，请根据图像和特征独立判断每个样本的故障类别。\n\n"
        "【核心强制原则】\n"
        "每个样本的分析必须是完全独立、自给自足的，不能进行跨样本对比。\n\n"
        "请严格模仿以下格式输出：\n\n"
        "样本 01：\n"
        "<think>你的独立推理过程...</think>\n"
        "A: 故障名称\n\n"
        "以下是这批数据的信息：\n"
    )


def build_prompt(fault_id, records):
    prompt = prompt_header()
    for record in records:
        if record["rms"] is None or record["kurtosis"] is None:
            feature_text = "RMS=缺失, Kurtosis=缺失"
        else:
            feature_text = f"RMS={record['rms']}, Kurtosis={record['kurtosis']}"

        prompt += (
            f"【样本 {record['sample_no']:02d} "
            f"(文件名: {record['copied_name']}, 数组索引: {record['array_index']})】\n"
            f"- 特征: {feature_text}\n"
            f"- 候选类别: C0, C1, C2, C3, C4\n\n"
        )
    return prompt


def should_keep_folder(folder_name):
    parsed = parse_folder_name(folder_name)
    if parsed is None:
        return False
    fault_id, condition = parsed
    if CONDITIONS and condition not in set(CONDITIONS):
        return False
    if CLASSES and fault_id not in set(CLASSES):
        return False
    return True


def make_batches_for_folder(split, folder_name, folder_path):
    parsed = parse_folder_name(folder_name)
    if parsed is None:
        return 0, []
    fault_id, condition = parsed

    images_all = list_images(folder_path)
    max_images = MAX_IMAGES_PER_SPLIT.get(split)
    images = images_all[:max_images] if max_images else images_all
    if not images:
        print(f"Skip empty folder: {folder_path}")
        return 0, []

    signals = load_processed_signals(folder_name, split)
    if max_images and len(images_all) > len(images):
        print(f"Limit: {split}/{folder_name} images={len(images_all)} -> {len(images)}")
    if signals is not None and len(signals) != len(images):
        print(f"Warning: {split}/{folder_name} selected_images={len(images)}, signals={len(signals)}")

    out_base = os.path.join(BATCH_ROOT, split, folder_name)
    os.makedirs(out_base, exist_ok=True)

    batch_count = 0
    merged_rows = []
    for start in range(0, len(images), BATCH_SIZE):
        batch_count += 1
        batch_images = images[start:start + BATCH_SIZE]
        batch_name = f"batch_{batch_count:04d}"
        batch_dir = os.path.join(out_base, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        records = []
        for offset, src_path in enumerate(batch_images):
            sample_no = offset + 1
            array_index = start + offset
            original_name = os.path.basename(src_path)
            copied_name = f"{sample_no:02d}_{original_name}"
            if ANONYMIZE_COPIED_NAME:
                copied_name = f"{sample_no:02d}_sample_{split}_{folder_name}_{array_index:06d}{os.path.splitext(original_name)[1].lower()}"

            dst_path = os.path.join(batch_dir, copied_name)
            shutil.copy2(src_path, dst_path)

            signal = signals[array_index] if signals is not None and array_index < len(signals) else None
            rms, kurt = signal_features(signal)

            record = {
                "sample_no": sample_no,
                "array_index": array_index,
                "original_name": original_name,
                "copied_name": copied_name,
                "source_path": src_path,
                "split": split,
                "folder": folder_name,
                "fault_id": fault_id,
                "fault_name": FAULT_TEXT.get(fault_id, fault_id),
                "condition": condition,
                "rms": rms,
                "kurtosis": kurt,
            }
            records.append(record)

        with open(os.path.join(batch_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(build_prompt(fault_id, records))

        with open(os.path.join(batch_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        for record in records:
            item = dict(record)
            item["batch"] = batch_name
            item["batch_dir"] = batch_dir
            item["response_file"] = ""
            item["reasoning_chain"] = ""
            item["reasoned_label"] = ""
            merged_rows.append(item)

    return batch_count, merged_rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_output_path(split):
    return os.path.join(OUT_DIR, f"large_{split}_infer_images.jsonl")


def main():
    os.makedirs(BATCH_ROOT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    total_batches = 0
    print("=" * 80)
    print("Build large-sample batches and infer jsonl")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Batch root: {BATCH_ROOT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Splits: {SPLITS}")
    print(f"Conditions: {CONDITIONS if CONDITIONS else 'all'}")
    print(f"Classes: {CLASSES if CLASSES else 'all'}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max images per split: {MAX_IMAGES_PER_SPLIT}")
    print(f"Anonymize copied_name: {ANONYMIZE_COPIED_NAME}")
    print("=" * 80)

    for split in SPLITS:
        split_dir = os.path.join(IMAGE_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"Missing split folder: {split_dir}")
            continue

        folder_names = sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name)) and should_keep_folder(name)
        )

        split_rows = []
        split_batches = 0
        for folder_name in folder_names:
            folder_path = os.path.join(split_dir, folder_name)
            batch_count, rows = make_batches_for_folder(split, folder_name, folder_path)
            total_batches += batch_count
            split_batches += batch_count
            all_rows.extend(rows)
            split_rows.extend(rows)
            print(f"{split}/{folder_name}: {batch_count} batches, {len(rows)} samples")

        out_path = split_output_path(split)
        write_jsonl(out_path, split_rows)
        print(f"{split}: {len(split_rows)} samples, {split_batches} batches -> {out_path}")

    all_path = os.path.join(OUT_DIR, "large_all_infer_images.jsonl")
    write_jsonl(all_path, all_rows)

    counts = {}
    for row in all_rows:
        key = f"{row['fault_id']}{row['condition']}"
        counts[key] = counts.get(key, 0) + 1

    print("=" * 80)
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {len(all_rows)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"Saved all jsonl: {all_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
