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
OUTPUT_DIR = os.path.join(DATA_DIR, "GPT_Image_Batches")

BATCH_SIZE = 10
SPLIT_TEXT = {"train": "训练集", "test": "测试集"}

FAULT_TEXT = {
    "C0": "正常工况",
    "C1": "第2轴RV40E-121行星轮齿裂纹故障",
    "C2": "第2轴RV40E-121行星轮齿断齿故障，全断齿",
    "C3": "第3轴RV20E-121太阳轮齿点蚀故障",
    "C4": "第3轴RV20E-121太阳轮齿断齿故障，全断齿",
}


def parse_folder_name(name):
    match = re.match(r"^(C\d+)(S\d+L\d+)$", name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
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
        "你是一个资深的机械故障诊断专家。我上传了多张CWT时频图，它们的文件名分别与下面的编号对应。\n"
        "我已经确切知道这些样本的真实故障类型。请你结合图像和特征，为每个样本生成推理过程。\n\n"
        "【核心强制原则】\n"
        "每个样本的分析必须是**完全独立、自给自足**的！这些数据未来会被彻底打散，用于训练只能单图输入的视觉大模型。\n"
        "因此，你**绝对不能**在回答中进行跨样本对比，**严禁使用**“与上一个样本类似”、“同样本01一样”、“如前所述”等词汇！"
        "每个样本都要假装是你今天看到的第一个样本，给出完整的物理推演。\n\n"
        "请严格模仿以下格式输出（必须包含 <think> 标签和 A: 答案）：\n\n"
        "样本 01：\n"
        "<think>你的独立推理过程...</think>\n"
        "A: 故障名称\n\n"
        "以下是这批数据的信息：\n"
    )


def build_prompt(fault_id, records):
    prompt = prompt_header()
    for record in records:
        if record["rms"] is None or record["kurtosis"] is None:
            feature_text = "均方根=缺失, 峭度=缺失"
        else:
            feature_text = f"均方根={record['rms']}, 峭度={record['kurtosis']}"

        prompt += (
            f"【样本 {record['sample_no']:02d} (文件名: {record['copied_name']}, 数组索引: {record['array_index']})】\n"
            f"- 特征: {feature_text}\n"
            f"- 真实标签: {FAULT_TEXT.get(fault_id, fault_id)}\n\n"
        )
    return prompt


def make_batches_for_folder(split, folder_name, folder_path):
    parsed = parse_folder_name(folder_name)
    if parsed is None:
        return 0
    fault_id, condition = parsed

    images = list_images(folder_path)
    if not images:
        print(f"跳过空文件夹: {folder_path}")
        return 0

    signals = load_processed_signals(folder_name, split)
    if signals is not None and len(signals) != len(images):
        print(f"警告: {split}/{folder_name} 图像数={len(images)}，信号数={len(signals)}")

    out_base = os.path.join(OUTPUT_DIR, split, folder_name)
    os.makedirs(out_base, exist_ok=True)

    batch_count = 0
    for start in range(0, len(images), BATCH_SIZE):
        batch_count += 1
        batch_images = images[start:start + BATCH_SIZE]
        batch_dir = os.path.join(out_base, f"batch_{batch_count:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        records = []
        for offset, src_path in enumerate(batch_images):
            sample_no = offset + 1
            array_index = start + offset
            original_name = os.path.basename(src_path)
            copied_name = f"{sample_no:02d}_{original_name}"
            dst_path = os.path.join(batch_dir, copied_name)
            shutil.copy2(src_path, dst_path)

            signal = signals[array_index] if signals is not None and array_index < len(signals) else None
            rms, kurt = signal_features(signal)

            records.append({
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
            })

        with open(os.path.join(batch_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(build_prompt(fault_id, records))

        with open(os.path.join(batch_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return batch_count


def main():
    print("=" * 80)
    print("按故障类别-工况文件夹分批整理图像")
    print(f"输入目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每批图像数: {BATCH_SIZE}")
    print("=" * 80)

    total_batches = 0
    for split in ["train", "test"]:
        split_dir = os.path.join(IMAGE_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"缺少数据集文件夹: {split_dir}")
            continue

        folder_names = sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name)) and parse_folder_name(name) is not None
        )

        for folder_name in folder_names:
            folder_path = os.path.join(split_dir, folder_name)
            count = make_batches_for_folder(split, folder_name, folder_path)
            total_batches += count
            print(f"{SPLIT_TEXT.get(split, split)}/{folder_name}: {count} 个批次")

    print("=" * 80)
    print(f"分批完成。总批次数: {total_batches}")
    print("=" * 80)


if __name__ == "__main__":
    main()
