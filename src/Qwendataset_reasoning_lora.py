import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", torch.uint8)

from transformers import AutoTokenizer

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None


FAULT_ID = {name: idx for idx, name in enumerate(["C0", "C1", "C2", "C3", "C4"])}
EXPERT_CLUSTER = {
    "C0": 0,
    "C1": 1,
    "C2": 1,
    "C3": 2,
    "C4": 2,
}

FAULT_NAME = {
    "C0": "正常工况",
    "C1": "第2轴RV40E-121行星轮齿裂纹故障",
    "C2": "第2轴RV40E-121行星轮齿断齿故障，全断齿",
    "C3": "第3轴RV20E-121太阳轮齿点蚀故障",
    "C4": "第3轴RV20E-121太阳轮齿断齿故障，全断齿",
}

ACTIVE_FAULT_IDS = ["C0", "C1", "C2", "C3", "C4"]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_processor_or_tokenizer(model_path):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            return processor, True
        except Exception as exc:
            print(f"AutoProcessor failed, use AutoTokenizer instead: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, False


def normalize_reasoning_target(reasoning_chain, answer_label):
    chain = (reasoning_chain or "").strip()

    if "<think>" in chain and "</think>" in chain:
        match = re.search(r"<think>(.*?)</think>", chain, flags=re.DOTALL)
        reasoning = match.group(1).strip() if match else ""
    else:
        answer_match = re.search(r"A\s*[:：]", chain)
        reasoning = chain[:answer_match.start()].strip() if answer_match else chain

    if not reasoning:
        reasoning = "根据时频图能量分布、冲击特征和统计特征进行故障诊断推理。"

    # Classification-first target: make the category token appear first.
    return f"A: {answer_label}\n<think>{reasoning}</think>"


class QwenReasoningLoRADataset(Dataset):
    """
    Dataset for plain LoRA supervised reasoning-chain tuning.

    The true label is used only as the target answer, not as an input field.
    This keeps training and inference prompts consistent.
    """

    def __init__(self, jsonl_path, model_path, base_dir=None, max_length=1024):
        super().__init__()
        self.data = read_jsonl(jsonl_path)
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(jsonl_path), ".."))
        self.max_length = max_length
        self.processor, self.supports_image = load_processor_or_tokenizer(model_path)

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, item):
        path = item.get("source_path") or item.get("image_path")
        if path is None:
            raise KeyError("jsonl item must contain source_path or image_path")
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = self._resolve_image_path(item)
        image = Image.open(image_path).convert("RGB")

        fault_id = str(item.get("fault_id", item.get("reasoned_label", "C0"))).upper()
        condition = str(item.get("condition", "未知工况"))
        label_id = FAULT_ID.get(fault_id, 0)
        expert_cluster_id = int(item.get("expert_cluster_id", EXPERT_CLUSTER.get(fault_id, 0)))

        rms = item.get("rms", "缺失")
        kurtosis_value = item.get("kurtosis", "缺失")
        copied_name = item.get("copied_name", os.path.basename(image_path))

        reasoning_target = normalize_reasoning_target(item.get("reasoning_chain", ""), fault_id)
        image_token = getattr(self.processor, "image_token", "<image>")

        system_prompt = "你是一个资深的机械故障诊断专家，能够根据RV减速器STFT时频图和统计特征判断故障类别。"
        candidate_text = "\n".join(f"{cid} {FAULT_NAME[cid]}" for cid in ACTIVE_FAULT_IDS)
        user_msg = (
            f"{image_token}\n"
            "请根据该样本进行故障诊断。注意：输入中没有真实标签，只能根据图像和特征推理。\n"
            f"文件名：{copied_name}\n"
            f"工况：{condition}\n"
            f"统计特征：RMS={rms}，Kurtosis={kurtosis_value}\n"
            f"候选类别包括：\n{candidate_text}\n"
            "请严格按以下格式输出：\n"
            "第一行只写 A: C?，例如 A: C0。\n"
            "第二行开始再用 <think>...</think> 给出简短推理过程。"
        )

        prompt_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_text + f"{reasoning_target}<|im_end|>\n"

        return {
            "prompt_text": prompt_text,
            "full_text": full_text,
            "image": image,
            "label_id": label_id,
            "fault_id": fault_id,
            "condition": condition,
            "expert_cluster_id": expert_cluster_id,
            "source_path": image_path,
        }


class QwenReasoningLoRACollator:
    def __init__(self, processor, supports_image=False, max_length=1024):
        self.processor = processor
        self.supports_image = supports_image
        self.max_length = max_length
        self.ignore_index = -100

        if hasattr(self.processor, "pad_token") and self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

    def _encode(self, texts, images=None):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }

        if self.supports_image and images is not None:
            try:
                return self.processor(text=texts, images=images, **kwargs)
            except TypeError:
                return self.processor(texts, images=images, **kwargs)

        try:
            return self.processor(text=texts, **kwargs)
        except TypeError:
            return self.processor(texts, **kwargs)

    def __call__(self, features):
        full_texts = [item["full_text"] for item in features]
        prompt_texts = [item["prompt_text"] for item in features]
        images = [item["image"] for item in features]
        label_ids = [item["label_id"] for item in features]
        expert_cluster_ids = [item["expert_cluster_id"] for item in features]

        inputs = self._encode(full_texts, images)
        prompt_inputs = self._encode(prompt_texts, images)

        labels = inputs["input_ids"].clone()
        for i in range(labels.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = self.ignore_index
            labels[i, inputs["attention_mask"][i] == 0] = self.ignore_index

        inputs["labels"] = labels
        inputs["label_id"] = torch.tensor(label_ids, dtype=torch.long)
        inputs["expert_cluster_id"] = torch.tensor(expert_cluster_ids, dtype=torch.long)
        return inputs
