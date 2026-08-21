import csv
import importlib
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", torch.uint8)

from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.dirname(__file__))

from Qwendataset_reasoning_lora import QwenReasoningLoRACollator, QwenReasoningLoRADataset
from simple_lora import inject_lora, print_trainable_parameters


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data1")
MODEL_PATH = os.path.join(ROOT_DIR, "MainFunc", "Qwen2.5-Omni-7B")
TRAIN_JSONL = os.path.join(DATA_DIR, "GPT_Reasoning", "train_reasoning_chains.jsonl")
SAVE_ROOT = os.path.join(DATA_DIR, "qwen_reasoning_lora_checkpoints_S2S3L2")
LOG_CSV = os.path.join(SAVE_ROOT, "training_log.csv")

# Five-class supervised baseline for reasoning-chain training.
TRAIN_CONDITIONS = ["S2L2", "S3L2"]
TRAIN_CLASSES = ["C0", "C1", "C2", "C3", "C4"]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 1024
SAVE_EVERY_EPOCH = True


def load_qwen_model(model_path):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
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


def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_trainable_weights(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    trainable_weights = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    path = os.path.join(save_dir, "lora_weights.pt")
    torch.save(trainable_weights, path)
    return path, len(trainable_weights)


def init_log_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "batch_step",
            "global_step",
            "is_optimizer_step",
            "loss",
            "epoch_avg_loss_so_far",
            "learning_rate",
            "batch_time_sec",
            "optimizer_step_time_sec",
            "elapsed_sec",
        ])


def append_log_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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


def main():
    torch.backends.cudnn.enabled = False
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("=" * 80)
    print("Qwen reasoning-chain fine-tuning with plain LoRA")
    print(f"Model path: {MODEL_PATH}")
    print(f"Train jsonl: {TRAIN_JSONL}")
    print(f"Train conditions: {TRAIN_CONDITIONS if TRAIN_CONDITIONS else 'all'}")
    print(f"Train classes: {TRAIN_CLASSES if TRAIN_CLASSES else 'all'}")
    print(f"Save root: {SAVE_ROOT}")
    print("=" * 80)

    model_load_start = time.perf_counter()
    model = load_qwen_model(MODEL_PATH)
    print(f"Model load time: {time.perf_counter() - model_load_start:.2f}s")

    if not hasattr(model.config, "vocab_size"):
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
            model.config.vocab_size = model.config.text_config.vocab_size

    freeze_base_model(model)
    replaced = inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )
    print(f"Plain LoRA injected layers: {replaced}")
    print_trainable_parameters(model)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_start = time.perf_counter()
    dataset = QwenReasoningLoRADataset(
        jsonl_path=TRAIN_JSONL,
        model_path=MODEL_PATH,
        max_length=MAX_LENGTH,
    )
    original_count = len(dataset.data)
    dataset.data = filter_records(dataset.data, TRAIN_CONDITIONS, TRAIN_CLASSES)
    print(f"Training sample filter: {original_count} -> {len(dataset.data)}")
    if len(dataset.data) == 0:
        raise RuntimeError("No training samples after filtering.")

    collator = QwenReasoningLoRACollator(
        processor=dataset.processor,
        supports_image=dataset.supports_image,
        max_length=MAX_LENGTH,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )
    print(f"Dataset prep time: {time.perf_counter() - data_start:.2f}s")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = max(1, len(train_loader) * EPOCHS // GRAD_ACCUMULATION)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.03)),
        num_training_steps=total_steps,
    )

    device = get_input_device(model)
    model.train()

    print("=" * 80)
    print(f"Samples: {len(dataset)}")
    print(f"batch_size: {BATCH_SIZE}, grad_accumulation: {GRAD_ACCUMULATION}")
    print(f"epochs: {EPOCHS}, lr: {LR}")
    print(f"Input device: {device}")
    print(f"Using image processor: {dataset.supports_image}")
    print("=" * 80)

    init_log_csv(LOG_CSV)
    run_start_time = time.perf_counter()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(progress, start=1):
            batch_start_time = time.perf_counter()
            optimizer_step_time = 0.0

            batch.pop("label_id", None)
            batch.pop("expert_cluster_id", None)
            inputs = move_batch_to_device(batch, device)

            outputs = model(**inputs)
            loss = outputs.loss / GRAD_ACCUMULATION
            loss.backward()

            step_loss = float(loss.detach().cpu()) * GRAD_ACCUMULATION
            epoch_loss += step_loss
            avg_loss_so_far = epoch_loss / step
            lr_now = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else LR
            is_optimizer_step = step % GRAD_ACCUMULATION == 0 or step == len(train_loader)

            if is_optimizer_step:
                opt_start_time = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer_step_time = time.perf_counter() - opt_start_time

            batch_time = time.perf_counter() - batch_start_time
            elapsed = time.perf_counter() - run_start_time
            progress.set_postfix({
                "loss": f"{step_loss:.4f}",
                "avg": f"{avg_loss_so_far:.4f}",
                "lr": f"{lr_now:.2e}",
                "time": f"{batch_time:.2f}s",
            })

            append_log_csv(LOG_CSV, [
                epoch,
                step,
                global_step,
                int(is_optimizer_step),
                f"{step_loss:.8f}",
                f"{avg_loss_so_far:.8f}",
                f"{lr_now:.12f}",
                f"{batch_time:.6f}",
                f"{optimizer_step_time:.6f}",
                f"{elapsed:.6f}",
            ])

        avg_loss = epoch_loss / max(len(train_loader), 1)
        epoch_time = time.perf_counter() - epoch_start_time
        print(f"Epoch {epoch} finished, avg_loss={avg_loss:.6f}, time={epoch_time:.2f}s")

        if SAVE_EVERY_EPOCH:
            save_dir = os.path.join(SAVE_ROOT, f"epoch_{epoch}")
            weight_path, count = save_trainable_weights(model, save_dir)
            print(f"Saved {count} trainable tensors: {weight_path}")

    final_dir = os.path.join(SAVE_ROOT, "final")
    weight_path, count = save_trainable_weights(model, final_dir)
    total_time = time.perf_counter() - run_start_time

    print("=" * 80)
    print(f"Training finished. Saved {count} trainable tensors: {weight_path}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Training log: {LOG_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
