import json
import os
import re


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data1")
BATCH_ROOT = os.path.join(DATA_DIR, "GPT_Image_Batches")
OUT_DIR = os.path.join(DATA_DIR, "GPT_Reasoning")

RESPONSE_FILENAMES = [
    "推理链.txt",
    "reasoning.txt",
    "response.txt",
    "gpt_response.txt",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_response_file(batch_dir):
    for name in RESPONSE_FILENAMES:
        path = os.path.join(batch_dir, name)
        if os.path.exists(path):
            return path
    return None


def extract_blocks(text):
    pattern = re.compile(
        r"(?:Sample|样本)\s*0*(\d+)\s*[:：]\s*(.*?)(?=(?:\n(?:Sample|样本)\s*0*\d+\s*[:：])|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = {}
    for match in pattern.finditer(text):
        sample_no = int(match.group(1))
        block = match.group(2).strip()
        blocks[sample_no] = normalize_reasoning_block(block)
    return blocks


def normalize_reasoning_block(block):
    block = block.strip()
    if not block:
        return ""

    if "<think>" not in block and "A:" in block:
        a_pos = block.rfind("A:")
        reasoning = block[:a_pos].strip()
        answer = block[a_pos:].strip()
        block = f"<think>{reasoning}</think>\n{answer}"
    elif "<think>" not in block and "A：" in block:
        a_pos = block.rfind("A：")
        reasoning = block[:a_pos].strip()
        answer = block[a_pos:].strip()
        block = f"<think>{reasoning}</think>\n{answer}"

    return block


def extract_answer_label(block):
    match = re.search(r"A\s*[:：]\s*(C[0-4])", block, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def read_metadata(batch_dir):
    path = os.path.join(batch_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metadata.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_split(split):
    split_dir = os.path.join(BATCH_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"Missing split folder: {split_dir}")
        return []

    merged = []
    folder_names = sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )

    for folder_name in folder_names:
        folder_dir = os.path.join(split_dir, folder_name)
        batch_names = sorted(
            name for name in os.listdir(folder_dir)
            if os.path.isdir(os.path.join(folder_dir, name)) and name.startswith("batch_")
        )

        for batch_name in batch_names:
            batch_dir = os.path.join(folder_dir, batch_name)
            metadata = read_metadata(batch_dir)
            response_file = find_response_file(batch_dir)

            if response_file is None:
                blocks = {}
                print(f"[Missing] {split}/{folder_name}/{batch_name}: no response txt")
            else:
                with open(response_file, "r", encoding="utf-8") as f:
                    blocks = extract_blocks(f.read())
                if len(blocks) != len(metadata):
                    print(
                        f"[Check] {split}/{folder_name}/{batch_name}: "
                        f"metadata={len(metadata)}, extracted={len(blocks)}"
                    )

            for record in metadata:
                sample_no = int(record["sample_no"])
                reasoning_chain = blocks.get(sample_no, "")
                item = dict(record)
                item["batch"] = batch_name
                item["batch_dir"] = batch_dir
                item["response_file"] = response_file or ""
                item["reasoning_chain"] = reasoning_chain
                item["reasoned_label"] = extract_answer_label(reasoning_chain)
                merged.append(item)

    return merged


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ensure_dir(OUT_DIR)

    all_rows = []
    for split in ["train", "test"]:
        rows = merge_split(split)
        out_path = os.path.join(OUT_DIR, f"{split}_reasoning_chains.jsonl")
        write_jsonl(out_path, rows)
        all_rows.extend(rows)
        print(f"{split}: {len(rows)} samples -> {out_path}")

    all_path = os.path.join(OUT_DIR, "all_reasoning_chains.jsonl")
    write_jsonl(all_path, all_rows)
    print("=" * 80)
    print(f"All merged samples: {len(all_rows)} -> {all_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
