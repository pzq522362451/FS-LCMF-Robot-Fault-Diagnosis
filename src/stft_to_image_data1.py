import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from PIL import Image
from tqdm import tqdm
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split

INPUT_BASE_DIR = r"../Data1"
OUTPUT_DIR = r"../Data1/My_VLM_Dataset"

LABEL_MAP = {
    0: "正常工况",
    1: "第2轴RV40E-121行星轮齿裂纹故障，宽0.5mm深0.5mm",
    2: "第2轴RV40E-121行星轮齿断齿故障，全断齿",
    3: "第3轴RV20E-121太阳轮齿点蚀故障",
    4: "第3轴RV20E-121太阳轮齿断齿故障，全断齿",
}

CONDITIONS = ["S1L2", "S2L2", "S3L2"]
CLASSES = ["C0", "C1", "C2", "C3", "C4"]

SLICE_LENGTH = 5000
OVERLAP_RATE = 0.0
SR = 100000
CWT_WAVELET = 'cmor'
CWT_SCALES = np.arange(1, 128)

MAX_SAMPLES_PER_CLASS = 100  # 设置为 None 表示使用所有样本，设置为数字如 100 表示每类最多取 100 个样本


def read_lvm_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('X_Value') or line.startswith('Channel'):
            data_start = i + 1
            break
    
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except:
                continue
    
    return np.array(data)


def normalize_data(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)


def slice_signal(data, slice_length, overlap_rate):
    step_size = int(slice_length * (1 - overlap_rate))
    signal_length = data.shape[-1]
    num_slices = (signal_length - slice_length) // step_size + 1
    
    slices = []
    for i in range(num_slices):
        start = i * step_size
        end = start + slice_length
        slices.append(data[start:end])
    
    return np.array(slices)


def apply_cwt_and_save_image(signal, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if signal.ndim >= 2:
        signal = signal.flatten()
    
    coefficients, frequencies = pywt.cwt(signal, CWT_SCALES, CWT_WAVELET, sampling_period=1/SR)
    coeffs = np.abs(coefficients)
    
    coeffs = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    ax.imshow(coeffs, cmap='jet', aspect='auto', origin='lower',
              extent=[0, len(signal)/SR, frequencies[-1], frequencies[0]])
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    image = Image.fromarray(img_array)
    resized_image = image.resize((256, 256), Image.Resampling.BILINEAR)
    resized_image.save(save_path)
    
    rms_value = float(np.sqrt(np.mean(signal ** 2)))
    kurt_value = float(kurtosis(signal))
    
    return rms_value, kurt_value


def normalize_to_255(matrix):
    matrix_min = matrix.min()
    matrix_max = matrix.max()
    if matrix_max - matrix_min == 0:
        return np.zeros_like(matrix, dtype=np.uint8)
    norm_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    return (norm_matrix * 255).astype(np.uint8)


def process_all_classes():
    all_data = {}
    
    for cls in CLASSES:
        for condition in CONDITIONS:
            folder_name = f"{cls}{condition}"
            folder_path = os.path.join(INPUT_BASE_DIR, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found")
                continue
            
            r05_folder = os.path.join(folder_path, "R05")
            lvm_file = os.path.join(r05_folder, "data.lvm")
            
            if not os.path.exists(lvm_file):
                print(f"Warning: {lvm_file} not found")
                continue
            
            print(f"Processing {folder_name}...")
            raw_data = read_lvm_file(lvm_file)
            
            if len(raw_data) == 0:
                print(f"Warning: No data in {lvm_file}")
                continue
            
            time_data = raw_data[:, 0]
            signal_data = raw_data[:, 1]
            
            normalized_data = normalize_data(signal_data)
            sliced = slice_signal(normalized_data, SLICE_LENGTH, OVERLAP_RATE)
            
            all_data[folder_name] = {
                'data': sliced,
                'label': int(cls[1]),
                'condition': condition
            }
    
    return all_data


def split_and_save(all_data):
    train_data = []
    train_labels = []
    train_conditions = []
    train_names = []
    
    test_data = []
    test_labels = []
    test_conditions = []
    test_names = []
    
    label_counts = {label: 0 for label in range(len(CLASSES))}
    
    for name, info in all_data.items():
        data = info['data']
        label = info['label']
        condition = info['condition']
        
        indices = np.arange(len(data))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_portion = data[train_idx]
        test_portion = data[test_idx]
        
        if MAX_SAMPLES_PER_CLASS is not None:
            remaining_train = MAX_SAMPLES_PER_CLASS - label_counts[label]
            if remaining_train <= 0:
                train_portion = train_portion[:0]
                train_idx = train_idx[:0]
            elif len(train_portion) > remaining_train:
                train_portion = train_portion[:remaining_train]
                train_idx = train_idx[:remaining_train]
            
            remaining_test = int(MAX_SAMPLES_PER_CLASS * 0.25) - (label_counts[label] - (len(train_idx) if len(train_idx) > 0 else 0))
            if remaining_test <= 0:
                test_portion = test_portion[:0]
                test_idx = test_idx[:0]
            elif len(test_portion) > remaining_test:
                test_portion = test_portion[:remaining_test]
                test_idx = test_idx[:remaining_test]
        
        train_data.append(train_portion)
        train_labels.extend([label] * len(train_portion))
        train_conditions.extend([condition] * len(train_portion))
        train_names.extend([f"{name}_window_{i}" for i in train_idx[:len(train_portion)]])
        
        test_data.append(test_portion)
        test_labels.extend([label] * len(test_portion))
        test_conditions.extend([condition] * len(test_portion))
        test_names.extend([f"{name}_window_{i}" for i in test_idx[:len(test_portion)]])
        
        label_counts[label] += len(train_portion) + len(test_portion)
    
    if len(train_data) > 0:
        train_data = np.vstack(train_data)
    else:
        train_data = np.array([])
    
    if len(test_data) > 0:
        test_data = np.vstack(test_data)
    else:
        test_data = np.array([])
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "raw_data"), exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "raw_data", "train_signals.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, "raw_data", "test_signals.npy"), test_data)
    
    print(f"\nTotal train samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    if MAX_SAMPLES_PER_CLASS is not None:
        print(f"Note: Using maximum {MAX_SAMPLES_PER_CLASS} samples per class")
    
    return {
        'train': {'data': train_data, 'labels': train_labels, 'conditions': train_conditions, 'names': train_names},
        'test': {'data': test_data, 'labels': test_labels, 'conditions': test_conditions, 'names': test_names}
    }


def generate_images_and_annotations(split_data):
    for split in ['train', 'test']:
        data = split_data[split]['data']
        labels = split_data[split]['labels']
        names = split_data[split]['names']
        
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        anno_dir = os.path.join(OUTPUT_DIR, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
        
        jsonl_path = os.path.join(anno_dir, f"{split}.jsonl")
        
        print(f"\nGenerating {split} images...")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(len(data))):
                signal = data[i]
                label_id = int(labels[i])
                label_name = LABEL_MAP.get(label_id, f"未知类别_{label_id}")
                sample_name = names[i]
                
                class_dir = os.path.join(img_dir, f"class_{label_id}")
                os.makedirs(class_dir, exist_ok=True)
                
                img_filename = f"{sample_name}.png"
                absolute_img_path = os.path.join(class_dir, img_filename)
                relative_img_path = f"images/{split}/class_{label_id}/{img_filename}"
                
                try:
                    rms, kurt = apply_cwt_and_save_image(signal, absolute_img_path)
                except Exception as e:
                    print(f"Error processing {sample_name}: {e}")
                    rms = 0.0
                    kurt = 0.0
                
                sample_data = {
                    "image_path": relative_img_path,
                    "raw_data_file": f"raw_data/{split}_signals.npy",
                    "array_index": i,
                    "label_id": label_id,
                    "label_name": label_name,
                    "features": {
                        "rms": round(rms, 4),
                        "kurtosis": round(kurt, 4)
                    },
                    "split": split,
                    "reasoning_chain": ""
                }
                
                f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
        print(f"Saved {split} annotations to {jsonl_path}")


def main():
    print("=" * 60)
    print("Processing Data1 folder for STFT image generation")
    print("=" * 60)
    
    print("\nStep 1: Loading and slicing data from .lvm files...")
    all_data = process_all_classes()
    
    print("\nStep 2: Splitting train/test sets...")
    split_data = split_and_save(all_data)
    
    print("\nStep 3: Generating STFT images and annotations...")
    generate_images_and_annotations(split_data)
    
    print("\n" + "=" * 60)
    print(f"Dataset generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
