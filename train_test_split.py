import json
import random

# File paths
input_file = "training.jsonl"
train_file = "train.jsonl"
val_file = "test.jsonl"

# Split ratio
train_ratio = 0.8  # 80% training, 20% validation

# Read the data from the original JSONL file
with open(input_file, "r",encoding='utf-8') as f:
    lines = f.readlines()

# Shuffle the data
random.shuffle(lines)

# Split the data
train_size = int(len(lines) * train_ratio)
train_data = lines[:train_size]
val_data = lines[train_size:]

# Write to train.jsonl
with open(train_file, "w",encoding='utf-8') as f:
    f.writelines(train_data)

# Write to val.jsonl
with open(val_file, "w",encoding='utf-8') as f:
    f.writelines(val_data)

print(f"Data split successfully! Training data: {len(train_data)} lines, Validation data: {len(val_data)} lines.")
