import json
import random
import glob
import os

# Directory containing your JSONL files
input_dir = "C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/python_scripts/deep learning based/transformer"  # Change this to your directory path
output_file = os.path.join(input_dir, "training.jsonl")

# Get a list of all JSONL files in the directory
file_paths = glob.glob(os.path.join(input_dir, "*.jsonl"))

# Check if any files were found
if not file_paths:
    print("❌ No JSONL files found in the specified directory.")
    exit()

combined_data = []

# Read and load each JSONL file
for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Load each line as a JSON object
                    combined_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON line in file: {file_path}")
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        exit()

# Check if there is any data to process
if not combined_data:
    print("❌ No valid JSON data found in the files.")
    exit()

# Shuffle the combined data for randomness
random.shuffle(combined_data)

# Optional: Remove duplicates (if needed)
combined_data = list({json.dumps(item, sort_keys=True): item for item in combined_data}.values())

# Save the shuffled and combined data into a new JSONL file
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Combined and shuffled data saved to: {output_file}")
except Exception as e:
    print(f"❌ Error while writing to file: {e}")
