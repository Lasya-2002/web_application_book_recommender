import pandas as pd
import json
import faiss
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reload FAISS indexes
index_title = faiss.read_index("books_faiss_title.index")
index_description = faiss.read_index("books_faiss_description.index")
index_category = faiss.read_index("books_faiss_category.index")

# Load dataset
df = pd.read_csv("books_with_embeddings.csv")

# Define instruction variants
instructions = [
    ("Recommend books based on a preferred genre", "I enjoy reading books in the '{}' genre. Can you suggest similar books?"),
    ("Suggest books from the same or similar authors", "I love books by {}. Can you recommend other books by the same author or similar authors?"),
    ("Provide high-rated book recommendations in a specific language", "I'm looking for some highly-rated books in {}. Can you suggest a few?"),
    ("Recommend books with similar descriptions", "I like books that have descriptions similar to '{}'. Can you suggest some?"),
    ("Suggest books with high ratings and review counts", "I want to read popular books with at least 4.0 rating and 100+ reviews. Can you help me find some?"),
    ("Recommend books based on both category and author", "I'm a fan of {}, and I enjoy {} books. Can you recommend something similar?"),
    ("Recommend hidden gems in a genre", "Can you recommend some lesser-known books in the '{}' genre?"),
    ("Provide book recommendations based on mood", "I'm in the mood for an exciting/thrilling/romantic/mysterious book. Any recommendations?"),
    ("Suggest books frequently bought together", "People who read '{}' also liked... Can you complete the list?"),
    ("Suggest beginner-friendly books", "I'm a beginner reader. Can you suggest some books that are easy to read?")
]

# Checkpoint file to track progress
checkpoint_file = "checkpoint.json"

# Load checkpoint progress or start fresh
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)
else:
    checkpoint_data = {str(i): 0 for i in range(len(instructions))}  # Start fresh if no checkpoint

def update_checkpoint(instruction_id, idx):
    """Update the checkpoint file with retry on PermissionError."""
    checkpoint_data[str(instruction_id)] = idx + 1
    success = False
    while not success:
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f)
            success = True
        except PermissionError:
            print("PermissionError when writing checkpoint. Retrying in 2 seconds...")
            time.sleep(2)

def find_similar_books(query_embedding, index, df, top_k=2):
    """Find books with the most similar embeddings using FAISS."""
    D, I = index.search(np.array([query_embedding]), top_k + 1)  # +1 to exclude itself
    valid_indices = [idx for idx in I[0][1:] if 0 <= idx < len(df)]
    
    if not valid_indices:
        return []
    
    return df.iloc[valid_indices]["Title"].tolist()

def process_instruction(instruction_id, instruction_text, instruction_input_template):
    """Process a specific instruction for all books in the dataset."""
    jsonl_filename = f"instruction_{instruction_id}.jsonl"

    # Track processed books to avoid duplication
    processed_books = set()
    if os.path.exists(jsonl_filename):
        with open(jsonl_filename, "r") as f:
            processed_books = {json.loads(line)["output"]["book_itself"] for line in f}

    start_index = checkpoint_data[str(instruction_id)]  # Resume from the last saved position
    print(f"Resuming Instruction {instruction_id}: {instruction_text} from book index {start_index}")

    with open(jsonl_filename, "a") as f_out:
        for idx in tqdm(range(start_index, len(df)), desc=f"Processing Instruction {instruction_id}"):
            row = df.iloc[idx]
            if row["Title"] in processed_books:
                continue  # Skip if already processed

            query_title_embedding = embedding_model.encode(str(row["Title"]), convert_to_numpy=True)
            query_description_embedding = embedding_model.encode(str(row["Description"]), convert_to_numpy=True)
            query_category_embedding = embedding_model.encode(str(row["Categories"]), convert_to_numpy=True)

            similar_title_books = find_similar_books(query_title_embedding, index_title, df)
            similar_description_books = find_similar_books(query_description_embedding, index_description, df)
            similar_category_books_faiss = find_similar_books(query_category_embedding, index_category, df)

            output_data = {
                "instruction": instruction_text,
                "input": instruction_input_template.format(
                    str(row["Categories"]), str(row["Authors"]), str(row["Language"]), str(row["Title"])
                ),
                "output": {
                    "book_itself": str(row["Title"]),
                    "categories": str(row["Categories"]),
                    "average_rating": float(row["Average Rating"]),  # Convert float64 to Python float
                    "ratings_count": int(row["Ratings Count"]),  # Convert int64 to Python int
                    "language": str(row["Language"]),
                    "similar_title_books": similar_title_books,
                    "similar_description_books": similar_description_books,
                    "similar_category_books_faiss": similar_category_books_faiss
                }
            }

            f_out.write(json.dumps(output_data) + "\n")

            # Update checkpoint every 100 iterations
            if (idx + 1) % 100 == 0:
                update_checkpoint(instruction_id, idx)

    # Final checkpoint update after finishing the loop
    update_checkpoint(instruction_id, len(df) - 1)

def parallel_process_instructions():
    """Run multiple instructions in parallel."""
    max_parallel_instructions = 3  # Run 3 instructions at a time to avoid memory contention

    with ThreadPoolExecutor(max_workers=max_parallel_instructions) as executor:
        futures = []
        for instruction_id, (instruction_text, instruction_input_template) in enumerate(instructions):
            futures.append(
                executor.submit(process_instruction, instruction_id, instruction_text, instruction_input_template)
            )

        # Wait for all tasks to finish
        for future in futures:
            future.result()

# Run the parallelized processing for all instructions
if __name__ == "__main__":
    start_time = time.time()
    parallel_process_instructions()
    end_time = time.time()
    print(f"✅ All instructions processed successfully in {(end_time - start_time) / 3600:.2f} hours!")
