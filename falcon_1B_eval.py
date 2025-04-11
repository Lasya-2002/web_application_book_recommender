from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import evaluate

model_name = "./falcon_1B_fine_tuned_final"

# Load model with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

test_dataset = load_dataset('json', data_files='test.jsonl', split='train[:1%]')

# Simplified tokenization (no manual tensor conversion)
def preprocess_function(examples):
    return tokenizer(
        [f"### Instruction: {inst}\n### Input: {inp}\n### Response: {out}" 
         for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,  # Increase batch size with automatic device mapping
    remove_columns=test_dataset.column_names
)

# Training arguments with memory optimization
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
    eval_accumulation_steps=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset,
)

results = trainer.evaluate()
print(f"Evaluation Loss: {results['eval_loss']:.4f}")
print(f"Perplexity: {np.exp(results['eval_loss']):.2f}")

rouge = evaluate.load("rouge")
generation_args = {
    "max_new_tokens": 128,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id
}

all_predictions = []
all_references = []

for i in tqdm(range(0, len(test_dataset)), desc="Generating Responses"):
    sample = test_dataset[i]
    prompt = f"### Instruction: {sample['instruction']}\n### Input: {sample['input']}\n### Response:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_args
        )
    
    prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    all_predictions.append(str(prediction).strip())
    all_references.append(str(sample["output"]).strip())

rouge_scores = rouge.compute(
    predictions=all_predictions,
    references=all_references,
    use_stemmer=True
)

print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")