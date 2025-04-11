from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
import evaluate
import warnings as wr
import numpy as np
from transformers import enable_full_determinism, set_seed
from accelerate import Accelerator, PartialState
import json
import gc

#  Set seeds for reproducibility
seed_value = 42
set_seed(seed_value)
enable_full_determinism(seed_value)
wr.filterwarnings('ignore')

#  Initialize Accelerate
accelerator = Accelerator()
partial_state = PartialState()

#  Load 10% of the dataset for initial runs
train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='test.jsonl', split='train[:1%]')

#  Falcon 3B Instruct Model and Tokenizer
model_name = 'tiiuae/Falcon3-1B-Instruct'

#  Load model with quantization and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    #load_in_8bit=True,
    use_cache=False
)
model.config.use_cache = False

print(torch.cuda.memory_summary(device=0, abbreviated=False))

#  Load tokenizer with special tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
#  Reduce sequence length to 512
max_length = 512

#  Prompt formatting function
def formatting_func(example):
    output_str = json.dumps(example['output'], ensure_ascii=False)
    text = (
        f"### Instruction: {example['instruction']}\n"
        f"### Input: {example['input']}\n"
        f"### Output: {output_str}"
    )
    return text

#  Tokenization and prompt generation
def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

#  Tokenize datasets
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

#  LoRA Configuration
#peft_config = LoraConfig(lora_alpha=32,lora_dropout=0.1,r=16,bias='none',task_type='CAUSAL_LM',target_modules=['query_key_value'])

# Prepare the model for LoRA fine-tuning
#model = prepare_model_for_kbit_training(model)
#model = get_peft_model(model, peft_config)

# Check available GPUs and enable parallelism with Accelerate
if partial_state.num_processes > 1:
    model.is_parallelizable = True
    model.model_parallel = True

model.gradient_checkpointing_enable()

# Training Arguments (Optimized)
args = TrainingArguments(
    output_dir='falcon_1B_fine_tuned',
    num_train_epochs=1,
    max_steps=-1,  # Start with 1 epoch to test speed
    per_device_train_batch_size=32,  # Reduced batch size for Falcon 3B
    gradient_accumulation_steps=2,
    warmup_ratio=0.03,
    optim='paged_adamw_8bit',
    logging_steps=20000,
    save_strategy='steps',
    save_steps=50000,
    evaluation_strategy='no',
    learning_rate=5e-5,
    bf16=True,  # Use BF16 for faster computation
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    eval_accumulation_steps=4
)

#  Load Accuracy Metric
accuracy_metric = evaluate.load("accuracy")

#  Compute Evaluation Metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1).flatten()
    labels = labels.flatten()

    #  Accuracy Calculation
    accuracy_results = accuracy_metric.compute(predictions=predictions, references=labels)

    #  Perplexity Calculation
    eval_loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits).permute(0, 2, 1), torch.tensor(labels)).item()
    perplexity = np.exp(eval_loss) if eval_loss < 100 else float('inf')

    return {
        "accuracy": accuracy_results['accuracy'],
        "perplexity": perplexity
    }

#  Data Collator for Causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#  Initialize Trainer with Accelerate
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#  Prepare Trainer with Accelerate
trainer = accelerator.prepare(trainer)

#  Start Fine-Tuning
model.config.use_cache = False
trainer.train()

#  Save Final Model
trainer.save_model('falcon_1B_fine_tuned_final')

del trainer
gc.collect()
torch.cuda.empty_cache()

#  Push Merged Model to Hugging Face Hub
trainer.push_to_hub('sri-lasya/falcon-1B-book-recommendation')
