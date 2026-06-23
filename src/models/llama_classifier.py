# training 2 was doing sentiment analysis, this is biotic interaction detection

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Load preprocessed dataset
df = pd.read_csv("training_data_cleaned.csv")
df = df.dropna(subset=["passage", "label"])

# Convert labels to text for biotic interaction
df["label"] = df["label"].apply(lambda x: "Interaction" if x == 1 else "No Interaction")

# Create prompts for biotic interaction detection
df["prompt"] = "Passage: " + df["passage"] + "\nDoes this passage describe a biotic interaction between two species? Answer: "

# Split into training and test sets
train_texts, _, train_labels, _ = train_test_split(
    df["prompt"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist()
)

# Initialize Accelerator
accelerator = Accelerator()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize the label and shift for causal LM
        label_ids = self.tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids

        # Shift label_ids for causal LM
        labels = label_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }

# Create dataset objects
train_dataset = TextDataset(train_texts, train_labels, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    learning_rate=2e-5,
    gradient_accumulation_steps=16,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
trainer.model.save_pretrained("llama3_biotic_interaction")
tokenizer.save_pretrained("llama3_biotic_interaction")
print("\n✅ Llama 3 model for biotic interaction detection trained and saved!")
