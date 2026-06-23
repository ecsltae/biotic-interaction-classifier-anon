# This code is ran. the model has been trained and saved.

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#  Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load preprocessed dataset
df = pd.read_csv("training_data.csv")

#  Ensure text and label columns have no NaN values
df = df.dropna(subset=["passage", "label"])

#  Convert labels to integers
df["label"] = df["label"].astype(int)

#  Split into training and test sets (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["passage"].astype(str).tolist(), 
    df["label"].tolist(), 
    test_size=0.2, 
    random_state=42
)

#  Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#  Custom Dataset Class
class PassageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

#  Create dataset objects
train_dataset = PassageDataset(train_texts, train_labels, tokenizer)
test_dataset = PassageDataset(test_texts, test_labels, tokenizer)

#  Load pre-trained model for classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)  # Move to GPU if available

#  Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

#  Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

#  Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  Train the model
trainer.train()

#  Save the trained model
model.save_pretrained("bert_classifier")
tokenizer.save_pretrained("bert_classifier")

print("\n BERT model trained and saved!")




"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load dataset (Replace with your actual dataset path)
dataset = load_dataset("your_dataset")

tokenizer = AutoTokenizer.from_pretrained("your_model")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def convert_to_tensors(dataset_split):
    input_ids = torch.tensor(dataset_split["input_ids"])
    attention_mask = torch.tensor(dataset_split["attention_mask"])
    labels = torch.tensor(dataset_split["label"])  # Adjust if labels are named differently
    return input_ids, attention_mask, labels

train_inputs, train_masks, train_labels = convert_to_tensors(tokenized_datasets["train"])
eval_inputs, eval_masks, eval_labels = convert_to_tensors(tokenized_datasets["validation"])

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_inputs, train_masks, train_labels = train_inputs.to(device), train_masks.to(device), train_labels.to(device)
eval_inputs, eval_masks, eval_labels = eval_inputs.to(device), eval_masks.to(device), eval_labels.to(device)

print("Tensor conversion complete. Dataset moved to:", device)

"""