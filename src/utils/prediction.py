#!/usr/bin/env python3
"""
Script to generate a CSV with predictions from transformer_BiomedBERT_model
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the test set
def load_test_set(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

# Load the model and tokenizer
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

# Make predictions
def predict_sentences(model, tokenizer, sentences, device):
    predictions = []
    for sentence in sentences:
        inputs = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()

        predictions.append(pred)

    return predictions

# Main function
def main():
    # Load test set
    test_set_path = 'eval_100.tsv'
    df = load_test_set(test_set_path)

    # Load model
    model_path = 'transformer_BiomedBERT_model'
    model, tokenizer, device = load_model(model_path)

    # Make predictions
    predictions = predict_sentences(model, tokenizer, df['sentence'], device)

    # Map numerical labels to 'positive' and 'negative' for sentiment
    df['true_sentiment'] = df['evaluation_pair_interacting'].map({1: 'positive', 0: 'negative'})
    df['BiomedBERT_prediction'] = predictions
    df['BiomedBERT_sentiment'] = df['BiomedBERT_prediction'].map({1: 'positive', 0: 'negative'})

    # Save the results to a CSV file
    output_file = 'predictions_with_BiomedBERT.csv'
    df.to_csv(output_file, columns=['sentence', 'evaluation_pair_interacting', 'BiomedBERT_prediction', 'true_sentiment', 'BiomedBERT_sentiment'], index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
