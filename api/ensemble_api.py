#!/usr/bin/env python3
"""
REST API for Biotic Interaction Classifier
Runs on 0.0.0.0 to be accessible from outside localhost
"""

import os
import sys
sys.path.insert(0, '/path/to/MetaP/classifier')

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Configuration
MODEL_DIR = '/path/to/MetaP/classifier/models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best ensemble configuration (4-model, F1=0.524)
ENSEMBLE_CONFIG = {
    'biomedbert_6k': {
        'path': f'{MODEL_DIR}/transformer_BiomedBERT_model_6k_original',
        'weight': 0.25,
    },
    'biomedbert_20k': {
        'path': f'{MODEL_DIR}/transformer_BiomedBERT_model_enhanced_20k',
        'weight': 0.25,
    },
    'biomedbert_quality_v2': {
        'path': f'{MODEL_DIR}/transformer_BiomedBERT_quality_v2',
        'weight': 0.25,
    },
    'biobert': {
        'path': f'{MODEL_DIR}/transformer_biobert_model',
        'weight': 0.25,
    },
}

BEST_THRESHOLD = 0.50

# Global model storage
models = {}
tokenizers = {}


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def load_models():
    """Load all ensemble models"""
    global models, tokenizers

    print(f"Loading models on {DEVICE}...")

    for name, config in ENSEMBLE_CONFIG.items():
        print(f"  Loading {name}...")
        tokenizers[name] = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForSequenceClassification.from_pretrained(config['path'])

        if DEVICE == 'cuda':
            model = model.half()

        model.to(DEVICE)
        model.eval()
        models[name] = model
        print(f"    -> {name} loaded (weight: {config['weight']})")

    print("All models loaded!")


@torch.no_grad()
def get_ensemble_predictions(texts, threshold=BEST_THRESHOLD):
    """Get predictions from the ensemble"""
    all_probs = {}

    for name, model in models.items():
        tokenizer = tokenizers[name]
        dataset = TextDataset(texts, tokenizer)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        probs = []
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())

        all_probs[name] = np.array(probs)

    # Weighted average
    ensemble_probs = np.zeros(len(texts))
    for name, probs in all_probs.items():
        ensemble_probs += ENSEMBLE_CONFIG[name]['weight'] * probs

    predictions = (ensemble_probs >= threshold).astype(int)

    return predictions, ensemble_probs


@app.route('/', methods=['GET', 'POST'])
def index():
    """Web interface for predictions"""
    result_html = ""
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('text', '')
        if input_text.strip():
            predictions, probabilities = get_ensemble_predictions([input_text], BEST_THRESHOLD)
            pred = predictions[0]
            prob = probabilities[0]
            label = "INTERACTION" if pred == 1 else "NO INTERACTION"
            color = "#28a745" if pred == 1 else "#dc3545"
            result_html = f'''
            <div style="margin: 20px 0; padding: 20px; border: 2px solid {color}; border-radius: 10px; background: #f8f9fa;">
                <h3 style="color: {color}; margin: 0 0 10px 0;">Prediction: {label}</h3>
                <p style="margin: 5px 0;"><b>Probability:</b> {prob:.3f}</p>
                <p style="margin: 5px 0;"><b>Threshold:</b> {BEST_THRESHOLD}</p>
                <p style="margin: 5px 0; color: #666;"><b>Text:</b> {input_text[:200]}{'...' if len(input_text) > 200 else ''}</p>
            </div>
            '''

    return f'''
    <html>
    <head>
        <title>Biotic Interaction Classifier</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .info {{ background: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            textarea {{ width: 100%; height: 150px; padding: 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 5px; }}
            button {{ background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #0056b3; }}
            .examples {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .example {{ cursor: pointer; color: #007bff; margin: 5px 0; }}
            .example:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Biotic Interaction Classifier</h1>
        <div class="info">
            <b>Model:</b> 4-model ensemble (BiomedBERT 6k + 20k + quality_v2 + BioBERT)<br>
            <b>Performance:</b> F1=0.524, Precision=0.579, Recall=0.478
        </div>

        <form method="POST">
            <h3>Enter a sentence to classify:</h3>
            <textarea name="text" placeholder="Enter a scientific sentence about species interactions...">{input_text}</textarea>
            <br><br>
            <button type="submit">Classify</button>
        </form>

        {result_html}

        <div class="examples">
            <h4>Example sentences (click to try):</h4>
            <p class="example" onclick="document.querySelector('textarea').value=this.innerText">The parasite Plasmodium falciparum infects human red blood cells causing malaria.</p>
            <p class="example" onclick="document.querySelector('textarea').value=this.innerText">Phylogenetic analysis of mitochondrial DNA sequences revealed evolutionary relationships.</p>
            <p class="example" onclick="document.querySelector('textarea').value=this.innerText">The predatory beetle feeds on aphids in agricultural ecosystems.</p>
            <p class="example" onclick="document.querySelector('textarea').value=this.innerText">Morphological characteristics were used to identify the species.</p>
        </div>

        <hr>
        <h3>API Endpoints:</h3>
        <ul>
            <li><b>GET /health</b> - Health check</li>
            <li><b>POST /predict</b> - JSON API for predictions</li>
            <li><b>POST /predict_csv</b> - Predict on CSV file</li>
        </ul>
    </body>
    </html>
    '''


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'models_loaded': list(models.keys()),
        'threshold': BEST_THRESHOLD,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict biotic interaction for a single sentence or list of sentences"""
    data = request.json

    if 'text' in data:
        texts = [data['text']]
    elif 'texts' in data:
        texts = data['texts']
    else:
        return jsonify({'error': 'Provide "text" or "texts" in request body'}), 400

    threshold = data.get('threshold', BEST_THRESHOLD)

    predictions, probabilities = get_ensemble_predictions(texts, threshold)

    results = []
    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': int(pred),
            'probability': float(prob),
            'label': 'interaction' if pred == 1 else 'no_interaction',
        })

    return jsonify({
        'threshold': threshold,
        'results': results,
    })


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Predict on a CSV file path"""
    data = request.json

    if 'file_path' not in data:
        return jsonify({'error': 'Provide "file_path" in request body'}), 400

    file_path = data['file_path']
    text_column = data.get('text_column', 'sentence')
    threshold = data.get('threshold', BEST_THRESHOLD)

    # Load CSV
    if file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    if text_column not in df.columns:
        return jsonify({'error': f'Column "{text_column}" not found. Available: {list(df.columns)}'}), 400

    texts = df[text_column].tolist()
    predictions, probabilities = get_ensemble_predictions(texts, threshold)

    # Save predictions
    output_path = file_path.replace('.tsv', '_predictions.csv').replace('.csv', '_predictions.csv')
    if output_path == file_path:
        output_path = file_path + '_predictions.csv'

    result_df = df.copy()
    result_df['ensemble_prediction'] = predictions
    result_df['ensemble_probability'] = probabilities
    result_df['ensemble_label'] = ['interaction' if p == 1 else 'no_interaction' for p in predictions]
    result_df.to_csv(output_path, index=False)

    return jsonify({
        'input_file': file_path,
        'output_file': output_path,
        'total_samples': len(texts),
        'predicted_interactions': int(sum(predictions)),
        'predicted_no_interactions': int(len(predictions) - sum(predictions)),
        'threshold': threshold,
    })


if __name__ == '__main__':
    # Load models at startup
    load_models()

    # Run on all interfaces (0.0.0.0) so it's accessible externally
    print("\n" + "="*60)
    print("BIOTIC INTERACTION CLASSIFIER API")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Threshold: {BEST_THRESHOLD}")
    print(f"Models: {list(ENSEMBLE_CONFIG.keys())}")
    print("\nEndpoints:")
    print("  GET  /health       - Health check")
    print("  POST /predict      - Predict single text or list")
    print("  POST /predict_csv  - Predict on CSV file")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
