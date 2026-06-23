import pickle
import uvicorn
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
svm_model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize FastAPI app
app = FastAPI(title="Binary Classifier API", description="Classifies passages as positive or negative.", docs_url="/")

# Define request body
class SingleTextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]  # List of passages

# Text cleaning function (same as used in preprocessing)
def preprocess_text(text):
    text = text.lower().strip()  # Lowercase & trim spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Single text prediction endpoint
@app.post("/predict/")
def predict_passage(request: SingleTextRequest):
    """
    Classifies a **single** input text as positive or negative.
    """
    text = preprocess_text(request.text)
    text_vectorized = vectorizer.transform([text])
    prediction = svm_model.predict(text_vectorized)[0]
    probability = svm_model.predict_proba(text_vectorized)[0][1]
    return {
        "text": request.text,
        "prediction": int(prediction),
        "probability": float(probability)
    }

# **Batch** prediction endpoint
@app.post("/predict_batch/")
def predict_passages(request: BatchTextRequest):
    """
    Classifies **multiple** input texts as positive or negative.
    """
    processed_texts = [preprocess_text(text) for text in request.texts]
    text_vectorized = vectorizer.transform(processed_texts)
    predictions = svm_model.predict(text_vectorized)
    probabilities = svm_model.predict_proba(text_vectorized)[:, 1]
    return [
        {
            "text": original_text,
            "prediction": int(pred),
            "probability": float(prob)
        }
        for original_text, pred, prob in zip(request.texts, predictions, probabilities)
    ]

# Run the API service
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
