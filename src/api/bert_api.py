from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI(title="Binary Classifier API", description="Classifies passages as positive or negative.", docs_url="/")

# Load model and tokenizer
model_path = "bert_classifier"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Request and response schemas
class TextClassificationRequest(BaseModel):
    text: list[str]

class SinglePrediction(BaseModel):
    prediction: int
    negative_probability: float
    positive_probability: float

class TextClassificationResponse(BaseModel):
    results: list[SinglePrediction]

# Endpoint
@app.post("/classify/", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    try:
        texts = request.text
        inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        results = []
        for pred, prob in zip(preds, probs):
            results.append(SinglePrediction(
                prediction=int(pred),
                negative_probability=float(prob[0]),
                positive_probability=float(prob[1])
            ))

        return TextClassificationResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
