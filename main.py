import uvicorn
import torch
from torch.nn.functional import softmax
import tensorflow as tf
import os
import numpy as np
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_path = "./model_artifact"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("./model_artifact")

app = FastAPI(title="Sentiment Analysis")

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

class Prediction(BaseModel):
  category: str
  confidence: Optional[float]

class Predictions(BaseModel):
    predictions: List[Prediction]

# instad of creating a class we could have also loaded this information
# from the model configuration. Better if you introduce new labels over time
class Category(Enum):
  NOT_LEAVE = 0
  LEAVE = 1


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE,
          response_model=Predictions,
          response_model_exclude_unset=True)
async def predict(request: Request):
    body = await request.json()
    print(body)
    instances = body["instances"]
    instances = [x['text'] for x in instances]
    print(instances)
    input_ids_list = tokenizer(instances, max_length=128, padding=True, truncation=True, return_tensors='pt')
    print(input_ids_list)
    with torch.no_grad():
        outputs = model(**input_ids_list)
    print(outputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    confidences = torch.max(probabilities, dim=1).values.tolist()

    predictions = []

    for predicted_class, confidence in zip(predicted_classes, confidences):
        category = Category(predicted_class.item()).name
        predictions.append(Prediction(category=category, confidence=confidence))

    return Predictions(predictions=predictions)


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)
