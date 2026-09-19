"""Sentiment API backed by a quantized ONNX DistilBERT.

Same contract as the PyTorch image it replaces:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"label": "POSITIVE", "score": 0.99}
"""
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
LABELS = ("NEGATIVE", "POSITIVE")
MAX_LEN = 256

app = FastAPI(title="DistilBERT Sentiment (ONNX int8)")

# Let a browser page (e.g. index.html opened locally or hosted elsewhere) call
# the API. Public demo with no cookies/auth, so any origin is fine; set
# CORS_ORIGINS="https://a.com,https://b.com" to lock it down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Single-threaded session keeps RSS small - the free instance has 0.1 CPU,
# extra threads buy nothing and cost memory.
_options = ort.SessionOptions()
_options.intra_op_num_threads = 1
_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    sess_options=_options,
    providers=["CPUExecutionProvider"],
)
tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
tokenizer.enable_truncation(max_length=MAX_LEN)

_input_names = {i.name for i in session.get_inputs()}


class PredictRequest(BaseModel):
    text: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    encoding = tokenizer.encode(request.text)

    feeds = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
    }
    # Some exports keep token_type_ids, some drop it - only feed what it wants.
    feeds = {name: value for name, value in feeds.items() if name in _input_names}

    logits = session.run(None, feeds)[0][0]
    probabilities = _softmax(logits)
    best = int(np.argmax(probabilities))

    return {"label": LABELS[best], "score": round(float(probabilities[best]), 4)}
