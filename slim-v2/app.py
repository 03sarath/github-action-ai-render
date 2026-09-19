"""Sentiment web app backed by a quantized ONNX DistilBERT.

    GET  /         -> the web page (index.html)
    GET  /health   -> {"status": "ok", "version": "v1"}
    POST /predict  -> {"label": "POSITIVE", "score": 0.99}
"""
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
APP_VERSION = os.environ.get("APP_VERSION", "dev")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
LABELS = ("NEGATIVE", "POSITIVE")
MAX_LEN = 256

app = FastAPI(title="DistilBERT Sentiment (ONNX int8)")

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


# The page and the API share one origin, so no CORS setup is needed.
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": APP_VERSION}


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
