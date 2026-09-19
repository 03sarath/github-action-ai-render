# ---------- Stage 1: export + quantize (heavy, thrown away) ----------
# PyTorch and optimum are only needed to PRODUCE the ONNX file. None of
# this reaches the final image, which is why it can be huge here.
FROM python:3.11-slim AS builder

ARG HF_MODEL=distilbert-base-uncased-finetuned-sst-2-english

RUN pip install --no-cache-dir \
      "optimum[onnxruntime]==1.24.0" \
      "transformers==4.46.3" \
      "torch==2.5.1" --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /build

# Export to ONNX, then quantize fp32 -> int8 (roughly 4x smaller weights).
RUN python -c "\
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer; \
from optimum.onnxruntime.configuration import AutoQuantizationConfig; \
from transformers import AutoTokenizer; \
m = ORTModelForSequenceClassification.from_pretrained('${HF_MODEL}', export=True); \
m.save_pretrained('/tmp/onnx'); \
AutoTokenizer.from_pretrained('${HF_MODEL}').save_pretrained('/tmp/onnx'); \
q = ORTQuantizer.from_pretrained('/tmp/onnx'); \
q.quantize(quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False), save_dir='/model'); \
print('export + quantization done')"

# The quantizer writes model_quantized.onnx - make the runtime name predictable.
RUN cp /tmp/onnx/tokenizer.json /model/tokenizer.json && \
    mv /model/model_quantized.onnx /model/model.onnx && \
    rm -f /model/model.onnx_data && \
    ls -lh /model

# ---------- Stage 2: runtime (small) ----------
FROM python:3.11-slim

# onnxruntime + tokenizers only. No torch, no transformers.
RUN pip install --no-cache-dir \
      onnxruntime==1.20.1 \
      tokenizers==0.20.3 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.34.0" \
      numpy==2.1.3

COPY --from=builder /model /model
COPY app.py /app/app.py

WORKDIR /app

ENV MODEL_DIR=/model \
    PORT=7860 \
    OMP_NUM_THREADS=1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
