from vllm import LLM
from transformers import AutoTokenizer
from utils.retrieve import build_en2th_prompt, extract_json, inference_mt

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch

from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import os

# =========================
# FastAPI schemas
# =========================
class EnglishRequest(BaseModel):
    wipo_id: int
    english: str

class TranslateResponse(BaseModel):
    translation: str


# =========================
# FastAPI start-tup
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # LLM setup
    llm = LLM(
        model=os.getenv("MODEL_ID", "unsloth/gemma-3-1b-it"),
        quantization="bitsandbytes",
        max_model_len=4096,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    

# =========================
# FastAPI start-tup
# =========================
app = FastAPI(title="En2Th Translation API (vLLM + RAG)", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate_en2th", response_model=TranslateResponse)
def translate_en2th(req: EnglishRequest):
    chat_str = build_en2th_prompt(
        wipo_id=req.wipo_id,
        english=req.english,
    )

    # vLLM inference
    raw_output = inference_mt(llm, chat_str)
    thai_cleaned = extract_json(raw_output, en2th=True)

    return TranslateResponse(
        translation=thai_cleaned,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
