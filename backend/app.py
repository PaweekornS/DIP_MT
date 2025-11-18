import os
import json
import re
from typing import Optional

import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn


# =========================
# Config / Paths
# =========================
ROOT_DIR = "/project/lt200304-dipmt/paweekorn"

INSTRUCTION_PATH = f"{ROOT_DIR}/data/prompt/base_en2th.txt"
WIPO_JSON_PATH = f"{ROOT_DIR}/data/wipo/WIPO.json"

EMBED_MODEL_NAME = "bge-m3"
EMBED_MODEL_PATH = f"{ROOT_DIR}/models/retriever/{EMBED_MODEL_NAME}"
FAISS_INDEX_PATH = f"{ROOT_DIR}/vector/en2th/{EMBED_MODEL_NAME}"

# model config – สามารถเปลี่ยนมาใช้ env var ได้
MODEL_DIR = os.getenv("MODEL_DIR", f"{ROOT_DIR}/models/base/gemma3-4b-it")
ADAPTER_DIR = None


# =========================
# FastAPI schemas
# =========================
class TranslateRequest(BaseModel):
    wipo_id: int
    english: str
    is_rag: bool = True


class TranslateResponse(BaseModel):
    thai_raw: str
    thai_cleaned: str

# =========================
# Global objects
# =========================
app = FastAPI(title="En2Th Translation API (vLLM + RAG)")

instruction: str = ""
wipo_data: dict[int, str] = {}

embeddings = None
vectorstore = None

tokenizer = None
llm: LLM = None
lora_request: Optional[LoRARequest] = None


# =========================
# Utility: Thai filtering / JSON extraction
# =========================
def filter_thai(text: str) -> str:
    pattern = r'[\u0e00-\u0e7f\s,.?!]+'
    matches = re.findall(pattern, text)
    return "".join(matches).strip().replace("\n", "")


def extract_json(text: str) -> str:
    text = text[text.rfind("{"):]
    pattern = r'''{\s*[\'\"]thai_translation[\'\"]:\s*[\'\"].*?[\'\"]\s*}'''
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        try:
            loaded = json.loads(matches[0])
            return loaded['thai_translation']
        except json.JSONDecodeError:
            return filter_thai(text)
    else:
        return filter_thai(text)


# =========================
# Retrieval
# =========================
def get_relevant_docs(query: str, k: int = 3) -> str:
    query_embedding = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)

    relevant = ""
    for doc in docs:
        relevant += f'''English: {doc.page_content}
Thai: {doc.metadata.get("thai", "")}\n
'''

    rag_result = (
        "\n## Retrieved References:\n" +
        relevant +
        "**Note:** If the retrieved references contain identical English terms "
        "with different Thai translations (ambiguity), you must use your expert "
        "judgment to select the most appropriate and contextually accurate Thai "
        "translation for the current input.\n"
    )
    return rag_result


# =========================
# Prompt formatting
# =========================
def build_prompt(wipo_id: int, english: str) -> tuple[str, Optional[str]]:
    wipo_label = wipo_data.get(int(wipo_id), "")
    rag_doc = get_relevant_docs(english, 3)

    prompt = instruction.format(
        WIPO=wipo_label,
        RAG_DOC=rag_doc,
        ENGLISH=english,
    )

    chat = [{"role": "user", "content": prompt}]
    chat_str = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return chat_str


# =========================
# Inference
# =========================
def run_inference_single(query: str) -> str:
    decoding_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1,
        max_tokens=4096, skip_special_tokens=True,
        repetition_penalty=1.15,
        frequency_penalty=0.2,
    )

    results = llm.generate(
        [query],
        decoding_params,
        lora_request=lora_request,
    )
    return results[0].outputs[0].text


# =========================
# FastAPI lifecycle
# =========================
@app.on_event("startup")
def startup_event():
    global instruction, wipo_data
    global embeddings, vectorstore
    global tokenizer, llm, lora_request

    # ---- Load instruction template ----
    with open(INSTRUCTION_PATH, "r") as f:
        instruction = f.read()

    # ---- Load WIPO mapping ----
    with open(WIPO_JSON_PATH, "r") as f:
        raw = json.load(f)
        wipo_data = {int(k): v for k, v in raw.items()}

    # ---- Embeddings + FAISS (RAG) ----
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
    )
    vectorstore_local = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # move vectorstore to GPU
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, vectorstore_local.index)
    vectorstore_local.index = gpu_index
    vectorstore = vectorstore_local

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # LLM setup
    llm = LLM(
        model=MODEL_DIR,
        quantization="bitsandbytes",
        max_model_len=4096,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )


# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Main endpoint
# =========================
@app.post("/translate_en2th", response_model=TranslateResponse)
def translate_en2th(req: TranslateRequest):
    chat_str = build_prompt(
        wipo_id=req.wipo_id,
        english=req.english,
    )

    # vLLM inference
    raw_output = run_inference_single(chat_str)
    thai_cleaned = extract_json(raw_output)

    return TranslateResponse(
        thai_raw=raw_output,
        thai_cleaned=thai_cleaned,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
