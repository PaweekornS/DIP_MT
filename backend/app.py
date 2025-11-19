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

EN2TH_PROMPT = f"{ROOT_DIR}/data/prompt/base_en2th.txt"
TH2EN_PROMPT = f"{ROOT_DIR}/data/prompt/base_th2en.txt"
WIPO_JSON_PATH = f"{ROOT_DIR}/data/wipo/WIPO.json"

EMBED_MODEL_NAME = "bge-m3"
EMBED_MODEL_PATH = f"{ROOT_DIR}/models/retriever/{EMBED_MODEL_NAME}"

ENG_FAISS_INDEX = f"{ROOT_DIR}/vector/en2th/{EMBED_MODEL_NAME}"
THA_FAISS_INDEX = f"{ROOT_DIR}/vector/th2en/{EMBED_MODEL_NAME}"

# model config – สามารถเปลี่ยนมาใช้ env var ได้
MODEL_DIR = os.getenv("MODEL_DIR", f"{ROOT_DIR}/models/base/gemma3-4b-it")
ADAPTER_DIR = None

os.environ['VLLM_CONFIGURE_LOGGING'] = "0"

# =========================
# FastAPI schemas
# =========================
class EnglishRequest(BaseModel):
    wipo_id: int
    english: str
    
class ThaiRequest(BaseModel):
    wipo_id: int
    thai: str

class TranslateResponse(BaseModel):
    translation: str

# =========================
# Utility: Thai filtering / JSON extraction
# =========================
def filter_thai(text: str) -> str:
    pattern = r'[\u0e00-\u0e7f\s,.?!]+'
    matches = re.findall(pattern, text)
    return "".join(matches).strip().replace("\n", "")


def extract_json(text: str, en2th=True) -> str:
    text = text[text.rfind("{"):]
    if en2th:
        pattern = r'''{\s*[\'\"]thai_translation[\'\"]:\s*[\'\"].*?[\'\"]\s*}'''
    else:
        pattern = r'''{\s*[\'\"]eng_translation[\'\"]:\s*[\'\"].*?[\'\"]\s*}'''
        
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        try:
            loaded = json.loads(matches[0])
            return loaded['thai_translation'] if en2th else loaded['eng_translation']
        except json.JSONDecodeError:
            return filter_thai(text)
    else:
        return filter_thai(text)


# =========================
# Retrieval
# =========================
def get_relevant_docs(query: str, k: int = 3, isEnglish: bool = True) -> str:
    query_embedding = embeddings.embed_query(query)
    if isEnglish:
        docs = eng_vectorstore.similarity_search_by_vector(query_embedding, k=k)

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
    else:
        docs = th_vectorstore.similarity_search_by_vector(query_embedding, k=k)
        relevant = ""
        for doc in docs:
            relevant += f'''Thai: {doc.page_content}
            English: {doc.metadata.get("eng", "")}\n'''
        rag_result = relevant
    return rag_result


# =========================
# Prompt formatting
# =========================
def build_en2th_prompt(wipo_id: int, english: str) -> tuple[str, Optional[str]]:
    wipo_label = wipo_data.get(int(wipo_id), "")
    rag_doc = get_relevant_docs(english, 3, isEnglish=True)

    prompt = en2th_instruction.format(
        WIPO=wipo_label,
        RAG_DOC=rag_doc,
        ENGLISH=english,
    )

    chat = [{"role": "user", "content": prompt}]
    chat_str = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return chat_str


def build_th2en_prompt(wipo_id: int, thai: str) -> tuple[str, Optional[str]]:
    wipo_label = wipo_data.get(int(wipo_id), "")
    rag_doc = get_relevant_docs(thai, 3, isEnglish=False)

    prompt = th2en_instruction.format(
        WIPO=wipo_label,
        RAG_DOC=rag_doc,
        THAI=thai,
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
        max_tokens=8192, skip_special_tokens=True,
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
app = FastAPI(title="En2Th Translation API (vLLM + RAG)")

@app.on_event("startup")
def startup_event():
    global en2th_instruction, th2en_instruction, wipo_data
    global embeddings, eng_vectorstore, th_vectorstore
    global tokenizer, llm, lora_request

    # ---- Load instruction template ----
    with open(EN2TH_PROMPT, "r") as f:
        en2th_instruction = f.read()
        
    with open(TH2EN_PROMPT, "r") as f:
        th2en_instruction = f.read()

    # ---- Load WIPO mapping ----
    with open(WIPO_JSON_PATH, "r") as f:
        raw = json.load(f)
        wipo_data = {int(k): v for k, v in raw.items()}

    # ---- Embeddings + FAISS (RAG) ----
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
    )
    eng_vectorstore = FAISS.load_local(
        ENG_FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    th_vectorstore = FAISS.load_local(
        THA_FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # move vectorstore to GPU
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, eng_vectorstore.index)
    eng_vectorstore.index = gpu_index

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # LLM setup
    llm = LLM(
        model=MODEL_DIR,
        quantization="bitsandbytes",
        max_model_len=8192,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    lora_request=None


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
def translate_en2th(req: EnglishRequest):
    chat_str = build_en2th_prompt(
        wipo_id=req.wipo_id,
        english=req.english,
    )

    # vLLM inference
    raw_output = run_inference_single(chat_str)
    thai_cleaned = extract_json(raw_output, en2th=True)

    return TranslateResponse(
        translation=thai_cleaned,
    )
    
@app.post("/translate_th2en", response_model=TranslateResponse)
def translate_th2en(req: ThaiRequest):
    chat_str = build_th2en_prompt(
        wipo_id=req.wipo_id,
        thai=req.thai,
    )

    # vLLM inference
    raw_output = run_inference_single(chat_str)
    eng_cleaned = extract_json(raw_output, en2th=False)

    return TranslateResponse(
        translation=eng_cleaned,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
