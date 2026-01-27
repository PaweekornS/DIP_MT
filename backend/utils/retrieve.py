from vllm import SamplingParams
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from typing import Optional
import json
import re
import os

# =============
# Setup
# =============
ROOT_DIR = os.getenv("ROOT_DIR") or ("/app" if os.path.exists("/app") else "./")
ROOT_DIR = os.path.abspath(ROOT_DIR)

MODEL_ID = os.getenv("MODEL_ID", "unsloth/gemma-3-1b-it")
WIPO_DICT = os.path.join(ROOT_DIR, "data", "WIPO.json")
EN2TH_PROMPT = os.path.join(ROOT_DIR, "data", "base_en2th.txt")
FAISS_INDEX = os.path.join(ROOT_DIR, "vectorstore")

with open(WIPO_DICT, "r") as f:
    raw = json.load(f)
    wipo_data = {int(k): v for k, v in raw.items()}
    
with open(EN2TH_PROMPT, "r") as f:
    instruction = f.read()
    
# ---- Embeddings + FAISS (RAG) ----
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",)
vectorstore = FAISS.load_local(
    FAISS_INDEX,
    embeddings,
    allow_dangerous_deserialization=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# ================
# extract answer
# ================
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


# ==============
# Retrieval
# ==============
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
def build_en2th_prompt(wipo_id: int, english: str) -> tuple[str, Optional[str]]:
    wipo_label = wipo_data.get(int(wipo_id), "")
    rag_doc = get_relevant_docs(english, 3, isEnglish=True)

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
def inference_mt(llm, query: str) -> str:
    decoding_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1,
        max_tokens=8192, skip_special_tokens=True,
        repetition_penalty=1.15,
        frequency_penalty=0.2,
    )

    results = llm.generate(
        [query],
        decoding_params,
    )
    return results[0].outputs[0].text