from nltk.corpus import stopwords
from datasets import Dataset
import re

# ============
# setup 
# ============
def retrieval_setup(_embeddings, _vectorstore, _cur, _instruction, _k):
    """Call this once from your main pipeline."""
    global embeddings, vectorstore, cur, instruction, top_k
    embeddings = _embeddings
    vectorstore = _vectorstore
    cur = _cur
    instruction = _instruction
    top_k = _k

# ======================================
# Retrieval Augmented Generation (RAG)
# ======================================
def rag_relevant_docs(query: str):
    """return top k most cosine similarity score"""
    query_embedding = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    results = [(doc.page_content, doc.metadata['thai']) for doc in docs]
    return results

# ======================================
# Full Text Search (FTS)
# ======================================
def construct_query(text: str):
    """construct condition matching for SQL query via fts5"""
    cleaned_text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in cleaned_text.split() if token not in stop_words]
    tokens = [f"ENG:{t} OR " for t in tokens]
    return "".join(tokens)[:-3]

def fts_relevant_docs(eng_text: str):
    """bm25 search via fts5 in SQL query"""
    query = construct_query(eng_text)
    res = cur.execute(f"""SELECT ENG, THA, bm25(wipo_table) AS rank
    FROM wipo_table
    WHERE wipo_table MATCH "{query}"
    ORDER BY rank
    LIMIT {top_k};
    """).fetchall()
    return res

# ======================================
# Formatting retrieved data
# ======================================
def process_query(query: str, how: str, skip_first: bool):
    """construct references prompt"""
    docs = fts_relevant_docs(query) if how=="fts" else rag_relevant_docs(query)
    docs = docs[1:] if skip_first else docs
    
    relevant = "\n## Retrieved References:\n"
    for doc in docs:
        relevant += f'''
English: {doc[0]}
Thai: {doc[1]}
'''
    return relevant + '''\n**Note:** If the retrieved references contain identical English terms with different Thai translations (ambiguity), you must use your expert judgment to select the most appropriate and contextually accurate translation for the current input.\n'''


# ============
# utility
# ============
def formatting_prompt(df, tokenizer, how="rag", skip_first=True, finetuning=False):
    batch = []
    for _, row in df.iterrows():
        prompt = instruction.format(
            WIPO=row['WIPO'],
            # RAG_DOC="",
            RAG_DOC=process_query(row['ENG'], how, skip_first),
            ENGLISH=row["ENG"]
        )
        chat = [{"role": "user", "content": prompt}]
        if finetuning:
            chat += [{"role": "assistant", "content": f"{{\"thai_translation\": {row['THA']} }}"}]
        chat = tokenizer.apply_chat_template(
            chat, tokenize=False, 
            add_generation_prompt=False if finetuning else True
        )
        batch.append({"text": chat})
    return Dataset.from_list(batch)
    
