# finchat-rag/app_faiss.py
import os
import json
from dotenv import load_dotenv
import requests
import numpy as np
import faiss
import time

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_EMBED_DEPLOYMENT and AZURE_CHAT_DEPLOYMENT):
    raise RuntimeError("Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_EMBEDDING_DEPLOYMENT, AZURE_CHAT_DEPLOYMENT")

INDEX_DIR = "finchat-rag/faiss_index"
INDEX_FILE = f"{INDEX_DIR}/faiss.index"
META_FILE = f"{INDEX_DIR}/metadata.json"

if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
    raise RuntimeError("No FAISS index / metadata found. Run ingest_faiss.py first.")

HEADERS = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}

# embeddings via REST
def azure_embed(texts):
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_EMBED_DEPLOYMENT}/embeddings?api-version={AZURE_API_VERSION}"
    payload = {"input": texts}
    for attempt in range(4):
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 200:
            return [item["embedding"] for item in r.json()["data"]]
        if r.status_code in (429, 503):
            time.sleep(1 * (2 ** attempt))
            continue
        raise RuntimeError(f"Embedding error: {r.status_code} {r.text}")
    raise RuntimeError("Embedding retries failed")

# chat completion via REST
def azure_chat(prompt, max_tokens=512, temperature=0.0):
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_CHAT_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    for attempt in range(4):
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 200:
            j = r.json()
            # Extract message content (Azure chat completions)
            try:
                return j["choices"][0]["message"]["content"]
            except Exception:
                return str(j)
        if r.status_code in (429, 503):
            time.sleep(1 * (2 ** attempt))
            continue
        raise RuntimeError(f"Chat error: {r.status_code} {r.text}")
    raise RuntimeError("Chat retries failed")

# load FAISS + metadata
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as fh:
    meta = json.load(fh)
texts = meta["texts"]
metadatas = meta["metadatas"]

def embed_query(q):
    emb = azure_embed([q])[0]
    return np.array(emb, dtype="float32")

def retrieve(query, top_k=4):
    v = embed_query(query).reshape(1, -1)
    D, I = index.search(v, top_k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(texts):
            continue
        hits.append({"text": texts[idx], "meta": metadatas[idx], "score": float(D[0][rank])})
    return hits

def make_prompt(query, retrieved):
    context = "\n\n---\n\n".join([f"Source: {r['meta']['source']} (chunk {r['meta']['chunk_id']})\n{r['text']}" for r in retrieved])
    prompt = f"""You are FinChat, a finance assistant.
Use ONLY the provided context to answer. If answer not present, say 'Not enough data'.

Context:
{context}

Question:
{query}

Answer concisely and list the source filenames and chunk ids at the end."""
    return prompt

def ask(query, top_k=4):
    retrieved = retrieve(query, top_k=top_k)
    if not retrieved:
        return "No documents found in the index."
    prompt = make_prompt(query, retrieved)
    return azure_chat(prompt)

def main():
    print("FinChat (Azure + FAISS) ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            ans = ask(q, top_k=4)
        except Exception as e:
            ans = f"Error: {e}"
        print("\nFinChat:", ans)
        print("-" * 60)

if __name__ == "__main__":
    main()
