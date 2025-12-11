#!/usr/bin/env python3
"""
ingest_faiss.py

Usage:
  - Put .txt or .pdf files under finchat-rag/data/
  - Create a .env with AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_EMBEDDING_DEPLOYMENT
  - pip install -r requirements.txt
    Minimum: requests numpy faiss-cpu tqdm pdfplumber python-dotenv
    Optional (better chunking): langchain
    Optional (OCR for scanned PDFs): pillow pytesseract

This script:
  - loads text files and PDFs
  - chunks them (langchain splitter if available, else word-based)
  - gets embeddings from Azure OpenAI (batching + retries)
  - normalizes embeddings and builds a FAISS index (IndexFlatIP for cosine)
  - saves index and metadata
"""
from pathlib import Path
import os
import json
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import faiss
import pdfplumber
import traceback

# optional imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# optional OCR imports
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# load env
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_EMBED_DEPLOYMENT):
    raise RuntimeError(
        "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_EMBEDDING_DEPLOYMENT in your .env"
    )

# paths
DATA_DIR = Path("finchat-rag/data")
INDEX_DIR = Path("finchat-rag/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "faiss.index"

HEADERS = {
    "api-key": AZURE_API_KEY,
    "Content-Type": "application/json"
}

# ---------------- Azure embeddings (batch + retries) ----------------
def azure_embeddings(texts, sleep_on_error=2.0):
    """
    texts: list[str] or single str (we send list)
    Returns list of embedding vectors (list of floats)
    """
    if isinstance(texts, str):
        texts = [texts]

    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_EMBED_DEPLOYMENT}/embeddings?api-version={AZURE_API_VERSION}"
    payload = {"input": texts}

    for attempt in range(4):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        except Exception as e:
            # network-level error -> retry with backoff
            print(f"[azure_embeddings] request exception attempt {attempt}: {e}")
            time.sleep(sleep_on_error * (2 ** attempt))
            continue

        if r.status_code == 200:
            j = r.json()
            if "data" not in j:
                raise RuntimeError(f"Unexpected response shape: {j}")
            return [item["embedding"] for item in j["data"]]
        else:
            msg = f"Embedding request failed: {r.status_code} {r.text}"
            print("[azure_embeddings]", msg)
            if r.status_code in (429, 503):
                # retry with exponential backoff
                time.sleep(sleep_on_error * (2 ** attempt))
                continue
            # other error -> raise
            raise RuntimeError(msg)

    raise RuntimeError("Embeddings failed after retries")


# ---------------- chunking ----------------
def chunk_text_with_langchain(text, chunk_size_chars=1000, overlap_chars=200):
    """
    Uses LangChain's RecursiveCharacterTextSplitter (characters-based, preserves sentences).
    Returns list[str]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " "],
    )
    return splitter.split_text(text)


def chunk_text_word_based(text, chunk_size_words=400, overlap_words=50):
    """
    Fallback word-based chunking (returns list of chunks)
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = chunk_size_words - overlap_words
    if step <= 0:
        step = chunk_size_words
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size_words])
        chunks.append(chunk)
        i += step
    return chunks


def chunk_text(text,
               use_langchain=True,
               langchain_chunk_chars=1000,
               langchain_overlap_chars=200,
               word_chunk_size=400,
               word_overlap=50):
    """
    Unified chunk_text function: prefer langchain if available and requested else fallback.
    """
    if use_langchain and LANGCHAIN_AVAILABLE:
        return chunk_text_with_langchain(text, chunk_size_chars=langchain_chunk_chars, overlap_chars=langchain_overlap_chars)
    else:
        return chunk_text_word_based(text, chunk_size_words=word_chunk_size, overlap_words=word_overlap)


# ---------------- load text (with optional OCR fallback) ----------------
def load_text(path: Path, try_ocr_if_empty: bool = True) -> str:
    """
    Reads text from .txt or .pdf. For PDFs, attempts pdfplumber extraction.
    If extraction yields empty string and OCR is available + try_ocr_if_empty, tries OCR.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = []
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    try:
                        txt = p.extract_text()
                        pages.append(txt or "")
                    except Exception as e:
                        print(f"[load_text] page extraction error in {path}: {e}")
                        pages.append("")
        except Exception as e:
            print(f"[load_text] pdfplumber could not open {path}: {e}")
            pages = []

        combined = "\n".join(pages).strip()
        if combined:
            return combined

        # fallback: OCR if available and requested
        if try_ocr_if_empty and OCR_AVAILABLE:
            print(f"[load_text] PDF {path} had no extracted text — trying OCR (pytesseract).")
            try:
                images = []
                from pdf2image import convert_from_path  # optional, might not be installed
                # try convert_from_path first (faster if available)
                try:
                    pil_pages = convert_from_path(str(path))
                except Exception:
                    pil_pages = []
                # fallback to pdfplumber images if convert_from_path not available
                if not pil_pages:
                    try:
                        with pdfplumber.open(path) as pdf:
                            for p in pdf.pages:
                                im = p.to_image(resolution=200).original
                                pil_pages.append(im)
                    except Exception:
                        pil_pages = []

                ocr_texts = []
                for im in pil_pages:
                    try:
                        txt = pytesseract.image_to_string(im)
                        ocr_texts.append(txt or "")
                    except Exception as e:
                        print(f"[load_text] pytesseract error on page: {e}")
                combined_ocr = "\n".join(ocr_texts).strip()
                return combined_ocr
            except Exception as e:
                print(f"[load_text] OCR fallback failed for {path}: {e}")

        return ""  # nothing extracted
    else:
        # generic text file
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[load_text] Could not read {path}: {e}")
            return ""


# ---------------- main ingestion ----------------
def main(
    use_langchain_chunking: bool = True,
    langchain_chunk_chars: int = 1000,
    langchain_overlap_chars: int = 200,
    word_chunk_size: int = 400,
    word_overlap: int = 50,
    batch_size: int = 32,
    sim_index_type: str = "cosine",  # "cosine" (normalize + IndexFlatIP) or "l2" (IndexFlatL2)
):
    print("Scanning:", DATA_DIR.resolve())
    if not DATA_DIR.exists():
        print("Create finchat-rag/data/ and add .txt or .pdf files.")
        return

    files = sorted(list(DATA_DIR.glob("**/*.*")))
    print(f"Found {len(files)} files to ingest.")
    if not files:
        print("No files found. Put .txt or .pdf files under finchat-rag/data/")
        return

    all_texts = []
    metadatas = []
    skipped = 0

    for f in files:
        try:
            txt = load_text(f)
        except Exception as e:
            print(f"[main] Failed to load {f}: {e}\n{traceback.format_exc()}")
            continue

        if not txt or not txt.strip():
            print(f"Skipping (empty): {f} (size={f.stat().st_size} bytes)")
            skipped += 1
            continue

        chunks = chunk_text(
            txt,
            use_langchain=use_langchain_chunking,
            langchain_chunk_chars=langchain_chunk_chars,
            langchain_overlap_chars=langchain_overlap_chars,
            word_chunk_size=word_chunk_size,
            word_overlap=word_overlap,
        )

        if not chunks:
            print(f"No chunks produced for {f} (chars={len(txt)}) — skipping")
            skipped += 1
            continue

        print(f"File: {f}  | chars: {len(txt)} | chunks: {len(chunks)}")
        # optionally show a sample
        sample = chunks[0][:300].replace("\n", " ")
        print(f"  sample chunk[0]: {sample!r}...")

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            metadatas.append({"source": str(f), "chunk_id": i})

    print(f"Total files: {len(files)}, skipped: {skipped}, chunks to embed: {len(all_texts)}")

    if not all_texts:
        print("No chunks found. Add .txt/.pdf to finchat-rag/data/ or enable OCR for scanned PDFs.")
        return

    # batch embeddings
    embeddings = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding batches"):
        batch = all_texts[i:i + batch_size]
        try:
            embs = azure_embeddings(batch)
        except Exception as e:
            print(f"[main] Azure embeddings failed on batch starting at {i}: {e}")
            raise
        if not embs or len(embs) != len(batch):
            raise RuntimeError(f"Unexpected embeddings length: got {len(embs)} for batch of {len(batch)}")
        embeddings.extend(embs)

    # convert to numpy float32
    xb = np.array(embeddings, dtype="float32")
    emb_dim = xb.shape[1]
    print(f"Embeddings shape: {xb.shape}  (dim={emb_dim})")

    # Normalize if using cosine similarity
    if sim_index_type == "cosine":
        # normalize rows in-place (faiss helper)
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(emb_dim)  # inner product on normalized vectors == cosine
    else:
        index = faiss.IndexFlatL2(emb_dim)

    index.add(xb)
    print(f"FAISS index size (ntotal): {index.ntotal}")

    # ensure index dir exists
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Wrote index -> {INDEX_FILE}")

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as fh:
        json.dump({"texts": all_texts, "metadatas": metadatas}, fh, ensure_ascii=False, indent=2)
    print(f"Wrote metadata -> {META_FILE}")

    print(f"Ingested {len(all_texts)} chunks. Saved index -> {INDEX_FILE}")


if __name__ == "__main__":
    if LANGCHAIN_AVAILABLE:
        print("langchain available — using RecursiveCharacterTextSplitter for chunking by default.")
    else:
        print("langchain not installed — using fallback word-based chunking. To enable better chunking, pip install langchain.")

    if OCR_AVAILABLE:
        print("pytesseract available — OCR fallback for scanned PDFs is enabled (if pdf2image or pdfplumber image fallback works).")
    else:
        print("pytesseract not available — scanned PDFs without embedded text will not be OCRed.")

    main()
