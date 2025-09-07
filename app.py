# ------------------- Document Q&A with LLaMA-2 (PDF/HTML/Markdown) -------------------

import os, tempfile, traceback
from textwrap import wrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from bs4 import BeautifulSoup
import markdown2
from PyPDF2 import PdfReader
import gradio as gr

# ---------------- CONFIG ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLAMA_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "hf_your_token_here"   # 🔑 replace with your Hugging Face token
CHUNK_CHARS = 1500                # ~300 tokens approx
CHUNK_OVERLAP = 300
TOP_K = 3

# ---------------- GLOBALS ----------------
embedder = SentenceTransformer(EMBED_MODEL)
documents, chunk_sources, index = [], [], None

# ---------------- File saving helper ----------------
def save_uploaded_to_path(f):
    if isinstance(f, str) and os.path.exists(f):
        return f
    if hasattr(f, "name") and os.path.exists(f.name):
        return f.name
    raw = None
    if hasattr(f, "file"):
        try: f.file.seek(0)
        except: pass
        try: raw = f.file.read()
        except: pass
    if raw is None:
        try: raw = f.read()
        except: pass
    if raw is None: raise ValueError("Could not read uploaded file bytes.")
    if isinstance(raw, str): raw = raw.encode("utf-8")
    orig = getattr(f, "orig_name", None) or getattr(f, "name", None) or "upload"
    ext = os.path.splitext(orig)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(raw); tmp.flush()
        return tmp.name

# ---------------- Extractors ----------------
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        name = os.path.basename(path)
        text, sources = "", []
        for i, page in enumerate(reader.pages):
            p = page.extract_text() or ""
            if p:
                text += p + "\n"
                sources.append(f"{name} - Page {i+1}")
        return text, sources
    except Exception as e:
        print("PDF extraction error:", e); traceback.print_exc()
        return "", [os.path.basename(path) + " - PDF (error)"]

def extract_text_from_markdown(path):
    try:
        with open(path, "rb") as fh: raw = fh.read()
        try: md_text = raw.decode("utf-8")
        except: md_text = raw.decode("utf-8", errors="ignore")
        html = markdown2.markdown(md_text)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines), [os.path.basename(path) + " - Markdown"]
    except Exception as e:
        print("Markdown extraction error:", e); traceback.print_exc()
        return "", [os.path.basename(path) + " - MD (error)"]

def extract_text_from_html(path):
    try:
        with open(path, "rb") as fh: raw = fh.read()
        soup = BeautifulSoup(raw, "html.parser")
        for s in soup(["script", "style"]): s.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines), [os.path.basename(path) + " - HTML"]
    except Exception as e:
        print("HTML extraction error:", e); traceback.print_exc()
        return "", [os.path.basename(path) + " - HTML (error)"]

# ---------------- Chunking ----------------
def chunk_text(text, source, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    if not text: return [], []
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end >= L: break
        start = end - overlap
    return chunks, [source] * len(chunks)

# ---------------- Index building ----------------
def process_files(uploaded_files):
    global documents, chunk_sources, index
    documents, chunk_sources = [], []
    for f in uploaded_files:
        try:
            path = save_uploaded_to_path(f)
            fname = path.lower()
            if fname.endswith(".pdf"): txt, sources = extract_text_from_pdf(path)
            elif fname.endswith(".md"): txt, sources = extract_text_from_markdown(path)
            elif fname.endswith(".html") or fname.endswith(".htm"): txt, sources = extract_text_from_html(path)
            else: 
                print("Skipping unsupported:", fname); continue
            if not txt: continue
            chs, srcs = chunk_text(txt, sources[0] if sources else fname)
            documents.extend(chs); chunk_sources.extend(srcs)
        except Exception as e:
            print("Error processing:", e); traceback.print_exc(); continue
    if documents:
        emb = embedder.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        dim = emb.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(emb)
        index = idx
        print("FAISS index built with", index.ntotal, "chunks")
    else:
        print("No docs to index")

# ---------------- Load LLaMA-2 (4-bit) ----------------
print("Loading LLaMA-2 7B Chat (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=HF_TOKEN)
gen_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
gen_pipeline = pipeline("text-generation", model=gen_model, tokenizer=tokenizer, max_new_tokens=300)
print("LLaMA-2 ready!")

# ---------------- RAG query ----------------
def rag_query(query, top_k=TOP_K):
    if index is None or index.ntotal == 0:
        return {"generated_answer": "No documents indexed. Upload files first.", "sources": []}
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(q_emb, top_k)
    retrieved, retrieved_sources = [], []
    for i in I[0]:
        if i < 0 or i >= len(documents): continue
        retrieved.append(documents[i]); retrieved_sources.append(chunk_sources[i])
    context = "\n".join(f"[{s}]\n{c}" for s, c in zip(retrieved_sources, retrieved))
    prompt = f"[INST] You are a helpful assistant. Use the following context to answer the question. Cite sources in [] when relevant.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    resp = gen_pipeline(prompt)[0]["generated_text"]
    return {"generated_answer": resp, "sources": retrieved_sources}

# ---------------- Gradio UI ----------------
def answer_question(uploaded_files, user_question):
    if not uploaded_files: return "Please upload a file.", ""
    if not user_question.strip(): return "Please enter a question.", ""
    process_files(uploaded_files)
    res = rag_query(user_question, top_k=TOP_K)
    return res["generated_answer"], "\n".join(res["sources"])

demo = gr.Interface(
    fn=answer_question,
    inputs=[gr.Files(file_types=[".pdf", ".md", ".html"], label="Upload Files"),
            gr.Textbox(lines=2, placeholder="Ask a question", label="Question")],
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="📄 Document Q&A with LLaMA-2",
    description="Upload PDFs, Markdown, or HTML files and ask natural questions. Uses FAISS + Sentence-Transformers + LLaMA-2 7B Chat (4-bit)."
)

demo.launch()
# ----------------------------------------------------------------------
