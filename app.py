import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import requests
import unicodedata
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG Document Analyzer", layout="wide")

st.title("📄 Scientific & Legal Document Analyzer (Hybrid RAG)")
st.caption("Powered by BM25 + FAISS Vector Search, LLM Generation & Citation Validation")

@st.cache_resource
def load_embedder():
    # BAAI/bge-small-en-v1.5 model for dense embeddings
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedder = load_embedder()

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def clean_extracted_text(text):
    """Normalizes unicode and strips unprintable/corrupt byte replacement artifacts."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(ch for ch in normalized if ch.isprintable() or ch in ['\n', ' '])
    return cleaned.replace("\ufffd", "").replace("", "").strip()

def parse_and_chunk_pdf(uploaded_file, chunk_size=512, chunk_overlap=64):
    """
    Extracts text using pdfplumber to bypass font CMap encoding issues.
    Falls back to PyMuPDF with text sanitization if pdfplumber yields empty output.
    """
    chunks = []
    uploaded_file.seek(0)
    
    # Primary Method: pdfplumber
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                clean_text = clean_extracted_text(raw_text)
                
                words = clean_text.split()
                if len(words) > 5:
                    for i in range(0, len(words), chunk_size - chunk_overlap):
                        chunk_text = " ".join(words[i:i + chunk_size])
                        if len(chunk_text.strip()) > 20:
                            chunks.append({
                                "text": chunk_text,
                                "page": page_num + 1,
                                "id": f"p{page_num+1}_c{i}"
                            })
    except Exception:
        pass

    # Fallback Method: PyMuPDF Block Extraction
    if not chunks:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            raw_text = "\n".join([b[4] for b in blocks if len(b) > 4 and b[4].strip()])
            clean_text = clean_extracted_text(raw_text)
            
            words = clean_text.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_text = " ".join(words[i:i + chunk_size])
                if len(chunk_text.strip()) > 20:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num + 1,
                        "id": f"p{page_num+1}_c{i}"
                    })

    return chunks

def hybrid_rrf_search(query, chunks, top_k=3, k=60):
    """Combines BM25 (Sparse) and Dense Vector Search via Reciprocal Rank Fusion."""
    corpus_texts = [c["text"] for c in chunks]
    
    # Sparse BM25 Search
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    # Dense Vector Search
    query_vec = embedder.encode([query])
    doc_vecs = embedder.encode(corpus_texts, show_progress_bar=False)
    dense_scores = cosine_similarity(query_vec, doc_vecs)[0]
    dense_top = np.argsort(dense_scores)[::-1][:top_k * 2]
    
    # Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(dense_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[idx] for idx, score in sorted_indices]

def generate_llm_answer(query, context, hf_api_key=""):
    """Generates response via Hugging Face API or uses an extractive fallback."""
    
    # Safely evaluate st.secrets without crashing on local execution
    if not hf_api_key:
        try:
            if "HF_API_KEY" in st.secrets:
                hf_api_key = st.secrets["HF_API_KEY"]
        except Exception:
            pass

    if hf_api_key and hf_api_key.strip():
        API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {hf_api_key.strip()}"}
        
        prompt = f"""<s>[INST] You are an academic assistant. Answer the question concisely and accurately based ONLY on the provided context.

Context:
{context}

Question: {query} [/INST]"""
        
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.2}},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].split("[/INST]")[-1].strip()
        except Exception:
            pass  # Fall back to rule-based synthesis on timeout or error

    # Local Extractive Fallback Engine
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
    top_sentences = sentences[:3] if len(sentences) >= 3 else sentences
    return f"Based on the retrieved context: {' '.join(top_sentences)}."

# -----------------------------------------------------------------------------
# 3. USER INTERFACE & MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
hf_api_key = st.sidebar.text_input("Hugging Face API Key (Optional)", type="password")

st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing & Indexing PDF..."):
        chunks = parse_and_chunk_pdf(uploaded_file)
        if len(chunks) > 0:
            st.sidebar.success(f"Successfully indexed {len(chunks)} text chunks!")
        else:
            st.sidebar.error("Could not extract readable text. The document might be scanned or image-only.")

    query = st.text_input("Enter your question about the document:")
    
    if query and len(chunks) > 0:
        # Step 1: Search relevant chunks
        retrieved_chunks = hybrid_rrf_search(query, chunks, top_k=3)
        best_context = retrieved_chunks[0]['text']
        
        # Step 2: Answer generation
        with st.spinner("Generating answer..."):
            generated_answer = generate_llm_answer(query, best_context, hf_api_key)
        
        st.subheader("🤖 Generated Answer")
        st.info(generated_answer)
        
        # Step 3: Context & Citations
        st.subheader("🔍 Retrieved Citations & Context")
        for idx, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Source Chunk #{idx+1} (Page {chunk['page']})", expanded=(idx == 0)):
                st.write(chunk['text'])
        
        # Step 4: Metric Evaluations
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(best_context, generated_answer)['rougeL'].fmeasure
        
        ans_vec = embedder.encode([generated_answer])
        ctx_vec = embedder.encode([best_context])
        sem_sim = cosine_similarity(ans_vec, ctx_vec)[0][0]

        st.subheader("📊 Citation Confidence Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Semantic Similarity (Answer vs. Source)", f"{sem_sim:.4f}")
        col2.metric("ROUGE-L Groundedness Score", f"{rouge_l:.4f}")
else:
    st.info("Please upload a PDF document from the sidebar to get started.")