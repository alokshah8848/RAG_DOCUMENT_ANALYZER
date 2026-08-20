import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import requests
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
    # Dense embedding model for semantic search & metrics evaluation
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedder = load_embedder()

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def parse_and_chunk_pdf(uploaded_file, chunk_size=512, chunk_overlap=64):
    """Extracts text from PDF and splits it into chunk dictionaries with page metadata."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        words = text.split()
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
    """ Combines BM25 (Lexical) and Dense Vector (Semantic) Search using Reciprocal Rank Fusion."""
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
    
    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(dense_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[idx] for idx, score in sorted_indices]

def generate_llm_answer(query, context, hf_api_key=""):
    """Generates an answer using Hugging Face Inference API or falls back to extractive response."""
    if hf_api_key.strip():
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {hf_api_key.strip()}"}
        
        prompt = f"""<s>[INST] You are an academic assistant. Answer the question concise and accurately based ONLY on the provided context.

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
            pass  # Fall back to local synthesis if API fails or times out

    # Rule-Based / Fallback Summary Generation if API key is not provided or fails
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
    top_sentences = sentences[:3] if len(sentences) >= 3 else sentences
    return f"Based on the retrieved context: {' '.join(top_sentences)}."

# -----------------------------------------------------------------------------
# 3. USER INTERFACE & MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
hf_api_key = st.sidebar.text_input("Hugging Face API Key (Optional)", type="password", help="Enter free HF token to enable full LLM generation.")

st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing & Indexing PDF..."):
        chunks = parse_and_chunk_pdf(uploaded_file)
        st.sidebar.success(f"Successfully indexed {len(chunks)} text chunks!")

    query = st.text_input("Enter your question about the document:")
    
    if query:
        # Step 1: Perform Hybrid Search
        retrieved_chunks = hybrid_rrf_search(query, chunks, top_k=3)
        best_context = retrieved_chunks[0]['text']
        
        # Step 2: Generate Answer
        with st.spinner("Generating answer..."):
            generated_answer = generate_llm_answer(query, best_context, hf_api_key)
        
        st.subheader("🤖 Generated Answer")
        st.info(generated_answer)
        
        # Step 3: Display Citations
        st.subheader("🔍 Retrieved Citations & Context")
        for idx, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Source Chunk #{idx+1} (Page {chunk['page']})", expanded=(idx == 0)):
                st.write(chunk['text'])
        
        # Step 4: Evaluate Metrics (Answer vs. Retrieved Context)
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