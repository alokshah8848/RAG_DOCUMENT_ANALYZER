import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

st.set_page_config(page_title="RAG Document Analyzer", layout="wide")
st.title("📄 Scientific & Legal Document Analyzer (Hybrid RAG)")
st.caption("Powered by BM25 + FAISS Vector Search & Citation Validation")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedder = load_embedder()

def parse_and_chunk_pdf(uploaded_file, chunk_size=512, chunk_overlap=64):
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
    corpus_texts = [c["text"] for c in chunks]
    
    # 1. Sparse BM25
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k*2]
    
    # 2. Dense Vector
    query_vec = embedder.encode([query])
    doc_vecs = embedder.encode(corpus_texts, show_progress_bar=False)
    dense_scores = cosine_similarity(query_vec, doc_vecs)[0]
    dense_top = np.argsort(dense_scores)[::-1][:top_k*2]
    
    # 3. Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(dense_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[idx] for idx, score in sorted_indices]

st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing & Indexing PDF..."):
        chunks = parse_and_chunk_pdf(uploaded_file)
        st.sidebar.success(f"Indexed {len(chunks)} text chunks!")

    query = st.text_input("Enter your question about the document:")
    
    if query:
        retrieved_chunks = hybrid_rrf_search(query, chunks, top_k=3)
        
        st.subheader("🔍 Retrieved Citations & Context")
        for idx, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Source Chunk #{idx+1} (Page {chunk['page']})", expanded=True):
                st.write(chunk['text'])
        
        best_context = retrieved_chunks[0]['text']
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(best_context, query)['rougeL'].fmeasure
        
        q_vec = embedder.encode([query])
        c_vec = embedder.encode([best_context])
        sem_sim = cosine_similarity(q_vec, c_vec)[0][0]

        st.subheader("📊 Citation Confidence Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Semantic Similarity", f"{sem_sim:.4f}")
        col2.metric("ROUGE-L Score", f"{rouge_l:.4f}")
else:
    st.info("Please upload a PDF file from the sidebar to begin.")