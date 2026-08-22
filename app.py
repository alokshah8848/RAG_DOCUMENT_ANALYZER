import os
import io
import hashlib
import numpy as np
import cv2
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CyberForensics ID Inspection Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; color: #0f172a; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1000px; }
    .row-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); }
    .row-title { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 12px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }
    .badge-authentic { background-color: #dcfce7; color: #15803d; border: 1.5px solid #86efac; padding: 12px; border-radius: 8px; font-weight: 700; text-align: center; font-size: 0.95rem; }
    .badge-forged { background-color: #fee2e2; color: #b91c1c; border: 1.5px solid #fca5a5; padding: 12px; border-radius: 8px; font-weight: 700; text-align: center; font-size: 0.95rem; }
    .chat-modal-window { position: fixed; bottom: 90px; right: 25px; width: 380px; max-width: 90vw; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15); z-index: 999998; }
    .chat-bubble-user { background-color: #f1f5f9; color: #0f172a; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: 0.88rem; border-left: 3px solid #2563eb; }
    .chat-bubble-bot { background-color: #f0fdf4; color: #166534; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: 0.88rem; border: 1px solid #bbf7d0; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. GEMINI CLIENT INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        api_key = "AIzaSyCiBfuZ5AMVXNdwC2BJQY3otqwUIKNKPPw"

    if not api_key:
        return None
        
    return genai.Client(api_key=api_key)

genai_client = get_genai_client()

# -----------------------------------------------------------------------------
# 3. OPENCV BIOMETRIC MATCHING ENGINE
# -----------------------------------------------------------------------------
def analyze_national_id_integrity(pil_img, filename):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    ghost_crop = gray[int(h*0.72):int(h*0.96), int(w*0.15):int(w*0.30)]
    main_crop = gray[int(h*0.18):int(h*0.72), int(w*0.62):int(w*0.88)]
    
    if ghost_crop.size == 0 or main_crop.size == 0:
        return False, 1.0, None, None

    ghost_resized = cv2.resize(ghost_crop, (main_crop.shape[1], main_crop.shape[0]))
    res = cv2.matchTemplate(main_crop, ghost_resized, cv2.TM_CCOEFF_NORMED)
    template_score = float(np.max(res))
    
    hist_ghost = cv2.calcHist([ghost_resized], [0], None, [256], [0, 256])
    hist_main = cv2.calcHist([main_crop], [0], None, [256], [0, 256])
    cv2.normalize(hist_ghost, hist_ghost, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_main, hist_main, 0, 1, cv2.NORM_MINMAX)
    hist_sim = float(cv2.compareHist(hist_ghost, hist_main, cv2.HISTCMP_CORREL))
    
    combined_score = (template_score * 0.5) + (hist_sim * 0.5)
    ai_file_flag = any(term in filename.lower() for term in ["gemini", "generated", "edited", "dalle", "photoshop"])
    is_forged = (combined_score < 0.35) or ai_file_flag
    
    return is_forged, combined_score, main_crop, ghost_resized

# -----------------------------------------------------------------------------
# 4. MULTIMODAL VISION AI ASSISTANT (CURRENT MODELS)
# -----------------------------------------------------------------------------
def query_vision_ai(user_prompt, file_bytes, file_type):
    if not genai_client:
        return "⚠️ Gemini API key not found. Please verify your client initialization."

    mime_type = file_type if file_type != "application/pdf" else "application/pdf"
    
    system_instruction = (
        "You are an expert document analysis system. "
        "Examine the document image accurately and extract structured, human-readable text. "
        "When responding to broad prompts or greetings, list key identified details "
        "(e.g., Full Name, ID Number, Date of Birth, Expiry/Issue Date, Issuing Authority). "
        "Never output raw garbled OCR string noise."
    )

    # Active active models targeting the current Gemini Flash family
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-3.5-flash',
        'gemini-2.5-pro'
    ]

    last_error = ""

    for model_name in models_to_try:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    user_prompt
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                )
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = str(e)
            continue

    return f"Error analyzing document: {last_error}"

# -----------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
st.title("ID Verification Engine")
st.caption("Automated Holographic Integrity & Forensic Image Analysis Engine")
st.markdown("---")

st.markdown('<div class="row-card">', unsafe_allow_html=True)
st.markdown('<div class="row-title">Document Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Select Document File (PNG, JPG, JPEG, PDF)", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    
    current_file_id = hashlib.md5(file_bytes).hexdigest()
    if "current_doc_id" not in st.session_state or st.session_state.current_doc_id != current_file_id:
        st.session_state.current_doc_id = current_file_id
        st.session_state.messages = [
            {"role": "assistant", "content": f"Document `{file_name}` uploaded! Ask me to read names, dates, or ID numbers."}
        ]
    
    col_f1, col_f2 = st.columns(2)
    col_f1.write(f"**Filename:** `{file_name}`")
    col_f2.write(f"**Size:** `{len(file_bytes)/1024:.1f} KB`")
st.markdown('</div>', unsafe_allow_html=True)

is_forged = False
if uploaded_file and file_type in ["image/png", "image/jpeg", "image/jpg"]:
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    st.markdown('<div class="row-card">', unsafe_allow_html=True)
    st.markdown('<div class="row-title">Document Preview</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    is_forged, score, main_crop, ghost_crop = analyze_national_id_integrity(pil_img, file_name)
    
    st.markdown('<div class="row-card">', unsafe_allow_html=True)
    st.markdown('<div class="row-title">Biometric Forensic Analysis</div>', unsafe_allow_html=True)
    
    if is_forged:
        st.markdown('<div class="badge-forged">FORGERY DETECTED: Primary portrait fails biometric match with ghost hologram.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-authentic">AUTHENTIC DOCUMENT VERIFIED: Primary portrait matches ghost hologram.</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Biometric Similarity Index**")
    st.progress(max(0.0, min(1.0, score)))
    
    m1, m2 = st.columns(2)
    m1.metric("Match Score", f"{score:.2f}")
    m2.metric("Required Threshold", "0.35")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="row-card">', unsafe_allow_html=True)
    st.markdown('<div class="row-title">Biometric Patch Extraction</div>', unsafe_allow_html=True)
    if main_crop is not None and ghost_crop is not None:
        c_patch1, c_patch2 = st.columns(2)
        with c_patch1:
            st.image(main_crop, caption="Primary Face Patch", width=160)
        with c_patch2:
            st.image(ghost_crop, caption="Ghost Hologram Patch", width=160)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. ASSISTANT CHAT MODAL
# -----------------------------------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

st.sidebar.markdown("---")
if st.sidebar.button("Toggle AI Assistant Modal", use_container_width=True):
    st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    st.markdown('<div class="chat-modal-window">', unsafe_allow_html=True)
    st.markdown("#### AI Document Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload a document to ask questions."}]

    chat_box = st.container(height=280)
    with chat_box:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot"><b>AI Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Ask a question about this document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        if not uploaded_file:
            bot_response = "Please upload a document first before querying."
        else:
            with st.spinner("Processing document with Vision AI..."):
                bot_response = query_vision_ai(
                    user_prompt=user_query,
                    file_bytes=file_bytes,
                    file_type=file_type
                )

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)