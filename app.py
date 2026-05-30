import streamlit as st
import os
import time
import sys
import glob
import requests
from ingest import run_ingestion, WORKSPACE_DIR, DB_DIR, OLLAMA_HOST
from rag_pipeline import RAGPipeline

# Avoid encoding errors on Windows when stdout/stderr are redirected
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Set page config for a premium, wide dashboard layout
st.set_page_config(
    page_title="FairPlay RAG - Sports Anti-Doping Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark-Aesthetic styling injected via CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
    }
    
    /* Custom Title Style */
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .title-gradient {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 300;
    }
    
    /* Sidebar premium touch */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #818cf8;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.3rem;
    }
    
    /* Source list customization */
    .source-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #c084fc;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .source-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Micro-animation for thinking indicator */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .5; }
    }
    .thinking {
        animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        color: #a855f7;
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 10px;
    }
    
    /* Audit card styles */
    .audit-card {
        background-color: #1e293b;
        border-left: 5px solid #818cf8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title Banner
st.markdown("""
<div class="title-container">
    <div class="title-gradient">🛡️ FairPlay Anti-Doping RAG</div>
    <div class="subtitle-text">Semantic policy compliance assistant powered by Google Gemini & Local Ollama</div>
</div>
""", unsafe_allow_html=True)

# Helper function to get pipeline instance securely
@st.cache_resource(show_spinner=False)
def get_rag_pipeline(api_key: str, embedding_model: str, llm_model: str, provider: str) -> RAGPipeline:
    return RAGPipeline(
        api_key=api_key, 
        embedding_model=embedding_model, 
        llm_model=llm_model, 
        provider=provider
    )

# Scan local folder for documents
local_pdf_files = [os.path.basename(f) for f in glob.glob(os.path.join(WORKSPACE_DIR, "*.pdf"))]
local_docx_files = [os.path.basename(f) for f in glob.glob(os.path.join(WORKSPACE_DIR, "*.docx"))]
all_local_files = sorted(local_pdf_files + local_docx_files)

# Dynamic local Ollama model detection
def get_local_ollama_models() -> tuple[list[str], bool]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", [])]
            return sorted(names), True
    except Exception:
        pass
    return [], False

ollama_model_list, ollama_connected = get_local_ollama_models()

# ----------------- SIDEBAR SETUP -----------------
st.sidebar.markdown('<div class="sidebar-header" style="margin-top: 0;">🌐 PROVIDER SELECTOR</div>', unsafe_allow_html=True)

selected_provider = st.sidebar.radio(
    "Select AI Provider:",
    options=["Ollama (Local - 100% Free)", "Google Gemini"],
    help="Ollama executes entirely on your CPU/GPU without cloud API keys."
)

active_provider = "Ollama" if "Ollama" in selected_provider else "Google Gemini"

# Provider-Specific Configurations
active_api_key = None
selected_llm_model = ""
selected_emb_model = ""

if active_provider == "Google Gemini":
    st.sidebar.markdown('<div class="sidebar-header">🔑 GEMINI CREDENTIALS</div>', unsafe_allow_html=True)
    default_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key:",
        value=default_api_key,
        type="password",
        help="Required for cloud embeddings and response generation."
    )
    active_api_key = api_key_input or default_api_key
    
    st.sidebar.markdown('<div class="sidebar-header">🤖 MODEL CONFIGURATION</div>', unsafe_allow_html=True)
    selected_llm_model = st.sidebar.selectbox(
        "Large Language Model (LLM):",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0
    )
    selected_emb_model = st.sidebar.selectbox(
        "Embedding Model:",
        options=["gemini-embedding-001", "gemini-embedding-2"],
        index=0
    )
else:
    # Ollama configuration
    st.sidebar.markdown('<div class="sidebar-header">🤖 LOCAL OLLAMA CONFIG</div>', unsafe_allow_html=True)
    if not ollama_connected:
        st.sidebar.error("⚠️ Local Ollama server not detected. Make sure Ollama is running at http://localhost:11434")
        selected_llm_model = st.sidebar.text_input("Local LLM Model:", value="llama3")
        selected_emb_model = st.sidebar.text_input("Local Embedding Model:", value="nomic-embed-text")
    else:
        st.sidebar.success("💡 Connected to local Ollama server.")
        selected_llm_model = st.sidebar.selectbox(
            "Select Local LLM Model:",
            options=ollama_model_list,
            index=0 if ollama_model_list else 0,
            help="Select the model downloaded in Ollama to formulate answers."
        )
        emb_candidates = [m for m in ollama_model_list if "embed" in m or "minilm" in m]
        emb_options = emb_candidates if emb_candidates else ollama_model_list
        selected_emb_model = st.sidebar.selectbox(
            "Select Local Embedding Model:",
            options=emb_options if emb_options else ["nomic-embed-text"],
            index=0,
            help="Select the embedding model downloaded in Ollama."
        )

# Database status check
db_status = {"initialized": False, "chunk_count": 0, "documents": []}
pipeline_ready = False

if active_provider == "Ollama" or (active_provider == "Google Gemini" and active_api_key):
    try:
        pipeline = get_rag_pipeline(active_api_key, selected_emb_model, selected_llm_model, active_provider)
        db_status = pipeline.check_db_status()
        pipeline_ready = True
        
        if active_provider == "Google Gemini":
            if st.sidebar.button("🔍 List Available Gemini Models", use_container_width=True):
                st.sidebar.markdown("**Available Cloud Models:**")
                client = pipeline.client
                for m in client.models.list():
                    st.sidebar.write(f"- `{getattr(m, 'name', str(m))}`")
    except Exception as e:
        st.sidebar.error(f"Initialization error: {e}")

# Ingestion Control UI
st.sidebar.markdown('<div class="sidebar-header">📁 KNOWLEDGE BASE</div>', unsafe_allow_html=True)

if not db_status["initialized"]:
    st.sidebar.warning(f"⚠️ Collection not initialized for embedding model '{selected_emb_model}'. Please ingest documents.")
else:
    st.sidebar.success(f"✅ Active: {db_status['chunk_count']} chunks indexed under '{selected_emb_model}'.")
    st.sidebar.markdown("**Currently Indexed in DB:**")
    for doc in db_status["documents"]:
        st.sidebar.markdown(f"- 📄 `{doc}`")

# Document selection to ingest
st.sidebar.markdown('<div class="sidebar-header">📝 FILE INGESTION SELECTOR</div>', unsafe_allow_html=True)
selected_ingest_files = st.sidebar.multiselect(
    "Select files to ingest:",
    options=all_local_files,
    default=all_local_files,
    help="Select which local PDF and DOCX files should be indexed into the ChromaDB vector store."
)

# Ingest Button Action
st.sidebar.markdown("---")
rebuild_db = st.sidebar.checkbox("Full Database Rebuild", value=False, help="Delete existing index and re-ingest all files.")
if st.sidebar.button("⚡ Ingest / Sync Documents", use_container_width=True):
    if active_provider == "Google Gemini" and not active_api_key:
        st.sidebar.error("❌ Gemini API Key is required to generate embeddings for document ingestion.")
    elif active_provider == "Ollama" and not ollama_connected:
        st.sidebar.error("❌ Cannot connect to local Ollama. Start Ollama and download the embedding model first.")
    elif not selected_ingest_files:
        st.sidebar.error("❌ Please select at least one document to ingest.")
    else:
        with st.sidebar.status("Processing and indexing documents...", expanded=True) as status:
            try:
                status.write(f"Scanning workspace for selected files using {active_provider}...")
                raw_count, chunk_count = run_ingestion(
                    api_key=active_api_key, 
                    force_rebuild=rebuild_db,
                    embedding_model=selected_emb_model,
                    selected_files=selected_ingest_files,
                    provider=active_provider
                )
                
                st.cache_resource.clear()
                
                status.update(label="✅ Ingestion Successful!", state="complete", expanded=False)
                st.toast(f"Successfully indexed {raw_count} documents into {chunk_count} semantic blocks.", icon="🚀")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                status.update(label="❌ Ingestion Failed", state="error")
                st.sidebar.error(f"Error during ingestion: {e}")
                st.sidebar.code(error_trace, language="python")

# ----------------- MAIN INTERFACE -----------------

# Chat and Audit History Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_trigger" not in st.session_state:
    st.session_state.query_trigger = None
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# Setup visual dual-panel tabs for cleaner UX
tab_chat, tab_history = st.tabs(["💬 Compliance Chat Panel", "📜 Detailed Audit & History Log"])

with tab_chat:
    # Quick Clickable suggested cards (shown only if chat is empty)
    if len(st.session_state.chat_history) == 0:
        st.markdown(f"### 🔍 Anti-Doping Assistant — Running via **{active_provider}**")
        st.markdown("Select a sample compliance query below or type your own:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 What is WADA's definition of a Prohibited Substance?\n\n(Click to query)", use_container_width=True):
                st.session_state.query_trigger = "What is the definition of a Prohibited Substance under WADA rules?"
            if st.button("🏥 What are the rules regarding Therapeutic Use Exemptions (TUEs)?\n\n(Click to query)", use_container_width=True):
                st.session_state.query_trigger = "What are the rules and requirements for obtaining a Therapeutic Use Exemption (TUE)?"
        with col2:
            if st.button("📢 What constitutes an Anti-Doping Rule Violation (ADRV)?\n\n(Click to query)", use_container_width=True):
                st.session_state.query_trigger = "What constitutes an Anti-Doping Rule Violation (ADRV)?"
            if st.button("🤫 What are the SCA Whistleblowing regulations?\n\n(Click to query)", use_container_width=True):
                st.session_state.query_trigger = "What are the whistleblower protection and reporting guidelines under Singapore Cricket Association rules?"

    st.markdown("---")

    # Render previous chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                st.markdown('<div class="source-header">📚 Retrieved Policy References:</div>', unsafe_allow_html=True)
                for idx, src in enumerate(message["sources"]):
                    with st.expander(f"📄 {src['source']} — Page/Section {src['page']} (Relevance: {src['relevance']*100:.1f}%)"):
                        st.markdown(f"*{src['text']}*")

    # Chat input bar
    user_input = st.chat_input("Ask a question about the sports anti-doping policies...")

    active_query = None
    if user_input:
        active_query = user_input
    elif st.session_state.query_trigger:
        active_query = st.session_state.query_trigger
        st.session_state.query_trigger = None

    # Processing Query
    if active_query:
        with st.chat_message("user"):
            st.markdown(active_query)
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        
        with st.chat_message("assistant"):
            if active_provider == "Google Gemini" and not active_api_key:
                st.error("🔒 Please enter a Google Gemini API Key in the sidebar configuration to execute queries.")
            elif active_provider == "Ollama" and not ollama_connected:
                st.error("❌ Ollama server is not running locally. Please launch Ollama to run the queries for free.")
            elif not db_status["initialized"]:
                st.warning(f"⚠️ ChromaDB has no vectors for embedding model '{selected_emb_model}'. Please select your files and click 'Ingest / Sync Documents' in the sidebar first.")
            else:
                thinking_placeholder = st.markdown(f'<div class="thinking">🔍 Searching policies and analyzing via {active_provider} ({selected_llm_model})...</div>', unsafe_allow_html=True)
                
                try:
                    pipeline = get_rag_pipeline(active_api_key, selected_emb_model, selected_llm_model, active_provider)
                    
                    # Run RAG Query
                    response_stream, sources = pipeline.query(active_query)
                    
                    thinking_placeholder.empty()
                    
                    # Render detailed step-by-step processing logs
                    if hasattr(pipeline, "process_logs") and pipeline.process_logs:
                        with st.expander("🛠... RAG Pipeline Execution Details", expanded=True):
                            for log in pipeline.process_logs:
                                st.markdown(f"**Step: {log['step']}**")
                                st.write(log['message'])
                                st.markdown("---")
                                
                    # Show top retrieved documents and scores before the response
                    if sources:
                        st.markdown("### 🎯 Top Retrieved Clauses & Similarity Scores:")
                        for idx, src in enumerate(sources[:5]):  # Top 3-5
                            st.markdown(
                                f"**Rank {idx+1}**: Relevance: `{src['relevance']*100:.1f}%` | "
                                f"Source: `{src['source']}` | Page/Section: `{src['page']}`"
                            )
                            st.caption(f"Snippet: *\"{src['text'][:180]}...\"*")
                        st.markdown("---")

                    # Stream the results
                    st.markdown("### 📝 Formulated Answer:")
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Render sources
                    if sources:
                        st.markdown('<div class="source-header">📚 Retrieved Policy References:</div>', unsafe_allow_html=True)
                        for idx, src in enumerate(sources):
                            with st.expander(f"📄 {src['source']} — Page/Section {src['page']} (Relevance: {src['relevance']*100:.1f}%)"):
                                st.markdown(f"*{src['text']}*")
                                
                    # Save assistant output to standard chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Save rich detailed structured history log
                    st.session_state.history_log.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query": active_query,
                        "answer": full_response,
                        "provider": active_provider,
                        "llm_model": selected_llm_model,
                        "emb_model": selected_emb_model,
                        "rewritten_queries": list(pipeline.rewritten_queries) if hasattr(pipeline, "rewritten_queries") else [],
                        "process_logs": list(pipeline.process_logs) if hasattr(pipeline, "process_logs") else [],
                        "sources": sources
                    })
                    
                except Exception as e:
                    thinking_placeholder.empty()
                    import traceback
                    st.error(f"An error occurred: {e}")
                    st.code(traceback.format_exc(), language="python")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}"
                    })

with tab_history:
    st.markdown("### 📜 System Execution History & RAG Audit Log")
    if not st.session_state.history_log:
        st.info("No queries have been executed yet in this session. Ask a question to generate audit traces.")
    else:
        # Display the logs in reverse order (newest first)
        for idx, entry in enumerate(reversed(st.session_state.history_log)):
            with st.expander(f"⏱️ [{entry['timestamp']}] Query: \"{entry['query'][:60]}...\""):
                st.markdown(f'<div class="audit-card"><strong>Prompt Query:</strong><br/>{entry["query"]}</div>', unsafe_allow_html=True)
                
                # Providers meta-badges
                st.markdown(
                    f"**Provider**: `{entry['provider']}` | "
                    f"**LLM Model**: `{entry['llm_model']}` | "
                    f"**Embedding Model**: `{entry['emb_model']}`"
                )
                
                # Query expansion / reformulations
                if entry['rewritten_queries']:
                    st.markdown("#### 🔄 Query Expansion & Synonyms:")
                    for r_idx, q in enumerate(entry['rewritten_queries']):
                        st.markdown(f"- **Attempt {r_idx+1}**: *\"{q}\"*")
                else:
                    st.markdown("#### 🔄 Query Expansion: `None (Direct Match)`")
                
                # Diagnostic RAG process steps
                st.markdown("#### 🛠️ RAG Pipeline Step Logs:")
                for step in entry['process_logs']:
                    st.markdown(f"- **{step['step']}**: {step['message']}")
                
                # Ranked retrieved results
                st.markdown("#### 🎯 Reranked Retrieved Clauses & Similarity Scores:")
                for s_idx, src in enumerate(entry['sources'][:5]):
                    st.markdown(
                        f"**Rank {s_idx+1} (Relevance Score: {src['relevance']*100:.1f}%)**: "
                        f"Page `{src['page']}` | File: `{src['source']}`"
                    )
                    st.caption(f"Context Text: *\"{src['text']}\"*")
                    st.markdown("---")
                    
                # Formulated LLM Answer
                st.markdown("#### 📝 Formulated Answer:")
                st.write(entry['answer'])
