import os
import glob
import re
import argparse
import sys
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
import chromadb
from google import genai
from google.genai import errors

# Load environment variables
load_dotenv()

# Avoid encoding errors on Windows when stdout/stderr are redirected
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Define default paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(WORKSPACE_DIR, "chroma_db")
OLLAMA_HOST = "http://localhost:11434"

def get_collection_name(embedding_model: str) -> str:
    """Generates a ChromaDB compliant collection name based on the embedding model."""
    # Chroma requires name between 3-63 chars, containing alphanumeric, underscores or hyphens
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', embedding_model)
    clean_name = clean_name.strip('_')
    # Truncate to ensure it fits within 63 chars (leaving room for prefix)
    truncated = clean_name[:45]
    return f"sports_rag_{truncated}"

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extracts text page-by-page from a PDF file."""
    pages_data = []
    try:
        reader = PdfReader(pdf_path)
        filename = os.path.basename(pdf_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "text": text.strip(),
                    "source": filename,
                    "page": page_idx + 1  # 1-indexed
                })
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return pages_data

def extract_text_from_docx(docx_path: str) -> list[dict]:
    """Extracts text paragraph-by-paragraph from a DOCX file."""
    pages_data = []
    try:
        doc = Document(docx_path)
        filename = os.path.basename(docx_path)
        
        # Group paragraphs to simulate pages/sections for metadata attribution
        current_text = []
        p_count = 0
        section_idx = 1
        
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                current_text.append(p.text.strip())
                p_count += 1
                
                # Create a "page" block every 5 paragraphs to keep chunk context manageable
                if p_count >= 5:
                    pages_data.append({
                        "text": "\n".join(current_text),
                        "source": filename,
                        "page": section_idx
                    })
                    current_text = []
                    p_count = 0
                    section_idx += 1
                    
        # Append remaining paragraphs
        if current_text:
            pages_data.append({
                "text": "\n".join(current_text),
                "source": filename,
                "page": section_idx
            })
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return pages_data

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Splits text into chunks at sentence boundaries where possible.
    Implements a sentence-based sliding window approach to preserve readability.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
                
            words = sentence.split(' ')
            sub_chunk = []
            sub_len = 0
            for word in words:
                if sub_len + len(word) + 1 > chunk_size:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        overlap_words = []
                        overlap_len = 0
                        for w in reversed(sub_chunk):
                            if overlap_len + len(w) + 1 <= chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_len += len(w) + 1
                            else:
                                break
                        sub_chunk = overlap_words
                        sub_len = overlap_len
                    sub_chunk.append(word)
                    sub_len += len(word) + 1
                else:
                    sub_chunk.append(word)
                    sub_len += len(word) + 1
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
                
        elif current_len + sentence_len + 1 > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) + 1 <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len
                
            current_chunk.append(sentence)
            current_len += sentence_len + 1
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_documents(directory: str, selected_files: list[str] = None) -> list[dict]:
    """Finds specified PDFs and Word docs and extracts their contents with page mappings."""
    documents_data = []
    
    # PDF parsing
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for pdf in pdf_files:
        filename = os.path.basename(pdf)
        if selected_files is None or filename in selected_files:
            print(f"Reading PDF: {filename}")
            documents_data.extend(extract_text_from_pdf(pdf))
        
    # Word parsing
    docx_files = glob.glob(os.path.join(directory, "*.docx"))
    for docx in docx_files:
        filename = os.path.basename(docx)
        if selected_files is None or filename in selected_files:
            print(f"Reading DOCX: {filename}")
            documents_data.extend(extract_text_from_docx(docx))
        
    return documents_data

def get_ollama_embeddings(model_name: str, texts: list[str]) -> list[list[float]]:
    """Fetches text embeddings from local Ollama instance."""
    # Attempt batch embedding endpoint /api/embed first
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": model_name, "input": texts},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["embeddings"]
    except Exception:
        pass
        
    # Fallback to /api/embeddings for older Ollama versions
    embeddings = []
    for text in texts:
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": model_name, "prompt": text},
                timeout=10
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama /api/embeddings returned status {response.status_code}: {response.text}")
            embeddings.append(response.json()["embedding"])
        except Exception as e:
            raise RuntimeError(f"Failed to connect or fetch embeddings from local Ollama: {e}")
    return embeddings

def run_ingestion(
    api_key: str = None, 
    force_rebuild: bool = False, 
    embedding_model: str = "gemini-embedding-001", 
    selected_files: list[str] = None,
    provider: str = "Google Gemini"
) -> tuple[int, int]:
    """Runs the document parsing, embedding, and indexing pipeline with parameterized files, models, and providers."""
    
    # Initialize local persistent Chroma DB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = get_collection_name(embedding_model)
    
    # Delete collection if force rebuild requested
    if force_rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
            
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("Scanning workspace for selected documents...")
    raw_docs = process_documents(WORKSPACE_DIR, selected_files)
    
    if not raw_docs:
        print("No documents found to process.")
        return 0, 0
        
    print(f"Extracted {len(raw_docs)} total document pages/blocks.")
    
    # Process into chunks
    processed_chunks = []
    for doc in raw_docs:
        chunks = split_text_into_chunks(doc["text"])
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc['source']}_p{doc['page']}_c{chunk_idx}"
            processed_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": chunk_idx
                }
            })
            
    print(f"Generated {len(processed_chunks)} chunks for indexing.")
    
    if not processed_chunks:
        return len(raw_docs), 0
        
    # Generate embeddings and add to ChromaDB in batches
    batch_size = 50
    total_indexed = 0
    
    print(f"Generating embeddings using {provider} - {embedding_model}...")
    
    # Setup Gemini client if needed
    client = None
    if provider == "Google Gemini":
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            gemini_key = gemini_key.strip()
        if not gemini_key:
            raise ValueError("Google Gemini API Key is missing. Please set it to proceed with Gemini embeddings.")
        client = genai.Client(api_key=gemini_key)

    for i in range(0, len(processed_chunks), batch_size):
        batch = processed_chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        
        try:
            if provider == "Google Gemini":
                embeddings = []
                for t in texts:
                    res = client.models.embed_content(
                        model=embedding_model,
                        contents=t
                    )
                    embeddings.append(res.embeddings[0].values)
            else:
                # Local Ollama
                embeddings = get_ollama_embeddings(embedding_model, texts)
            
            # Add to Chroma DB collection
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            total_indexed += len(batch)
            print(f"Indexed {total_indexed}/{len(processed_chunks)} chunks...")
            
        except errors.APIError as e:
            print(f"Gemini API Error during batch embedding: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error during embedding/indexing: {e}")
            raise e
            
    print(f"Successfully finished ingestion. Total indexed: {total_indexed} chunks.")
    return len(raw_docs), total_indexed
