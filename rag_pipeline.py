import os
import re
import requests
import json
import chromadb
from google import genai
from google.genai import types
from google.genai import errors
from ingest import get_collection_name, OLLAMA_HOST

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(WORKSPACE_DIR, "chroma_db")

class OllamaStreamWrapper:
    """Wrapper class to match the .text attribute signature of Gemini's stream chunk."""
    def __init__(self, text: str):
        self.text = text

class RAGPipeline:
    def __init__(
        self, 
        api_key: str = None, 
        embedding_model: str = "gemini-embedding-001", 
        llm_model: str = "gemini-2.5-flash",
        provider: str = "Google Gemini"
    ):
        """Initializes RAG parameters for Google Gemini or local Ollama."""
        self.provider = provider
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.api_key = api_key
        
        # Initialize Gemini client if using Google Gemini
        if self.provider == "Google Gemini":
            if not api_key:
                raise ValueError("Google Gemini API Key is required when provider is Gemini.")
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            
        # Initialize persistent Chroma Client
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        
        # Open collection mapping to specific embedding model
        collection_name = get_collection_name(self.embedding_model)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            self.collection = None

    def check_db_status(self) -> dict:
        """Checks if the specific database for the active embedding model is initialized."""
        collection_name = get_collection_name(self.embedding_model)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            return {"initialized": False, "chunk_count": 0, "documents": []}
            
        if self.collection is None:
            return {"initialized": False, "chunk_count": 0, "documents": []}
            
        count = self.collection.count()
        if count == 0:
            return {"initialized": False, "chunk_count": 0, "documents": []}
            
        # Extract unique documents indexed
        metadata = self.collection.get(include=["metadatas"])
        documents = set()
        if metadata and "metadatas" in metadata:
            for meta in metadata["metadatas"]:
                if meta and "source" in meta:
                    documents.add(meta["source"])
                    
        return {
            "initialized": True,
            "chunk_count": count,
            "documents": sorted(list(documents))
        }

    def _get_ollama_query_embedding(self, query: str) -> list[float]:
        """Fetches query embedding from local Ollama instance."""
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": self.embedding_model, "prompt": query},
                timeout=10
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama embedding error: {response.text}")
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding from local Ollama model {self.embedding_model}: {e}")

    def _stream_ollama_generation(self, system_instruction: str, user_content: str):
        """Generates streaming responses from local Ollama and yields unified text chunks."""
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": user_content,
                    "system": system_instruction,
                    "options": {"temperature": 0.2},
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            if response.status_code != 200:
                error_msg = f"Ollama generation failed: Status {response.status_code} - {response.text}"
                yield OllamaStreamWrapper(error_msg)
                return
                
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    text_chunk = chunk.get("response", "")
                    if text_chunk:
                        yield OllamaStreamWrapper(text_chunk)
                        
        except Exception as e:
            yield OllamaStreamWrapper(f"\n[Generation Error: {e}]")

    def query(self, user_query: str, k: int = 5) -> tuple[any, list[dict]]:
        """
        Performs vector search on ChromaDB, compiles prompt, and streams
        the response from Gemini or Ollama.
        """
        # Ensure database is loaded
        collection_name = get_collection_name(self.embedding_model)
        if self.collection is None:
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                raise RuntimeError(f"ChromaDB collection for '{self.embedding_model}' is not initialized. Please ingest documents first.")

        # 1. Embed query
        if self.provider == "Google Gemini":
            try:
                embed_response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=user_query
                )
                query_embedding = embed_response.embeddings[0].values
            except errors.APIError as e:
                raise RuntimeError(f"Error generating query embedding with model {self.embedding_model}: {e}")
        else:
            # Local Ollama
            query_embedding = self._get_ollama_query_embedding(user_query)

        # 2. Query ChromaDB for top K matches
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # 3. Parse and rank sources
        sources = []
        context_parts = []
        
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for idx, (doc_text, meta, dist) in enumerate(zip(docs, metadatas, distances)):
                relevance_score = max(0.0, min(1.0, 1.0 - dist))
                
                source_info = {
                    "text": doc_text,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "Unknown"),
                    "relevance": relevance_score
                }
                sources.append(source_info)
                
                context_parts.append(
                    f"Source: {source_info['source']}\n"
                    f"Page/Section: {source_info['page']}\n"
                    f"Relevance: {relevance_score:.2f}\n"
                    f"Content:\n{doc_text}\n"
                    f"---"
                )
                
        context_str = "\n".join(context_parts)

        # 4. Construct Prompt
        system_instruction = (
            "You are an expert Anti-Doping Policy AI Assistant. Your job is to answer questions "
            "about anti-doping policies, therapeutic exemptions, WADA codes, and whistleblowing rules "
            "strictly using the provided source context blocks. \n\n"
            "Follow these strict guidelines:\n"
            "1. Provide a detailed, clear, and comprehensive explanation in natural, human-readable language. "
            "Explain the rules and policies step-by-step to help the user understand them easily.\n"
            "2. Rely ONLY on the facts present in the source context. Do not invent, speculate, or extrapolate.\n"
            "3. Cite your sources inline or at the end of the statement by referencing the document name and page/section number "
            "(e.g., [2021_wada_code.pdf, Page 12]).\n"
            "4. If the context does not contain the information needed to answer, state: "
            "'I am sorry, but the provided anti-doping policy documents do not contain the answer to that question.' "
            "Do not answer using external knowledge."
        )

        user_content = (
            f"System Instructions:\n"
            f"{system_instruction}\n\n"
            f"Context from Sports Anti-Doping Policies:\n"
            f"=========================================\n"
            f"{context_str}\n"
            f"=========================================\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )

        # 5. Call LLM (Gemini or Ollama) with stream enabled
        if self.provider == "Google Gemini":
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=self.llm_model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        max_output_tokens=1500
                    )
                )
                return response_stream, sources
            except errors.APIError as e:
                raise RuntimeError(f"Gemini LLM Generation Error with model {self.llm_model}: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected generation error: {e}")
        else:
            # Local Ollama
            response_stream = self._stream_ollama_generation(system_instruction, user_content)
            return response_stream, sources
