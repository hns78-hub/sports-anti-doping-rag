import os
import re
import requests
import json
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from ingest import get_collection_name, OLLAMA_HOST

# Load environment variables from .env file
load_dotenv()

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
        
        # Load API keys from parameters or environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip()
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")
        if self.cohere_api_key:
            self.cohere_api_key = self.cohere_api_key.strip()
        
        self.rewritten_queries = []
        self.process_logs = []
        
        # Initialize Gemini client if using Google Gemini
        if self.provider == "Google Gemini":
            if not self.api_key:
                raise ValueError("Google Gemini API Key is required when provider is Gemini. Please set it in your .env file or UI.")
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

    def _rewrite_query(self, query: str) -> str:
        """Uses the active LLM provider to rewrite the query for better semantic retrieval."""
        prompt = (
            f"You are a search query optimizer for a Sports Anti-Doping RAG system.\n"
            f"The user query \"{query}\" did not find good matches. "
            f"Please rewrite and expand this query to improve search results. "
            f"Use descriptive keywords, expand acronyms (like WADA, TUE, ADRV), "
            f"and focus on regulatory policy terms. "
            f"Provide ONLY the raw optimized query string. Do not include quotes, conversational preamble, or explanations."
        )
        
        try:
            if self.provider == "Google Gemini":
                response = self.client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=150
                    )
                )
                rewritten = response.text.strip()
            else:
                # Ollama
                response = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "options": {"temperature": 0.7},
                        "stream": False
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    rewritten = response.json().get("response", "").strip()
                else:
                    rewritten = query # Fallback
            
            # Clean up output
            rewritten = rewritten.strip('"\'')
            if rewritten.lower().startswith("rewritten query:"):
                rewritten = rewritten[len("rewritten query:"):].strip()
            return rewritten if rewritten else query
            
        except Exception as e:
            print(f"Error during query rewriting: {e}")
            return query

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

    def _cohere_rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        """Reranks document chunks using Cohere Rerank v3.0 REST API."""
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.cohere_api_key}"
            }
            
            # Prepare text inputs for Cohere
            doc_texts = [d["text"] for d in documents]
            
            payload = {
                "model": "rerank-english-v3.0",
                "query": query,
                "documents": doc_texts,
                "top_n": top_n
            }
            
            response = requests.post(
                "https://api.cohere.com/v1/rerank",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Cohere Rerank API returned status {response.status_code}: {response.text}")
                return documents[:top_n]
                
            results = response.json().get("results", [])
            reranked_docs = []
            for r in results:
                orig_idx = r["index"]
                new_score = r["relevance_score"]
                
                # Copy and update relevance score
                doc = documents[orig_idx].copy()
                doc["relevance"] = new_score
                reranked_docs.append(doc)
                
            return reranked_docs
            
        except Exception as e:
            print(f"Error during Cohere Rerank: {e}")
            return documents[:top_n]

    def query(self, user_query: str, k: int = 5) -> tuple[any, list[dict]]:
        """
        Performs vector search on ChromaDB, reranks via Cohere (if available), 
        compiles prompt, and streams the response from Gemini or Ollama.
        """
        # Ensure database is loaded
        collection_name = get_collection_name(self.embedding_model)
        if self.collection is None:
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                raise RuntimeError(f"ChromaDB collection for '{self.embedding_model}' is not initialized. Please ingest documents first.")

        current_query = user_query
        self.rewritten_queries = []
        self.process_logs = []  # Detailed steps carried out by the RAG
        max_retries = 3
        
        sources = []
        context_str = ""

        # Step 1: Log initial search
        self.process_logs.append({
            "step": "Pipeline Initialized",
            "message": f"Starting RAG query pipeline using **{self.provider}**.\n"
                       f"- LLM Model: `{self.llm_model}`\n"
                       f"- Embedding Model: `{self.embedding_model}`\n"
                       f"- Cohere Reranker: `{'Enabled (rerank-english-v3.0)' if self.cohere_api_key else 'Disabled (Fallback to Cosine)'}`"
        })

        for attempt in range(max_retries + 1):
            attempt_prefix = f"Attempt {attempt + 1}" if attempt > 0 else "Initial Attempt"
            
            # Step 2: Log Embedding Generation
            self.process_logs.append({
                "step": f"Generating Query Embedding ({attempt_prefix})",
                "message": f"Converting query to semantic vector using `{self.embedding_model}`.\n"
                           f"Query: *\"{current_query}\"*"
            })

            # 1. Embed query
            if self.provider == "Google Gemini":
                try:
                    embed_response = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=current_query
                    )
                    query_embedding = embed_response.embeddings[0].values
                except errors.APIError as e:
                    raise RuntimeError(f"Error generating query embedding with model {self.embedding_model}: {e}")
            else:
                # Local Ollama
                query_embedding = self._get_ollama_query_embedding(current_query)

            # Step 3: Log Vector Database Query
            # If Cohere reranking is active, fetch a larger pool of candidates to rerank
            fetch_k = max(k * 3, 12) if self.cohere_api_key else k
            self.process_logs.append({
                "step": f"Querying ChromaDB ({attempt_prefix})",
                "message": f"Performing Cosine Similarity matching in ChromaDB.\n"
                           f"Retrieving top `{fetch_k}` nearest candidate clauses."
            })

            # 2. Query ChromaDB for matches
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )

            # 3. Parse and rank sources
            candidates = []
            
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
                    candidates.append(source_info)
            
            # Step 4: Reranking with Cohere Rerank v3
            top_relevance = 0.0
            if self.cohere_api_key and candidates:
                self.process_logs.append({
                    "step": f"Cohere Rerank v3.0 ({attempt_prefix})",
                    "message": f"Passing `{len(candidates)}` vector matches to Cohere API.\n"
                               f"Executing deep semantic reranking using `rerank-english-v3.0`."
                })
                sources = self._cohere_rerank(current_query, candidates, top_n=k)
                if sources:
                    top_relevance = sources[0]["relevance"]
            else:
                # Fallback directly to cosine order
                sources = candidates[:k]
                if sources:
                    top_relevance = sources[0]["relevance"]

            # Format context string
            context_parts = []
            for src in sources:
                context_parts.append(
                    f"Source: {src['source']}\n"
                    f"Page/Section: {src['page']}\n"
                    f"Relevance: {src['relevance']:.2f}\n"
                    f"Content:\n{src['text']}\n"
                    f"---"
                )
            context_str = "\n".join(context_parts)
            
            # Step 5: Log Similarity Scores & Threshold Check
            self.process_logs.append({
                "step": f"Similarity Evaluation ({attempt_prefix})",
                "message": f"Evaluating retrieved results against quality threshold (>= 50%).\n"
                           f"- **Top Reranked Score: {top_relevance * 100:.1f}%**"
            })

            # If top match matches or exceeds similarity threshold, or we exhausted retries
            if top_relevance >= 0.5 or attempt == max_retries:
                if top_relevance >= 0.5:
                    self.process_logs.append({
                        "step": "Retrieval Finalized",
                        "message": f"Rerank score **{top_relevance * 100:.1f}%** satisfies quality threshold. Proceeding to formulate final answer."
                    })
                else:
                    self.process_logs.append({
                        "step": "Retrieval Finalized (Fallback)",
                        "message": f"Maximum rewrites ({max_retries}) reached. Proceeding to answer with best available context."
                    })
                break
                
            # Otherwise, rewrite and loop again
            self.process_logs.append({
                "step": f"Triggering Query Reformulation",
                "message": f"Top similarity score ({top_relevance * 100:.1f}%) is below 50% threshold.\n"
                           f"Triggering LLM query expansion to improve regulatory coverage..."
            })
            
            rewritten = self._rewrite_query(current_query)
            self.rewritten_queries.append(rewritten)
            current_query = rewritten

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
