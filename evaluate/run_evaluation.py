import os
import sys
import json
import time
import pandas as pd

# Add the workspace root folder to the python path to resolve local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

def run_ragas_evaluation():
    print("Initializing evaluation...")
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    golden_set_file = os.path.join(eval_dir, "golden_set.json")
    
    if not os.path.exists(golden_set_file):
        print(f"Golden set file not found at: {golden_set_file}")
        return
        
    with open(golden_set_file, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
        
    print(f"Loaded {len(golden_set)} evaluation golden questions.")
    
    # Initialize the RAG Pipeline
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY is missing from environment.")
        return
        
    pipeline = RAGPipeline(
        api_key=gemini_key.strip(),
        embedding_model="gemini-embedding-001",
        llm_model="gemini-2.5-flash",
        provider="Google Gemini"
    )
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("Generating answers and retrieving context for the golden set...")
    for idx, item in enumerate(golden_set):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{idx+1}/10] Querying: '{q[:50]}...'")
        
        # Exponential backoff retry loop for querying the pipeline to respect RPM limits
        full_answer = None
        q_contexts = None
        
        for attempt in range(5):
            try:
                # Query RAG
                response_stream, sources = pipeline.query(q)
                
                # Consume stream to get full answer
                full_answer = ""
                for chunk in response_stream:
                    if chunk and hasattr(chunk, "text") and chunk.text:
                        full_answer += chunk.text
                        
                # Context list of strings
                q_contexts = [src["text"] for src in sources] if sources else ["No context retrieved"]
                break
                
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                    delay = 15.0 * (2 ** attempt)
                    print(f"RAG query rate limit hit. Sleeping for {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Non-rate-limit error: {e}. Retrying anyway...")
                    time.sleep(5.0)
                    
        if full_answer is None or q_contexts is None:
            full_answer = "Error: Failed to fetch answer due to rate limits."
            q_contexts = ["Error"]
            
        questions.append(q)
        answers.append(full_answer if full_answer.strip() else "I am sorry, but the provided anti-doping policy documents do not contain the answer to that question.")
        contexts.append(q_contexts)
        ground_truths.append(gt)
        
        # Sleep for a baseline of 12 seconds between questions to remain below Gemini's 5 RPM free-tier limit and Cohere's 10 RPM trial limit
        if idx < len(golden_set) - 1:
            print("Sleeping 12s to respect API rate limits...")
            time.sleep(12)
            
    print("Assembling dataset for Ragas...")
    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(eval_dict)
    
    print("Configuring Google Gemini Chat and Embeddings as Ragas Judge...")
    # Configure evaluator llm and embeddings using langchain-google-genai
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key.strip(),
        temperature=0.0,
        max_retries=10
    )
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_key.strip(),
        max_retries=10
    )
    
    # Run evaluation
    print("Running Ragas evaluation metrics (faithfulness, answer_relevancy, context_recall, context_precision)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            max_workers=1
        )
        
        print("\nEvaluation Completed Successfully!")
        print(f"Global Scores:\n{result}")
        
        # Save scores to result JSON
        scores_df = result.to_pandas()
        
        results_data = {
            "global_scores": {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevance": float(result.get("answer_relevancy", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0))
            },
            "per_question_results": []
        }
        
        for index, row in scores_df.iterrows():
            results_data["per_question_results"].append({
                "id": index + 1,
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
                "faithfulness": float(row.get("faithfulness", 0.0)) if not pd.isna(row.get("faithfulness")) else 0.0,
                "answer_relevance": float(row.get("answer_relevancy", 0.0)) if not pd.isna(row.get("answer_relevancy")) else 0.0,
                "context_recall": float(row.get("context_recall", 0.0)) if not pd.isna(row.get("context_recall")) else 0.0,
                "context_precision": float(row.get("context_precision", 0.0)) if not pd.isna(row.get("context_precision")) else 0.0
            })
            
        results_file = os.path.join(eval_dir, "eval_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved evaluation results successfully to: {results_file}")
        
    except Exception as e:
        print(f"Ragas Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_ragas_evaluation()
