import os
import json
import time
import sys

# Add the workspace root folder to the python path to resolve local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from ingest import process_documents, WORKSPACE_DIR

# Load environment variables
load_dotenv()

def generate_golden_set():
    print("Loading source documents...")
    raw_docs = process_documents(WORKSPACE_DIR)
    if not raw_docs:
        print("No documents found in the workspace.")
        return
        
    print(f"Loaded {len(raw_docs)} document pages/blocks.")
    
    # Let's take a sample of document blocks to generate a diverse set of questions
    # We will pick paragraphs from different files to ensure good coverage
    docs_by_source = {}
    for doc in raw_docs:
        src = doc["source"]
        if src not in docs_by_source:
            docs_by_source[src] = []
        docs_by_source[src].append(doc)
        
    # Prepare text snippets from each source to feed to Gemini
    context_blocks = []
    for src, pages in docs_by_source.items():
        # Select up to 3 pages/sections evenly spaced from each document
        step = max(1, len(pages) // 3)
        selected_pages = pages[::step][:3]
        for p in selected_pages:
            # Truncate text block to keep context size reasonable
            text_snippet = p["text"][:1200]
            context_blocks.append(
                f"Document: {p['source']}\n"
                f"Page/Section: {p['page']}\n"
                f"Content:\n{text_snippet}\n"
                f"---"
            )
            
    full_context = "\n".join(context_blocks)
    
    # Initialize Gemini client
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
         raise ValueError("GEMINI_API_KEY not found in environment.")
         
    client = genai.Client(api_key=gemini_key.strip())
    
    prompt = (
        f"You are a sports anti-doping policy expert compiling an evaluation golden dataset.\n"
        f"Based strictly on the provided document snippets, please generate exactly 10 distinct, realistic, "
        f"and precise compliance questions.\n"
        f"Each question must include:\n"
        f"1. A clear, specific question.\n"
        f"2. A detailed, accurate ground_truth answer derived *strictly* from the text snippet.\n"
        f"3. The exact source_doc filename (e.g. '2021_wada_code.pdf').\n"
        f"4. The reference_location (e.g. 'Page 12' or 'Section 3').\n\n"
        f"Ensure questions cover different aspects: therapeutic use exemptions, prohibited substances, "
        f"whistleblower protections, SCA rules, and anti-doping violations.\n\n"
        f"Output the results STRICTLY as a plain JSON list of objects. Each object must have these keys:\n"
        f"\"id\" (integer, 1 to 10), \"question\" (string), \"ground_truth\" (string), \"source_doc\" (string), \"reference_location\" (string).\n"
        f"Do not wrap the JSON in markdown code blocks like ```json ... ```, conversational text, or explanations. Provide only raw JSON."
    )
    
    print("Calling Gemini API to synthesize the golden set...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(text=f"Document Snippets:\n{full_context}\n\n{prompt}")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        raw_json = response.text.strip()
        
        # Clean markdown wrappers if any
        if raw_json.startswith("```"):
            lines = raw_json.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_json = "\n".join(lines).strip()
            
        golden_set = json.loads(raw_json)
        
        # Ensure evaluate directory exists
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(eval_dir, exist_ok=True)
        
        output_file = os.path.join(eval_dir, "golden_set.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(golden_set, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Golden set successfully created at: {output_file}")
        print(f"Generated {len(golden_set)} question-answer evaluation pairs.")
        
    except Exception as e:
        print(f"Error generating golden set: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_golden_set()
