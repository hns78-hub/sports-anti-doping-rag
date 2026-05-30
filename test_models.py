import os
from google import genai

gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    # Read from key input or ask user, but let's see if we can find it or ask user
    # For now, let's assume we can fetch it or we can pass a dummy key if listing doesn't need auth,
    # but listing models DOES need a valid key.
    # Let's try to print the error or use the active key.
    print("GEMINI_API_KEY not found in environment.")
    # We will list models using the client
else:
    client = genai.Client(api_key=gemini_key)
    try:
        print("Listing models...")
        for m in client.models.list():
            print(f"Name: {m.name}, Supported Methods: {m.supported_stage}")
    except Exception as e:
        print(f"Error: {e}")
