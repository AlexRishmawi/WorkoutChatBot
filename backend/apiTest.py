"""
test_api.py
-----------
Interactive & Automated verification script for the Workout RAG FastAPI service.
Tests:
  1. GET  /status   - Current topology data
  2. POST /upload   - Uploads a training spreadsheet & builds indices
  3. POST /query    - Chats with the backend using conversational history
  4. POST /reset    - Flushes vectors and extracted JSON files
"""

import os
import sys
import json

try:
    import requests
except ImportError:
    print("This test script requires the 'requests' library.")
    print("Please run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    # Call load_dotenv immediately so it parses our local environment configuration file
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv is not installed. Will look directly at environment variables.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
# Adjust this to point to your actual spreadsheet file
XLSX_PATH = "WorkoutProgram.xlsx"  

# Resolve API key from environment, supporting both keys configured across your files
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def print_section(title: str):
    print("\n" + "═" * 70)
    print(f" {title}")
    print("═" * 70)


def check_server_running() -> bool:
    """Verifies that the FastAPI server is listening before starting tests."""
    try:
        requests.get(BASE_URL, timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        print(f" Error: Cannot connect to FastAPI server at {BASE_URL}")
        print("Please ensure your server is running by executing:")
        print("   python app.py")
        return False


def test_status():
    """Tests GET /status route."""
    print_section("Testing /status Endpoint")
    
    url = f"{BASE_URL}/status"
    response = requests.get(url)
    
    print(f"HTTP Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def test_upload(xlsx_path: str):
    """Tests POST /upload route."""
    print_section(f"Testing /upload Endpoint with '{xlsx_path}'")
    
    if not os.path.exists(xlsx_path):
        print(f" Aborted: Local file '{xlsx_path}' not found.")
        print("   To test the upload endpoint, please copy an actual .xlsx workout log")
        print("   to this directory and update the XLSX_PATH variable in this script.")
        return False

    url = f"{BASE_URL}/upload"
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    # Send as multipart/form-data
    with open(xlsx_path, "rb") as f:
        files = {"file": (os.path.basename(xlsx_path), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(url, headers=headers, files=files)
        
    print(f"HTTP Status Code: {response.status_code}")
    if response.status_code == 200:
        print(" Success! Program uploaded and indexed.")
        print(json.dumps(response.json(), indent=2))
        return True
    else:
        print(f" Upload Failed: {response.text}")
        return False


def test_query(question: str, history: list = None) -> dict:
    """Tests POST /query route with query condensation and chat history."""
    if history is None:
        history = []
        
    print_section(f"Testing /query Endpoint: '{question}'")
    print(f"Active History Depth: {len(history)} turns")
    
    url = f"{BASE_URL}/query"
    headers = {
        "Content-Type": "application/json"
    }
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    payload = {
        "question": question,
        "chat_history": history,
        "use_reranker": True,
        "model": "gemini-2.5-flash"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    print(f"HTTP Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"\n Condensed Standalone Query:\n   \"{data['standalone_query']}\"")
        print(f"\n LLM Answer:\n{data['answer']}")
        print(f"\n Retrieved Sources: {len(data['sources'])} chunks")
        for idx, src in enumerate(data["sources"][:5], 1):
            print(f"   [{idx}] {src.get('week')} | {src.get('day')} | {src.get('exercise_name')}")
        if len(data["sources"]) > 5:
            print(f"   ... and {len(data['sources']) - 5} more sources.")
        return data
    else:
        print(f" Query Failed: {response.text}")
        return {}


def test_reset():
    """Tests POST /reset route."""
    print_section("Testing /reset Endpoint")
    url = f"{BASE_URL}/reset"
    response = requests.post(url)
    print(f"HTTP Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))


# ---------------------------------------------------------------------------
# Execution Execution Sequence
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not check_server_running():
        sys.exit(1)

    # Inform the user what API key was loaded
    if API_KEY:
        masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) > 8 else "Found"
        print(f" API Key resolved: {masked_key}")
    else:
        print(" Warning: No Gemini API Key was found in your environment or .env file.")

    # 1. Inspect initial server state
    status_data = test_status()
    
    # 2. Check if we need to upload a spreadsheet to seed the RAG pipeline
    is_indexed = status_data.get("is_indexed", False)
    
    if not is_indexed:
        print("\n Server does not have an active program indexed.")
        # If running from command line, allow passing spreadsheet path as argument
        test_file = sys.argv[1] if len(sys.argv) > 1 else XLSX_PATH 
        uploaded = test_upload(test_file)
        if not uploaded:
            print("\n Skipping query tests since no dataset could be uploaded.")
            sys.exit(1)
    else:
        print("\n Server is already initialized with an active dataset. Proceeding directly to queries.")

    # 3. Simulate a multi-turn conversation to test conversational condensation (history)
    chat_history = []
    
    # Turn 1: Initial query
    q1 = "What did I do for my primary Bench Press on Week 1?"
    r1 = test_query(q1, chat_history)
    
    if r1:
        # Append Turn 1 history
        chat_history.append({"role": "user", "content": q1})
        chat_history.append({"role": "assistant", "content": r1["answer"]})
        
        # Turn 2: Contextual follow-up query relying on query condensation
        q2 = "How did that weight compare to Week 2?"
        r2 = test_query(q2, chat_history)
        
        if r2:
            # Append Turn 2 history
            chat_history.append({"role": "user", "content": q2})
            chat_history.append({"role": "assistant", "content": r2["answer"]})
            
            # Turn 3: Multi-week trend tracking (should trigger exhaustive history retrieval)
            q3 = "Show me my entire bench progression over time"
            test_query(q3, chat_history)

    print_section("RAG API Verification Complete!")