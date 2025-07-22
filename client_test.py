# client_example.py

import requests
import json

API_URL = "http://localhost:9000/chat/stream"

# --- Conversation with User 1 ---
user_1_id = "user-alex-123"
print(f"--- Starting conversation for {user_1_id} ---")

queries_user_1 = [
    "জন্ম নিবন্ধন করার প্রক্রিয়া কি?",
    "ধন্যবাদ"
]

for query in queries_user_1:
    print(f"\n>>> User: {query}")
    print("<<< Bot: ", end="", flush=True)
    
    payload = {"user_id": user_1_id, "query": query}
    
    try:
        # The stream=True parameter is crucial
        with requests.post(API_URL, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an exception for bad status codes
            
            final_sources = []
            # Iterate over the response line by line
            for line in response.iter_lines():
                if line:
                    # Decode the line from bytes to a string and parse the JSON
                    event = json.loads(line.decode('utf-8'))
                    
                    if event["type"] == "answer_chunk":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "final_data":
                        final_sources = event["content"].get("sources", [])
                    elif event["type"] == "error":
                        print(f"\n[ERROR]: {event['content']}", end="", flush=True)
            
            if final_sources:
                print(f"\n[তথ্যসূত্র: {', '.join(final_sources)}]")
            print() # Newline for cleaner separation
            
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")

print("\n" + "="*50 + "\n")

# --- Conversation with User 2 (to demonstrate session isolation) ---
user_2_id = "user-brian-456"
print(f"--- Starting conversation for {user_2_id} ---")

queries_user_2 = [
    "বিআরটিএ অফিসের ফোন নাম্বার কি?"
]

for query in queries_user_2:
    print(f"\n>>> User: {query}")
    print("<<< Bot: ", end="", flush=True)
    
    payload = {"user_id": user_2_id, "query": query}
    with requests.post(API_URL, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                event = json.loads(line.decode('utf-8'))
                if event["type"] == "answer_chunk":
                    print(event["content"], end="", flush=True)
        print()