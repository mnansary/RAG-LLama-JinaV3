import os
import json
import requests
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

# Load environment variables from the .env file in the current directory
load_dotenv()

class RetrieverClient:
    """
    A client to interact with the FastAPI Retriever Service.
    It handles one query at a time and simplifies making requests.
    """
    def __init__(self):
        """
        Initializes the client by loading the retriever service URL from the
        environment variables.
        
        Raises:
            ValueError: If RETRIEVER_URL is not set in the .env file.
        """
        self.retriever_url = os.getenv("RETRIEVER_URL")
        if not self.retriever_url:
            raise ValueError(
                "RETRIEVER_URL not found in environment variables. "
                "Please add 'RETRIEVER_URL=http://localhost:7000' to your .env file."
            )
        
        # The full API endpoint for retrieving documents
        self.api_endpoint = f"{self.retriever_url.rstrip('/')}/retrieve"
        print(f"✅ RetrieverClient initialized. Targeting API at: {self.api_endpoint}")

    def retrieve(
        self,
        query: str,
        k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sends a single query to the retriever service and returns the results.

        Args:
            query (str): The user's question (in English or Bengali).
            k (int): The number of documents to retrieve. Defaults to 3.
            filters (dict, optional): A dictionary for metadata filtering.
                                      Example: {"category": "ভূমি সেবা"}

        Returns:
            dict: The JSON response from the API, containing the query and a
                  list of retrieved passages.
                  
        Raises:
            requests.exceptions.RequestException: For connection errors, timeouts, etc.
            requests.exceptions.HTTPError: If the API returns an error status (e.g., 500).
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        print(f"\n-> Sending query to retriever: \"{query}\" (k={k}, filters={filters})")

        # Construct the JSON payload that matches the server's Pydantic model
        payload = {
            "query": query,
            "k": k,
            "filters": filters
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Set a reasonable timeout (in seconds)
            )
            
            # This will automatically raise an exception for 4xx or 5xx responses
            response.raise_for_status()
            
            print("✅ Successfully received a response from the API.")
            return response.json()

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e.response.status_code} {e.response.reason}")
            print(f"   Server response: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: Failed to connect to the Retriever API at {self.api_endpoint}.")
            print(f"   Please ensure the service is running and the URL is correct.")
            raise

# --- Example of how to use the client ---
if __name__ == "__main__":
    # First, make sure your .env file has this line:
    # RETRIEVER_URL="http://localhost:7000"

    try:
        # 1. Initialize the client
        client = RetrieverClient()
        
        # 2. Perform different types of queries
        
        # --- Example 1: Simple search (default k=3) ---
        print("\n" + "="*50)
        query1 = "জন্ম নিবন্ধন সনদের জন্য আবেদন করার নিয়ম কি?"
        results_1 = client.retrieve(query=query1)
        print("\n--- Results for Example 1 ---")
        print(json.dumps(results_1, indent=2, ensure_ascii=False))

        # --- Example 2: Search with a custom k value ---
        print("\n" + "="*50)
        query2 = "সরকারি চাকরি এবং আবেদন প্রক্রিয়া"
        results_2 = client.retrieve(query=query2, k=5)
        print("\n--- Results for Example 2 ---")
        print(json.dumps(results_2, indent=2, ensure_ascii=False))

        # --- Example 3: Search using a metadata filter ---
        print("\n" + "="*50)
        query3 = "জমির মালিকানা যাচাই"
        # This filter assumes your documents have a "category" in their metadata.
        filters3 = {"category": "ভূমি সেবা"}
        results_3 = client.retrieve(query=query3, k=2, filters=filters3)
        print("\n--- Results for Example 3 ---")
        print(json.dumps(results_3, indent=2, ensure_ascii=False))
        
        # --- Example 4: A query that might not find results ---
        print("\n" + "="*50)
        query4 = "How to build a spaceship with bananas?"
        results_4 = client.retrieve(query=query4, k=5)
        print("\n--- Results for Example 4 ---")
        print(json.dumps(results_4, indent=2, ensure_ascii=False))

    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"\nAn error occurred during client execution: {e}")