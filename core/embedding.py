import os
import json
import requests
import math
from typing import List
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# TASK_MAPPING remains the same (zero-indexed)
TASK_MAPPING = {
    'retrieval.query': 0,
    'retrieval.passage': 1,
    'separation': 2,
    'classification': 3,
    'text-matching': 4,
}

class JinaV3ApiEmbeddings:
    """
    LangChain-compatible embedding class that calls a Jina V3 FastAPI service.
    It reads its configuration from a .env file and supports automatic request batching.
    """
    def __init__(self, timeout: int = 60, batch_size: int = 4):
        """
        Initializes the API-based embedder.

        Args:
            timeout (int): The request timeout in seconds for each batch.
            batch_size (int): The maximum number of texts to send in a single API call.
        
        Raises:
            ValueError: If EMBEDDING_URL is not set in the .env file.
        """
        embedding_url = os.getenv("EMBEDDING_URL")
        if not embedding_url:
            raise ValueError(
                "EMBEDDING_URL not found in environment variables. "
                "Please create a .env file with 'EMBEDDING_URL=http://your_service_url'"
            )

        self.timeout = timeout
        self.batch_size = batch_size
        self.api_url = f"{embedding_url}/v1/embeddings"
        print(f"✅ JinaV3ApiEmbeddings configured for: {self.api_url}")
        print(f"   Max batch size set to: {self.batch_size}")

    def _embed_batch(self, texts: List[str], task_id: int) -> List[List[float]]:
        """A private method to handle the API call for a single batch."""
        if not texts:
            return []

        payload_inputs = [{"text": str(text), "task_id": task_id} for text in texts]
        payload = {"inputs": payload_inputs}

        try:
            response = requests.post(
                self.api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            # Sort results by original index to ensure order is maintained
            data = sorted(response_json['data'], key=lambda item: item['index'])
            return [item['embedding'] for item in data]
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error connecting to Jina API: {e}")
            print(f"   Response Body: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Jina API at {self.api_url}. Error: {e}")
            raise

    def embed(self, texts: List[str], task: str) -> List[List[float]]:
        """
        Generic method to embed a list of texts for a specified task, with batching.

        Args:
            texts (List[str]): The list of texts to embed.
            task (str): The embedding task to perform.

        Returns:
            List[List[float]]: A list of embeddings in the same order as the input texts.
        """
        if task not in TASK_MAPPING:
            raise ValueError(f"Invalid task '{task}'. Available tasks are: {list(TASK_MAPPING.keys())}")
        
        task_id = TASK_MAPPING[task]
        all_embeddings = []
        
        num_texts = len(texts)
        num_batches = math.ceil(num_texts / self.batch_size)

        print(f"-> Embedding {num_texts} texts for task '{task}' in {num_batches} batches...")

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            print(f"   - Processing batch {batch_num} of {num_batches} (size: {len(batch_texts)})...")
            
            batch_embeddings = self._embed_batch(batch_texts, task_id)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convenience wrapper for embed(..., task='retrieval.passage'). Inherits batching."""
        return self.embed(texts, task='retrieval.passage')

    def embed_query(self, text: str) -> list[float]:
        """Convenience wrapper for embed(..., task='retrieval.query'). Inherits batching."""
        embeddings = self.embed([text], task='retrieval.query')
        if not embeddings:
            raise ValueError("API returned no embedding for the query.")
        return embeddings[0]


# --- Test code demonstrating the new functionality ---
if __name__ == "__main__":
    try:
        # --- Setup: Load Configuration ---
        # The embedder now reads its URL from the .env file automatically
        embedder = JinaV3ApiEmbeddings(batch_size=4)
        
        
        # --- Test 1: Bengali Government Service Queries ---
        print("### Test 1: Embedding Bengali Government Service Queries ###")
        
        # a) Batch embedding documents (passages)
        bengali_documents = [
            "কিভাবে জন্ম নিবন্ধন সনদের জন্য আবেদন করতে হয়?",
            "জাতীয় পরিচয়পত্র সংশোধন করার প্রক্রিয়া কি?",
            "পাসপোর্ট রিনিউ করতে কি কি কাগজপত্র লাগে?",
            "জমির খতিয়ান তোলার নিয়ম ও খরচ কেমন?"
        ]
        
        print("\n--- a) Testing embed_documents() with Bengali text ---")
        doc_embeddings_bn = embedder.embed_documents(bengali_documents)
        print(f"\nSuccessfully received {len(doc_embeddings_bn)} embeddings for {len(bengali_documents)} Bengali documents.")
        assert len(doc_embeddings_bn) == len(bengali_documents)
        print("✅ Assertion Passed: Input and output counts match.")
        if doc_embeddings_bn:
            print(f"   Dimension of embeddings: {len(doc_embeddings_bn[0])}")

        # b) Single embedding for a query
        print("\n--- b) Testing embed_query() with a Bengali query ---")
        bengali_query = "অনলাইনে ট্রেড লাইসেন্স কিভাবে পাবো?"
        print(f"Query: '{bengali_query}'")
        query_embedding_bn = embedder.embed_query(bengali_query)
        print(f"\nSuccessfully received 1 embedding for the Bengali query.")
        if query_embedding_bn:
            print(f"   Dimension of embedding: {len(query_embedding_bn)}")
        
        print("-" * 60)

        # --- Test 2: English Batching Functionality (as before) ---
        print("### Test 2: Verifying batching logic with English text ###")
        documents_for_batch_test = [
            "The sun is the star at the center of the Solar System.", "Jupiter is the largest planet.",
            "The Moon is Earth's only natural satellite.", "Mars is the fourth planet from the Sun.",
            "The Great Wall of China is a series of fortifications.", "The Amazon River is the largest river.",
            "An atom is the smallest unit of ordinary matter."
        ]
        
        print(f"\nTesting batching with {len(documents_for_batch_test)} documents and a batch size of {embedder.batch_size}")
        
        doc_embeddings_en = embedder.embed_documents(documents_for_batch_test)
        
        print(f"\nSuccessfully received {len(doc_embeddings_en)} embeddings for {len(documents_for_batch_test)} English documents.")
        assert len(documents_for_batch_test) == len(doc_embeddings_en)
        print("✅ Assertion Passed: Input and output counts match.")


    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")