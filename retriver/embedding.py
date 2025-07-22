import os
import json
import requests
import math
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load environment variables from the .env file
load_dotenv()

class JinaV3TritonEmbeddings:
    """
    An embedding client adapted to communicate directly with a Triton Inference Server
    running a Jina V3 model. It handles tokenization and Triton's specific
    request/response format.
    """
    def __init__(self, timeout: int = 60, batch_size: int = 8):
        """
        Initializes the Triton-based embedder.

        Args:
            timeout (int): The request timeout in seconds.
            batch_size (int): The maximum number of texts to process in a single batch.
        
        Raises:
            ValueError: If EMBEDDING_URL is not set in the .env file.
        """
        base_url = os.getenv("EMBEDDING_URL")
        if not base_url:
            raise ValueError(
                "EMBEDDING_URL not found in environment variables. "
                "Please create a .env file with 'EMBEDDING_URL=http://your_triton_url:port'"
            )

        self.timeout = timeout
        # The batch_size in the client should not exceed the max_batch_size in the model's Triton config
        self.batch_size = batch_size 
        self.api_url = f"{base_url.rstrip('/')}/v2/models/jina_query/infer"

        print("-> Initializing Hugging Face tokenizer for Jina V3...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v3", 
            trust_remote_code=True
        )
        print(f"✅ JinaV3TritonEmbeddings configured for: {self.api_url}")
        print(f"   Max batch size set to: {self.batch_size}")

    def _build_triton_payload(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, Any]:
        """A helper to construct the exact JSON payload Triton requires."""
        return {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                }
            ],
            "outputs": [
                {
                    # CHANGE: Request 'text_embeds' which is the last_hidden_state
                    "name": "text_embeds"
                }
            ]
        }

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """A private method to tokenize, send, and parse a single batch for Triton."""
        if not texts:
            return []

        # The original model supports up to 8192 tokens
        tokens = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=8192,
            return_tensors="np"
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        payload = self._build_triton_payload(input_ids, attention_mask)
        
        try:
            response = requests.post(
                self.api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            
            # CHANGE: The output name in the response matches the requested name 'text_embeds'
            output_data = next((out for out in response_json['outputs'] if out['name'] == 'text_embeds'), None)
            if output_data is None:
                raise ValueError("Triton response did not contain 'text_embeds' output.")

            shape = output_data['shape']
            # This is the last_hidden_state from the model
            flat_embeddings = np.array(output_data['data'], dtype=np.float32)
            last_hidden_state = flat_embeddings.reshape(shape)

            # Perform mean pooling using the attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = np.maximum(input_mask_expanded.sum(1), 1e-9)
            pooled_embeddings = sum_embeddings / sum_mask

            # CHANGE: Add L2 normalization step
            normalized_embeddings = pooled_embeddings / np.linalg.norm(pooled_embeddings, ord=2, axis=1, keepdims=True)
            
            return normalized_embeddings.tolist()

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error connecting to Triton: {e}")
            print(f"   Response Body: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Triton at {self.api_url}. Error: {e}")
            raise

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Embeds a list of queries using batching.
        """
        all_embeddings = []
        num_queries = len(queries)
        num_batches = math.ceil(num_queries / self.batch_size)

        print(f"-> Embedding {num_queries} queries in {num_batches} batches...")

        for i in range(0, num_queries, self.batch_size):
            batch_texts = queries[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            print(f"   - Processing batch {batch_num} of {num_batches} (size: {len(batch_texts)})...")
            
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query."""
        embeddings = self._embed_batch([text])
        if not embeddings:
            raise ValueError("API returned no embedding for the query.")
        return embeddings[0]


# --- Test code demonstrating the new Triton-compatible client ---
if __name__ == "__main__":
    try:
        # This will now automatically connect to your Triton server via the gateway
        # Ensure batch_size does not exceed the `max_batch_size` in your Triton config.pbtxt
        embedder = JinaV3TritonEmbeddings(batch_size=4) 
        
        # --- Test 1: Bengali Government Service Queries ---
        print("\n### Test 1: Embedding Bengali Government Service Queries ###")
        
        bengali_documents_as_queries = [
            "কিভাবে জন্ম নিবন্ধন সনদের জন্য আবেদন করতে হয়?",
            "জাতীয় পরিচয়পত্র সংশোধন করার প্রক্রিয়া কি?",
            "পাসপোর্ট রিনিউ করতে কি কি কাগজপত্র লাগে?",
            "জমির খতিয়ান তোলার নিয়ম ও খরচ কেমন?"
        ]
        
        print("\n--- a) Testing embed_queries() with Bengali text ---")
        query_embeddings_bn = embedder.embed_queries(bengali_documents_as_queries)
        print(f"\nSuccessfully received {len(query_embeddings_bn)} embeddings for {len(bengali_documents_as_queries)} Bengali queries.")
        assert len(query_embeddings_bn) == len(bengali_documents_as_queries)
        print("✅ Assertion Passed: Input and output counts match.")
        if query_embeddings_bn:
            print(f"   Dimension of embeddings: {len(query_embeddings_bn[0])}")

        print("\n--- b) Testing embed_query() with a Bengali query ---")
        bengali_query = "অনলাইনে ট্রেড লাইসেন্স কিভাবে পাবো?"
        print(f"Query: '{bengali_query}'")
        single_query_embedding_bn = embedder.embed_query(bengali_query)
        print(f"\nSuccessfully received 1 embedding for the Bengali query.")
        if single_query_embedding_bn:
            print(f"   Dimension of embedding: {len(single_query_embedding_bn)}")
        
        print("-" * 60)

        # --- Test 2: English Batching Functionality ---
        print("### Test 2: Verifying batching logic with English text ###")
        english_queries_for_batch_test = [
            "The sun is the star at the center of the Solar System.", "Jupiter is the largest planet.",
            "The Moon is Earth's only natural satellite.", "Mars is the fourth planet from the Sun.",
            "The Great Wall of China is a series of fortifications.", "The Amazon River is the largest river.",
            "An atom is the smallest unit of ordinary matter."
        ]
        
        print(f"\nTesting batching with {len(english_queries_for_batch_test)} queries and a batch size of {embedder.batch_size}")
        
        doc_embeddings_en = embedder.embed_queries(english_queries_for_batch_test)
        
        print(f"\nSuccessfully received {len(doc_embeddings_en)} embeddings for {len(english_queries_for_batch_test)} English queries.")
        assert len(english_queries_for_batch_test) == len(doc_embeddings_en)
        print("✅ Assertion Passed: Input and output counts match.")


    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")