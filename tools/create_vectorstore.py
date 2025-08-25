import os
import json
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoTokenizer
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
# CHANGE: Import Embeddings from LangChain for proper interface implementation
from langchain.embeddings.base import Embeddings
from tqdm import tqdm
import warnings

# Ignore common warnings
warnings.filterwarnings("ignore")

# --- Model & Path Configuration ---
# Model names as they are named in your Triton model repository
PASSAGE_MODEL_NAME = "jina_passage"
QUERY_MODEL_NAME = "jina_query"
# Foundational model for the tokenizer
TOKENIZER_NAME = "jinaai/jina-embeddings-v3"
# Sequence length must match the one used to build the TensorRT engine
MAX_SEQUENCE_LENGTH = 8192

# CHANGE: The class now inherits from LangChain's Embeddings base class
class JinaV3TritonEmbeddings(Embeddings):
    """
    An embedding client that communicates with Triton Inference Server for Jina V3 models.
    It handles separate passage and query models, performs the necessary post-processing
    (mean pooling and normalization), and conforms to the LangChain Embeddings interface.
    """
    def __init__(
        self,
        base_url: str,
        passage_model: str,
        query_model: str,
        tokenizer_name: str,
        timeout: int = 120,
    ):
        self.timeout = timeout
        self.passage_api_url = f"{base_url.rstrip('/')}/v2/models/{passage_model}/infer"
        self.query_api_url = f"{base_url.rstrip('/')}/v2/models/{query_model}/infer"
        
        print("-> Initializing Hugging Face tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        print("✅ Tokenizer loaded.")
        print(f"   - Passage embeddings will be sent to: {self.passage_api_url}")
        print(f"   - Query embeddings will be sent to: {self.query_api_url}")

    def _build_triton_payload(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, Any]:
        """Constructs the JSON payload for Triton's v2 HTTP endpoint."""
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
                    # CHANGE: Request the correct output tensor 'text_embeds',
                    # which represents the last hidden state from the model.
                    "name": "text_embeds"
                }
            ]
        }

    def _execute_inference(self, texts: List[str], api_url: str) -> List[List[float]]:
        """A generic method to tokenize, send, and correctly process a batch for Triton."""
        if not texts:
            return []

        tokens = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="np"
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        payload = self._build_triton_payload(input_ids, attention_mask)
        
        try:
            response = requests.post(
                api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            
            # CHANGE: Correctly parse the 'text_embeds' output
            output_data = next((out for out in response_json['outputs'] if out['name'] == 'text_embeds'), None)
            if output_data is None:
                raise ValueError("Triton response did not contain the 'text_embeds' output.")

            shape = output_data['shape']
            # This is the last_hidden_state from the model, not the final embedding
            last_hidden_state = np.array(output_data['data'], dtype=np.float32).reshape(shape)
            
            # CHANGE: Add the required mean pooling step
            # Use the attention_mask to correctly average the token embeddings
            input_mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = np.maximum(input_mask_expanded.sum(1), 1e-9)
            pooled_embeddings = sum_embeddings / sum_mask

            # CHANGE: Add the required L2 normalization step
            normalized_embeddings = pooled_embeddings / np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
            
            return normalized_embeddings.tolist()

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error connecting to Triton: {e}")
            print(f"   Response Body: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Triton at {api_url}. Error: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents (passages) using the 'jina_passage' model."""
        return self._execute_inference(texts, self.passage_api_url)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query using the 'jina_query' model."""
        embeddings = self._execute_inference([text], self.query_api_url)
        if not embeddings:
            raise ValueError("Triton API returned no embedding for the query.")
        return embeddings[0]


def create_vector_store_from_csv(
    csv_path: str,
    text_column: str,
    metadata_columns: list,
    vector_db_path: str,
    embedding_func: JinaV3TritonEmbeddings,
    batch_size: int = 16
) -> None:
    """Loads a CSV, processes it in batches, and creates a Chroma vector store."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\nSuccessfully loaded CSV from '{csv_path}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at '{csv_path}'")
        return

    documents_to_add = []
    print("-> Preparing documents from DataFrame...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        passage_text = row.get(text_column)
        if pd.isna(passage_text) or not str(passage_text).strip():
            continue
        metadata = {col: row.get(col, "") for col in metadata_columns if pd.notna(row.get(col))}
        doc = Document(page_content=str(passage_text), metadata=metadata)
        documents_to_add.append(doc)

    if not documents_to_add:
        print("No valid documents found in the CSV to add.")
        return

    print(f"\n-> Initializing or loading Chroma vector store at: {vector_db_path}")
    # This will load an existing DB or create a new one.
    # The embedding function is only used when adding new documents.
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embedding_func
    )

    print(f"-> Adding {len(documents_to_add)} documents to Chroma in batches of {batch_size}...")
    # This loop processes documents in batches, sending them to Chroma,
    # which then calls our custom embed_documents method internally.
    for i in tqdm(range(0, len(documents_to_add), batch_size), desc="Adding documents to Chroma"):
        batch = documents_to_add[i:i + batch_size]
        # Chroma's add_documents will call embedding_func.embed_documents on the text from the batch
        vectorstore.add_documents(documents=batch)

    print("-> Persisting the vector store to disk...")
    # Persisting is crucial to save the changes made by add_documents
    vectorstore.persist()
    
    try:
        # Get the total count from the collection to confirm success
        total_docs_in_db = vectorstore._collection.count()
        print(f"\n✅ Success! Vector store now contains {total_docs_in_db} documents.")
    except Exception as e:
        print(f"Could not retrieve final document count. Error: {e}")

    print(f"Vector store persisted at: {vector_db_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Configuration ---
    TRITON_SERVER_URL = "http://localhost:6000"
    CSV_FILE_PATH = "/home/vpa/Downloads/cleaned_data_20250825.csv" # Make sure this path is correct
    VECTOR_DB_PATH = "prototype" # Directory to save the ChromaDB files

    print("--- Starting Vector Store Creation Process ---")
    print(f"Triton Server URL set to: {TRITON_SERVER_URL}")

    # Initialize the Triton client embedding function
    embedding_function = JinaV3TritonEmbeddings(
        base_url=TRITON_SERVER_URL,
        passage_model=PASSAGE_MODEL_NAME,
        query_model=QUERY_MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME
    )

    # Define which columns from the CSV to use for content and metadata
    text_col = 'text' 
    metadata_cols = ['category','subcategory','service','topic','url','passage_id'] 

    # Run the main process to create the vector store
    create_vector_store_from_csv(
        csv_path=CSV_FILE_PATH,
        text_column=text_col,
        metadata_columns=metadata_cols,
        vector_db_path=VECTOR_DB_PATH,
        embedding_func=embedding_function,
        # Adjust batch size based on your GPU's VRAM and the max_batch_size in your Triton config
        # A larger batch size (e.g., 16, 32, 64) leads to much faster processing.
        batch_size=1
    )
    
    print("\n--- Process finished. ---")