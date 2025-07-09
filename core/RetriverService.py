import os
import json
import warnings
from typing import Dict, Any, List

# --- Environment and Local Imports ---
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

# Import your configured JinaV3ApiEmbeddings class from its file
# This assumes 'embedding.py' is in the same directory.
from .embedding import JinaV3ApiEmbeddings

# Load environment variables from .env file
load_dotenv()
warnings.filterwarnings("ignore")

class RetrieverService:
    def __init__(self, vector_db_path: str):
        """
        Initializes the retriever service.
        It automatically configures the embedding model from environment variables.
        
        Args:
            vector_db_path (str): The path to the persisted ChromaDB directory.
        """
        print("Initializing RetrieverService...")
        
        # The embedding model now configures itself by reading EMBEDDING_URL from .env
        self.embedding_model = JinaV3ApiEmbeddings()
        
        # Initialize ChromaDB with the specified path and the embedding function
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        try:
            db_count = self.vectorstore._collection.count()
            print(f"✅ RetrieverService initialized successfully. Vector store at '{vector_db_path}' contains {db_count} documents.")
        except Exception as e:
            # This can happen if the database is empty or has issues
            print(f"⚠️ Warning: Could not get document count from vector store: {e}")

    def retrieve(self, query: str, k: int = 3, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieves relevant passages with dynamic k and metadata filtering.

        Args:
            query (str): The user's question (in English or Bengali).
            k (int): The number of documents to retrieve.
            filters (dict, optional): A dictionary for metadata filtering.
                                      Example: {"category": "ভূমি সেবা"}

        Returns:
            dict: A dictionary containing the query and a list of retrieved passages.
        """
        print(f"\nPerforming retrieval for query: \"{query}\" with k={k} and filters={filters}")
        
        # Use similarity_search_with_score with dynamic k and the 'filter' argument
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filters
        )

        retrieved_passages = []
        if not docs_with_scores:
            print("No relevant documents found with the given criteria.")
        else:
            for doc, score in docs_with_scores:
                # The score from Chroma is a distance metric (lower is better)
                retrieved_passages.append({
                    "text": doc.page_content,
                    "url": doc.metadata.get("url", "URL not found"),
                    "score": score,
                    "metadata": doc.metadata
                })
            print(f"Found {len(retrieved_passages)} relevant passages.")
        
        # Sort by score (ascending, since lower distance is better)
        retrieved_passages.sort(key=lambda x: x["score"])

        return {
            "query": query,
            "retrieved_passages": retrieved_passages
        }

# --- Example Usage with Bengali Government Service Queries ---
if __name__ == "__main__":
    # 1. Load configuration from .env
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
    if not VECTOR_DB_PATH:
        raise ValueError("VECTOR_DB_PATH not set in the .env file.")

    # 2. Initialize the service
    # The service will fail gracefully if the path doesn't exist, but retrieval will be empty.
    retriever_service = RetrieverService(vector_db_path=VECTOR_DB_PATH)
    
    # --- Example 1: Simple search (default k=3) for a common service ---
    print("\n\n--- Example 1: Simple Search (default k=3) ---")
    query1 = "জন্ম নিবন্ধন সনদের জন্য আবেদন করার নিয়ম কি?"
    results_1 = retriever_service.retrieve(query=query1)
    print(json.dumps(results_1, indent=2, ensure_ascii=False))

    # --- Example 2: Search with a custom k value for broader results ---
    print("\n\n--- Example 2: Custom 'k' Search (k=5) ---")
    query2 = "সরকারি চাকরি এবং আবেদন প্রক্রিয়া"
    results_2 = retriever_service.retrieve(query=query2, k=5)
    print(json.dumps(results_2, indent=2, ensure_ascii=False))

    # --- Example 3: Search using a metadata filter to narrow down results ---
    # Imagine your documents have a "category" in their metadata.
    print("\n\n--- Example 3: Metadata Filter Search ---")
    query3 = "জমির মালিকানা যাচাই"
    filters3 = {"category": "ভূমি সেবা"} # Find docs about land ownership in the "Land Services" category
    results_3 = retriever_service.retrieve(query=query3, k=3, filters=filters3)
    print(json.dumps(results_3, indent=2, ensure_ascii=False))

    # --- Example 4: Combined search with both filter and custom k ---
    print("\n\n--- Example 4: Combined Filter and Custom 'k' Search ---")
    query4 = "নতুন ই-পাসপোর্ট"
    filters4 = {"category": "পাসপোর্ট ও ভিসা"} # Find docs in the "Passport & Visa" category
    results_4 = retriever_service.retrieve(query=query4, k=2, filters=filters4)
    print(json.dumps(results_4, indent=2, ensure_ascii=False))