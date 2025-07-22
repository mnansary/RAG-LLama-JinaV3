# retriver/logic.py

import os
import warnings
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# CORRECTED IMPORT: Use a relative import to find the sibling file.
from .embedding import JinaV3TritonEmbeddings

load_dotenv()
warnings.filterwarnings("ignore")

# I've renamed the class to RetrieverService for clarity, you can keep Retriever if you prefer
class RetrieverService:
    def __init__(self, vector_db_path: str):
        print("Initializing RetrieverService...")
        self.embedding_model = JinaV3TritonEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        try:
            db_count = self.vectorstore._collection.count()
            print(f"✅ RetrieverService initialized. DB at '{vector_db_path}' has {db_count} docs.")
        except Exception as e:
            print(f"⚠️ Warning: Could not get document count from vector store: {e}")

    # This is the single-query method for simple use cases
    def retrieve(self, query: str, k: int = 3, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        print(f"-> Performing single retrieval for query: \"{query}\"")
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filters
        )
        return self._format_results(docs_with_scores)

    # ADDED BACK: The crucial batch retrieval method for the API
    def retrieve_batch(self, queries: List[str], embeddings: List[List[float]], k_values: List[int], filters_list: List[Dict]) -> List[List[Dict[str, Any]]]:
        """
        Retrieves documents for a BATCH of queries using pre-computed embeddings.
        """
        results_batch = []
        for i in range(len(queries)):
            docs_with_scores = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                embedding=embeddings[i], k=k_values[i], filter=filters_list[i]
            )
            formatted_results = self._format_results(docs_with_scores)
            results_batch.append(formatted_results)
        return results_batch

    # ADDED BACK: The helper method to format results consistently
    def _format_results(self, docs_with_scores: List[tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Helper to format documents and scores into a clean dictionary list."""
        retrieved_passages = []
        if not docs_with_scores:
            return []
        
        for doc, score in docs_with_scores:
            # relevance_scores are 0-1 (higher is better), distance scores are the opposite
            retrieved_passages.append({
                "text": doc.page_content,
                "url": doc.metadata.get("url", "URL not found"),
                "score": score,
                "metadata": doc.metadata
            })
        
        retrieved_passages.sort(key=lambda x: x["score"], reverse=True)
        return retrieved_passages