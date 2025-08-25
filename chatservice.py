import json
import os
import re
from collections import deque
from typing import Dict, Any, Deque, Tuple, Generator

from dotenv import load_dotenv

# Correctly import the specified classes from your other modules
from core.DBService import RetrieverClient
from core.LLMservice import LLMService
from core.prompts import ANALYST_PROMPT, STRATEGIST_PROMPTS

# Load environment variables from your .env file
load_dotenv()


class ProactiveChatService:
    """
    Orchestrates a multi-stage chat pipeline involving an Analyst, a Retriever,
    and a Strategist to provide context-aware, fact-based responses streamed
    to the user.
    """

    def __init__(self, history_length: int = 5):
        """
        Initializes the chat service by setting up the necessary components.

        Args:
            history_length (int): The number of conversation turns to keep in memory.
        """
        print("Initializing ProactiveChatService...")

        # 1. Initialize the Retriever Client from DBService.py
        self.retriever = RetrieverClient()

        # 2. Initialize the LLM Service from LLMservice.py
        base_url = os.getenv("LLM_MODEL_BASE_URL")
        api_key = os.getenv("LLM_MODEL_API_KEY")
        model_name = os.getenv("LLM_MODEL_NAME")

        if not all([base_url, api_key, model_name]):
            raise ValueError(
                "One or more environment variables for the LLM service are missing. "
                "Please check your .env file for LLM_MODEL_BASE_URL, LLM_MODEL_API_KEY, and LLM_MODEL_NAME."
            )

        self.llm_service = LLMService(api_key=api_key, base_url=base_url, model=model_name)

        # 3. Initialize conversation history
        self.history: Deque[Tuple[str, str]] = deque(maxlen=history_length)
        print(f"✅ ProactiveChatService initialized successfully. History window: {history_length} turns.")

    def _format_history(self) -> str:
        """Formats the conversation history into a readable string for prompts."""
        if not self.history:
            return "No conversation history yet."
        return "\n".join([f"User: {user_q}\nAI: {ai_a}" for user_q, ai_a in self.history])

    def _run_analyst_stage(self, user_input: str, history_str: str) -> Dict[str, Any] | None:
        """
        Executes the Analyst stage to get a structured JSON plan from the LLM.
        """
        print("\n----- 🕵️ Analyst Stage -----")
        
        # Format the prompt using the LangChain template
        formatted_prompt = ANALYST_PROMPT.format_prompt(history=history_str, question=user_input)
        
        # --- FIX: Convert the PromptValue object to a string ---
        prompt_as_string = formatted_prompt.to_string()

        try:
            response_text = self.llm_service.invoke(
                prompt_as_string, # <-- Pass the string, not the object
                temperature=0.0,
                max_tokens=1024
            )
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                plan = json.loads(json_str)
                print("✅ Analyst plan generated and extracted successfully.")
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                return plan
            else:
                print(f"❌ CRITICAL: Analyst stage failed. No valid JSON block found in the response.")
                print(f"LLM Response was: {response_text}")
                return None
        except json.JSONDecodeError as e:
            print(f"❌ CRITICAL: Analyst stage failed with JSONDecodeError: {e}")
            return None
        except Exception as e:
            print(f"❌ CRITICAL: An unexpected error occurred in the Analyst stage: {e}")
            return None

    def _run_retriever_stage(self, plan: Dict[str, Any]) -> Tuple[str, list]:
        """
        Executes the Retriever stage based on the Analyst's plan.
        """
        print("\n----- 📚 Retriever Stage -----")
        query = plan.get("query_for_retriever")
        k = 1 # Default value as it's not in the analyst prompt
        filters = plan.get("metadata_filter", None)

        if not query:
            print("Skipping retrieval as no query was provided in the plan.")
            return "No retrieval was performed.", []

        print(f"🔍 Querying retriever with: '{query}', k={k}, filters={filters}")
        try:
            retrieval_results = self.retriever.retrieve(query, k=k, filters=filters)
            retrieved_passages = retrieval_results.get("retrieved_passages", [])

            if not retrieved_passages:
                print("⚠️ Retriever found no documents.")
                return "No information found matching the query.", []

            combined_context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_passages])
            print(f"✅ Retriever found {len(retrieved_passages)} documents.")
            return combined_context, retrieved_passages
        except Exception as e:
            print(f"❌ CRITICAL: An error occurred in the Retriever stage: {e}")
            return "An error occurred while retrieving information.", []

    def _run_strategist_stage(self, plan: Dict[str, Any], context: str, user_input: str, history_str: str) -> Generator[str, None, None]:
        """Executes the Strategist stage, streaming the final response."""
        print("\n----- 🎭 Strategist Stage -----")
        strategy = plan.get("response_strategy", "RESPOND_WARMLY")
        print(f"✍️ Executing strategy: '{strategy}'")

        prompt_template = STRATEGIST_PROMPTS.get(strategy)
        if not prompt_template:
            print(f"❌ WARNING: Invalid strategy '{strategy}'. Defaulting to warm response.")
            prompt_template = STRATEGIST_PROMPTS["RESPOND_WARMLY"]
        
        # Format the prompt using the LangChain template
        formatted_prompt = prompt_template.format_prompt(
            context=context,
            question=user_input,
            history=history_str
        )

        # --- FIX: Convert the PromptValue object to a string ---
        prompt_as_string = formatted_prompt.to_string()
        
        return self.llm_service.stream(
            prompt_as_string, # <-- Pass the string, not the object
            temperature=0.0,
            max_tokens=4096,
            repetition_penalty=1.2
        ),strategy

    def chat(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """
        Main entry point for the chat service. Orchestrates the pipeline and yields events.
        """
        print(f"\n==================== NEW CHAT TURN: User said '{user_input}' ====================")
        history_str = self._format_history()

        plan = self._run_analyst_stage(user_input, history_str)
        if not plan:
            yield {"type": "error", "content": "দুঃখিত, আমি এই মুহূর্তে আপনার অনুরোধটি প্রক্রিয়া করতে পারছি না। অনুগ্রহ করে আপনার প্রশ্নটি ভিন্নভাবে করার চেষ্টা করুন।"}
            return

        combined_context, retrieved_passages = self._run_retriever_stage(plan)

        answer_generator,strategy = self._run_strategist_stage(plan, combined_context, user_input, history_str)

        full_answer_list = []
        for chunk in answer_generator:
            full_answer_list.append(chunk)
            yield {
                "type": "answer_chunk",
                "content": chunk
            }

        final_answer = "".join(full_answer_list).strip()
        self.history.append((user_input, final_answer))
        if strategy in ["PROVIDE_DIRECT_INFO","REDIRECT_AND_CLARIFY"]:
            sources = []
            if retrieved_passages:
                unique_urls = set()
                unique_pids = set()
                for doc in retrieved_passages:
                    if doc.get("metadata") and doc["metadata"].get("url"):
                        unique_urls.add(doc["metadata"]["url"])
                    if doc.get("metadata") and doc["metadata"].get("passage_id"):
                        unique_pids.add(doc["metadata"]["passage_id"])
                sources = list(unique_urls)+[f"source_passage:{i}" for i in list(unique_pids)]
            
            print(f"\nFinal sources extracted: {sources}")

            yield {
                "type": "final_data",
                "content": {"sources": sources}
            }
        print("\n-------------------- STREAM COMPLETE --------------------")


if __name__ == "__main__":
    chat_service = ProactiveChatService(history_length=5)

    test_conversation = [
        "জন্ম নিবন্ধন করার প্রক্রিয়া কি?",
        "বিআরটিএ অফিসের ফোন নাম্বার কি?",
        "আপনি কি আমার হয়ে একটি আবেদনপত্র পূরণ করে দিতে পারবেন?",
        "আচ্ছা, বাংলাদেশের বর্তমান প্রধানমন্ত্রীর নাম কি?",
        "ধন্যবাদ",
    ]

    for turn in test_conversation:
        print(f"\n\n\n>>>>>>>>>>>>>>>>>> User Input: {turn} <<<<<<<<<<<<<<<<<<")
        print("\n<<<<<<<<<<<<<<<<<< Bot Response >>>>>>>>>>>>>>>>>>")

        final_sources = []
        try:
            for event in chat_service.chat(turn):
                if event["type"] == "answer_chunk":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "final_data":
                    final_sources = event["content"].get("sources", [])
                elif event["type"] == "error":
                    print(event["content"], end="", flush=True)

            if final_sources:
                print(f"\n\n[তথ্যসূত্র: {', '.join(final_sources)}]")

        except Exception as e:
            print(f"\n\nAn unexpected error occurred during the chat turn: {e}")

        print("\n<<<<<<<<<<<<<<<<<< End of Response >>>>>>>>>>>>>>>>>>")