import requests
import json
from typing import Generator, Dict, Any

import os
import requests
from typing import Any, Generator
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class LLMService:
    """
    A simple, synchronous client for a custom LLM web service.
    
    This class exclusively uses a streaming endpoint for all communication,
    loading its configuration from environment variables.
    """

    def __init__(self):
        """
        Initializes the client. It reads the base URL from the 'MODEL_URL'
        environment variable defined in a .env file.
        
        Raises:
            ValueError: If the MODEL_URL is not found in the environment.
        """
        base_url = os.getenv("MODEL_URL")
        if not base_url:
            raise ValueError(
                "MODEL_URL not found in environment variables. "
                "Please create a .env file with 'MODEL_URL=http://your_service_url'"
            )
            
        self.base_url = base_url
        # Both invoke() and stream() will use the same streaming endpoint.
        self.generate_url = f"{self.base_url}/generate_stream"
        print(f"✅ LLMService initialized for endpoint: {self.generate_url}")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Sends a request and returns the complete text response.
        
        This method internally uses the streaming endpoint and concatenates
        all chunks into a single string before returning.

        Args:
            prompt (str): The prompt to send to the model.
            **kwargs: Additional generation parameters (e.g., temperature, max_tokens).

        Returns:
            str: The fully generated text from the model.
        """
        try:
            # Consume the generator provided by the stream method
            chunks = [chunk for chunk in self.stream(prompt, **kwargs)]
            return "".join(chunks)
        except requests.exceptions.RequestException as e:
            # The error is already printed in the stream method, so we just re-raise.
            raise

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Connects to the streaming endpoint and yields text chunks as they arrive.
        This is a synchronous generator.

        Args:
            prompt (str): The prompt to send to the model.
            **kwargs: Additional generation parameters (e.g., temperature, max_tokens).

        Yields:
            Generator[str, None, None]: A generator that yields text chunks.
        """
        payload = {"prompt": prompt, **kwargs}
        try:
            # The 'stream=True' parameter is crucial.
            # It tells 'requests' not to download the whole body at once.
            with requests.post(self.generate_url, json=payload, timeout=60, stream=True) as response:
                response.raise_for_status()
                
                # iter_content with decode_unicode decodes byte chunks into strings.
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
                        
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] An error occurred during API request: {e}")
            raise

#-------------------------------------------------------------------------------------

def main():
    """
    Demonstrates the usage of the LLM service with English and Bengali prompts.
    """
    try:
        # Initialize the service client (it reads config from .env)
        service = LLMService()
    except ValueError as e:
        print(f"[Configuration Error] {e}")
        return

    # --- 1. Test English Prompts ---
    print("\n" + "="*60)
    print("--- 1. Testing English Prompts ---")
    print("="*60)

    # a) Test the synchronous invoke() method
    print("\n--- a) Testing invoke() method (English) ---")
    try:
        invoke_prompt = "Write a short, dramatic story about a lonely lighthouse keeper who discovers a message in a bottle."
        response_text = service.invoke(
            prompt=invoke_prompt,
            temperature=0.8,
            max_tokens=256
        )
        print("\n[Full Response from invoke()]:")
        print(response_text)
    except Exception as e:
        print(f"\n[Invoke Failed]: {e}")
        
    # b) Test the synchronous stream() method
    print("\n--- b) Testing stream() method (English)---")
    try:
        stream_prompt = "What are the three most important features of the NVIDIA A6000 GPU for AI workloads?"
        
        print(f"\n[Streaming Response for: '{stream_prompt}']:")
        for chunk in service.stream(prompt=stream_prompt, temperature=0.2, max_tokens=150):
            print(chunk, end="", flush=True)

        print("\n--- Stream finished ---\n")
        
    except Exception as e:
        print(f"\n[Stream Failed]: {e}")

    # --- 2. Test Bengali (Bangla) Prompts ---
    print("\n" + "="*60)
    print("--- 2. Testing Bengali (Bangla) Prompts ---")
    print("="*60)

    # a) Test invoke() with a Bengali prompt
    print("\n--- a) Testing invoke() method (Bengali) ---")
    try:
        invoke_prompt_bn = "বাংলাদেশের রাজধানীর নাম কি এবং এটি কোন নদীর তীরে অবস্থিত?"
        print(f"\n[Prompt]: {invoke_prompt_bn}")
        response_text_bn = service.invoke(
            prompt=invoke_prompt_bn,
            temperature=0.3,
            max_tokens=100
        )
        print("\n[Full Response from invoke()]:")
        print(response_text_bn)
    except Exception as e:
        print(f"\n[Invoke Failed]: {e}")

    # b) Test stream() with a Bengali prompt
    print("\n--- b) Testing stream() method (Bengali) ---")
    try:
        stream_prompt_bn = "বাংলা নববর্ষের ইতিহাস সম্পর্কে একটি সংক্ষিপ্ত অনুচ্ছেদ লিখুন।"
        print(f"\n[Streaming Response for: '{stream_prompt_bn}']:")
        
        for chunk in service.stream(prompt=stream_prompt_bn, temperature=0.7, max_tokens=256):
            print(chunk, end="", flush=True)
            
        print("\n--- Stream finished ---\n")

    except Exception as e:
        print(f"\n[Stream Failed]: {e}")


if __name__ == "__main__":
    main()