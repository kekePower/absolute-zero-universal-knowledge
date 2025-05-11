import asyncio
import os
import requests
import time
import json
from typing import Optional, Dict, Any, List
import openai
from .config import (PRIMARY_API_KEY, 
                     OLLAMA_ENABLED, OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME, 
                     API_RPM_LIMIT, MIN_ITER_SLEEP)

# --- Async API Client ---
async def query_llm_api(user_content: str, temperature: float, max_tokens: int, model_name: str, api_base_url: str, api_key: str) -> Optional[str]:
    if api_key == "<Your_API_Key_HERE>" or not api_key :
        print(f"ERROR: API Key for {model_name} is not set. Please set PRIMARY_API_KEY environment variable.")
        return None
    if not model_name:
        print(f"Warning: Model name is empty for a query. Skipping API call.")
        return None # Explicitly return None if no model_name

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )
    try:
        chat_completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            # stream=False # Explicitly False for non-streaming
        )
        response_content = chat_completion.choices[0].message.content
        # print(f"DEBUG: LLM Response ({model_name}): {response_content[:100]}...") # Optional: for debugging
        return response_content
    except openai.APIConnectionError as e:
        print(f"ERROR: API Connection Error for {model_name}: {e}")
        # Potentially implement retry logic here if desired
    except openai.RateLimitError as e:
        print(f"ERROR: API Rate Limit Error for {model_name}: {e}. Consider adjusting API_RPM_LIMIT or script logic.")
        # Potentially implement retry with backoff here
    except openai.APIStatusError as e:
        print(f"ERROR: API Status Error for {model_name} (Status: {e.status_code}): {e.response}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred with {model_name}: {e}")
    return None # Return None on any error

# --- New Ollama API Client Function ---
def get_ollama_completion(prompt_text: str, system_prompt: str, model_name: str = OLLAMA_MODEL_NAME) -> str | None:
    """Interacts with a local Ollama instance to get a completion.

    Args:
        prompt_text: The user prompt for the Ollama model.
        system_prompt: The system prompt to guide the Ollama model.
        model_name: The Ollama model to use (defaults to OLLAMA_MODEL_NAME from config).

    Returns:
        The Ollama model's response text, or None if disabled or an error occurs.
    """
    if not OLLAMA_ENABLED:
        # If Ollama is not enabled, we might return the original prompt or handle it differently.
        # For now, let's indicate it's not processed by Ollama or return None.
        # Depending on usage, returning prompt_text directly might be an option if the flow expects a string.
        print("Ollama integration is disabled. Skipping refinement.")
        return None # Or prompt_text, depending on how it's integrated into the main flow

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "system": system_prompt,
        "stream": False  # We want the full response, not a stream
    }
    
    api_url = f"{OLLAMA_API_BASE_URL}/generate"

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        response_data = response.json()
        refined_instructions = response_data.get("response")
        
        if refined_instructions:
            # print(f"Ollama ({model_name}) refined instructions: {refined_instructions[:200]}...") # For debugging
            return refined_instructions.strip()
        else:
            print(f"Error: No 'response' field in Ollama output. Full response: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API at {api_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Ollama: {e}. Response text: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while calling Ollama: {e}")
        return None
