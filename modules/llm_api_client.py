import asyncio
import os
from typing import Optional, Dict, Any, List
import openai

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
