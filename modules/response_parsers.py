import re
import json
from typing import Optional, Tuple, Dict, Any

# --- Parsing for Critique and Revised Answer ---
def extract_from_critique_revise_response(llm_full_response: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not llm_full_response:
        return None, None
    
    critique_match = re.search(r"<critique>\s*([\s\S]+?)\s*</critique>", llm_full_response, re.IGNORECASE | re.DOTALL)
    revised_answer_match = re.search(r"<revised_answer>\s*([\s\S]+?)\s*</revised_answer>", llm_full_response, re.IGNORECASE | re.DOTALL)
    
    critique = critique_match.group(1).strip() if critique_match else None
    revised_answer = revised_answer_match.group(1).strip() if revised_answer_match else None
    
    # Fallback if primary tags are missing but content might still be there
    if not critique and not revised_answer and llm_full_response:
        # Heuristic: if it doesn't find the tags, but one of them is at the start, try to grab it
        if llm_full_response.lower().startswith("<critique>"):
            critique_match_fallback = re.match(r"<critique>\s*([\s\S]+?)\s*</critique>", llm_full_response, re.IGNORECASE | re.DOTALL)
            if critique_match_fallback : critique = critique_match_fallback.group(1).strip()
        elif llm_full_response.lower().startswith("<revised_answer>"):
            revised_answer_match_fallback = re.match(r"<revised_answer>\s*([\s\S]+?)\s*</revised_answer>", llm_full_response, re.IGNORECASE | re.DOTALL)
            if revised_answer_match_fallback : revised_answer = revised_answer_match_fallback.group(1).strip()

    if not critique and revised_answer:
        print(f"Warning: Extracted <revised_answer> but no <critique>. Response: {llm_full_response[:200]}...")
    elif critique and not revised_answer:
        print(f"Warning: Extracted <critique> but no <revised_answer>. Response: {llm_full_response[:200]}...")
    elif not critique and not revised_answer and llm_full_response:
         print(f"Warning: Could not extract <critique> or <revised_answer>. Response: {llm_full_response[:200]}...")

    return critique, revised_answer

# --- Parsing LLM's <answer> content ---
def extract_from_answer_tag(llm_full_response: Optional[str], task_type_for_heuristic: Optional[str] = None) -> Optional[str]:
    if not llm_full_response:
        return None

    last_think_end_pos = -1
    # Iterate to find the end position of the *last* occurrence of </think> or </thought>
    for think_tag_variant in [r"</think>", r"</thought>"]: # Consider making these constants
        matches = list(re.finditer(think_tag_variant, llm_full_response, re.IGNORECASE | re.DOTALL))
        if matches:
            last_think_end_pos = max(last_think_end_pos, matches[-1].end())
    
    if last_think_end_pos != -1:
        # Content after the last </think> or </thought> tag
        final_answer_content = llm_full_response[last_think_end_pos:].strip()
        if final_answer_content: 
            # print(f"  DEBUG: Extracted answer content after last think tag: '{final_answer_content[:100]}...'" ) # Optional debug
            return final_answer_content
        else:
            # This case means </think> was at the very end or only whitespace followed.
            print(f"Warning: Found '</think>' tag, but no substantial content followed. Response snippet: {llm_full_response[-100:]}...")
            return "" # Return empty string if think tag was last and no content followed
    else:
        # No </think> or </thought> tag found
        print(f"Warning: No '</think>' or '</thought>' tag found. Returning entire response. Response snippet: {llm_full_response[:200]}...")
        return llm_full_response.strip() # Return the full response, stripped, if no think tag

def _fix_json_string(json_str: str) -> str:
    # Replace Python-style booleans/None with JSON style
    json_str = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
    # Attempt to fix common single quote issues if they are likely used for keys/strings
    # This is a bit more robust than overly aggressive regexes for all quotes.
    try:
        # Regex to find 'key': or 'string', but be careful not to mess up words like "it's"
        # This specifically looks for single quotes around words that are likely keys or simple strings
        # Handles: 'key': value, {'key': 'value'}, ['string']
        # It's complex because of nested structures and avoiding apostrophes.
        # A simpler approach might be to try parsing and if it fails, then try replacing single quotes that are part of a key or string literal pattern.
        pass # Placeholder for more advanced single quote fixing if necessary after initial attempts
    except Exception as e:
        print(f"Error during specific single quote fixing: {e}")
    return json_str

def parse_json_from_answer(answer_content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not answer_content: return None

    # Attempt 1: Direct JSON parsing
    try: return json.loads(answer_content)
    except json.JSONDecodeError:
        # Attempt 2: Try with fixed common issues (True/False/None)
        fixed_str_initial = _fix_json_string(answer_content)
        try: return json.loads(fixed_str_initial)
        except json.JSONDecodeError:
            # Attempt 3: Look for JSON within Markdown code blocks ```json ... ```
            match = re.search(r"```json\s*([\s\S]+?)\s*```", answer_content, re.DOTALL)
            if match:
                json_from_markdown = match.group(1).strip()
                try: return json.loads(json_from_markdown) # Try parsing this directly
                except json.JSONDecodeError:
                    # Try fixing the markdown-extracted JSON too
                    fixed_json_from_markdown = _fix_json_string(json_from_markdown)
                    try: return json.loads(fixed_json_from_markdown)
                    except json.JSONDecodeError as e_md_fixed:
                        print(f"JSON parse from fixed markdown block also failed: {e_md_fixed}")
                        print(f"Content from markdown block: {json_from_markdown[:300]}...")
                        # Fall through to final error
                    else: print("Successfully parsed JSON from fixed markdown block.") # Success for fixed markdown
                else: print("Successfully parsed JSON from direct markdown block.") # Success for direct markdown
            
            # Attempt 4: If no markdown, or markdown failed, try more aggressive fixes on the initial fixed string
            # This is where you might add regex for single to double quotes if _fix_json_string doesn't handle it sufficiently
            # For example, trying to replace single quotes used for strings/keys if they are the cause.
            # Example (use with caution, can corrupt valid single quotes in text):
            # cautiously_fixed_str = re.sub(r"(?<!\\)'", '"', fixed_str_initial) # very broad, might break things
            # try: return json.loads(cautiously_fixed_str)
            # except json.JSONDecodeError as e_aggressive:
            # print(f"JSON parse failed even after aggressive quote replacement: {e_aggressive}")

            print(f"JSON parse failed for content (first 300 chars): {answer_content[:300]}...")
            return None
        else: print("Successfully parsed JSON after initial _fix_json_string.") # Success for initial fix
    else: print("Successfully parsed JSON directly.") # Success for direct parse
    return None # Should be unreachable if logic is correct, but as a safeguard
