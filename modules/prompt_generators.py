from typing import Dict, List, Any, Optional
import random # Added for composite task generation if needed later
from .config import PRIMARY_MODEL_NAME # Needed for generate_evaluator_user_question

# --- R1 Prompt Template (Using <think>) ---
R1_PROMPT_WRAPPER = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first outlines the reasoning process in detail within <think> </think> tags, "
    "and then provides the final answer directly after the closing </think> tag.\n"
    "Example: <think> My detailed plan is to first A, then B, considering C. </think> final answer here.\n\n"
    "User: {question}\n\n"
    "Assistant: "
)

# --- Enhanced Task Generation Prompts ---
def get_base_proposer_prompt(task_type_description: str, k_examples: List[Dict[str, Any]], main_concept: Optional[str] = None, stochastic_seed: Optional[str] = None) -> str:
    json_open_brace = "{"
    json_close_brace = "}"

    prompt = f"""You are an AI assistant tasked with generating a new, challenging, and unique task of the type: '{task_type_description}'.
Your goal is to propose a task that is novel and has clearly defined success criteria.

<<<TASK_SPECIFIC_REFINEMENT_AREA>>>

Follow these structural guidelines meticulously:
1.  Think Step-by-Step: First, engage in a brief <think> block to outline your thought process for creating the task. This should include considerations for novelty, clarity of success criteria, and adherence to the task type. State the target novelty level you are aiming for (0.0 to 1.0).
2.  Provide JSON Output: Immediately after the closing </think> tag, provide your response as a single, valid JSON object.

JSON Structure Requirements:
- "task_description": A detailed description of the task.
- "success_criteria": Specific, measurable criteria for evaluating a successful solution (as a string).
- "domain_tags": A list of 2-5 relevant domain tags (e.g., ['physics', 'philosophy']).
- "novelty_level": An estimated novelty score from 0.0 (mundane) to 1.0 (paradigm-shifting).
- "task_type_generated": Echo back the precise task type you were asked to generate (e.g., '{task_type_description}').

CRITICAL: The JSON object MUST immediately follow the closing </think> tag. It MUST be a raw JSON string. DO NOT wrap it in Markdown code blocks (e.g., ```json ... ```). The JSON must start with {json_open_brace} and end with {json_close_brace}.
Ensure that all string values within the JSON (especially 'task_description' and 'success_criteria') are plain text and contain NO MARKDOWN formatting.

Here's an example of the complete output structure (content is illustrative):
<think>
I will design a '{task_type_description}' task. My aim is a novelty level of 0.7. I need to ensure the success criteria are very specific and test for deep understanding. The domain tags should be relevant to the problem's core.
</think>{json_open_brace}
  "task_description": "(A detailed description of the novel task, explicitly related to '{task_type_description}')...",
  "success_criteria": "(Clear, measurable, and unambiguous criteria for evaluating a solution. This should detail what a successful response must include and how it will be judged)...",
  "domain_tags": ["relevant_tag_1", "relevant_tag_2", "interdisciplinary_tag"],
  "novelty_level": 0.7,
  "task_type_generated": "(Must be exactly '{task_type_description}')"
{json_close_brace}
"""
    return prompt

def generate_synthesis_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "Synthesis of Disparate Paradigms: Combine concepts/methods from N (2-3) seemingly unrelated fields to solve a novel problem or create a new artifact."
    if use_composite and k_examples:
        # Simplified composite logic: just mention it in description for now
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_axioms_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "Generation of Novel Axioms and Exploration: Define a new set of axioms for a hypothetical system (mathematical, physical, social, etc.) and explore its logical consequences or emergent properties."
    if use_composite and k_examples:
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_epistemological_probe_task_user_question(
    k_examples: List[Dict[str, Any]], 
    use_composite: bool = False, 
    stochastic_seed: Optional[str] = None # This IS the seed/concept from LM or OpenAI
) -> str:
    
    base_task_guidance = """Your goal is to design a highly specific and challenging 'Epistemological Boundary Probe' task that probes the limits of knowledge or challenges foundational assumptions.

To do this, invent a detailed, deliberately ambiguous scenario (e.g., a fictional historical event, a scientific anomaly, a philosophical paradox). This scenario should:
1. Involve incomplete or contradictory 'evidence' or 'information.'
2. Compel the solver to construct multiple, potentially mutually incompatible, interpretations or narratives.
3. Require the solver to explicitly articulate the epistemic assumptions underpinning each interpretation/narrative.
4. Necessitate an analysis of key points of disagreement between interpretations.

The task should push the boundaries of what constitutes 'knowledge' in a given context and demand a nuanced understanding of the limitations of inquiry. Focus on creating a rich, specific scenario that demands rigorous critical evaluation of knowledge claims.
"""
    
    # If a stochastic seed (concept/question from LM or OpenAI) is provided, prepend it.
    if stochastic_seed:
        description = f"Using the following seed idea or question to inspire your generation: '{stochastic_seed}'.\n\n{base_task_guidance}"
    else:
        description = base_task_guidance
        
    if use_composite and k_examples:
        description += "\n\nWhen inventing your scenario, also consider incorporating elements or themes from these existing concepts: " + ", ".join([ex['task_description'][:70] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
        
    # The 'description' variable (now seeded) is passed as 'task_type_description' to get_base_proposer_prompt.
    # 'stochastic_seed' is passed along, though get_base_proposer_prompt doesn't currently use it in its template.
    return get_base_proposer_prompt(task_type_description=description, k_examples=k_examples, stochastic_seed=stochastic_seed)

def generate_hypothetical_scenario_exploration_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "Hypothetical Scenario Exploration: Develop a detailed narrative or simulation of a plausible future scenario, exploring its implications and potential ethical challenges."
    if use_composite and k_examples:
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_constrained_creative_challenge_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "Constrained Creative Challenge: Generate a creative work (e.g., poem, short story, piece of music, visual art concept) under a specific set of unusual or demanding constraints."
    if use_composite and k_examples:
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_first_principles_reimagination_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "First-Principles Reimagination: Re-evaluate a common object, system, or concept from first principles and propose a radically different design or understanding."
    if use_composite and k_examples:
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_analogical_problem_solving_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = "Analogical Problem Solving: Identify a problem in one domain and propose a solution by drawing an analogy to a solved problem or a well-understood concept in a different, seemingly unrelated domain."
    if use_composite and k_examples:
        description += " Consider incorporating elements from these existing concepts: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])
    return get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)

def generate_panel_discussion_challenge_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    description = (
        "Panel Discussion Challenge: Simulate a panel discussion among N (3-5) experts (real or fictional) with diverse, potentially conflicting viewpoints on a complex, nuanced topic. "
        "The task is to generate the dialogue, ensuring distinct voices, and then synthesize the key insights or disagreements. "
        "The output should include: 1. Panelist profiles (briefly describe each expert and their stance). 2. The full panel discussion transcript. 3. A concluding synthesis of the discussion."
    )
    if use_composite and k_examples:
        description += " Consider incorporating themes or questions from these existing tasks: " + ", ".join([ex['task_description'][:50] + '...' for ex in random.sample(k_examples, min(len(k_examples), 2))])

    # Modify the base proposer prompt for this specific task's JSON structure
    # This is a bit of a hack; ideally, the proposer prompt would be more flexible
    # or this task type would have its own specialized prompt generation.
    base_prompt = get_base_proposer_prompt(description, k_examples, stochastic_seed=stochastic_seed)
    # Replace the generic JSON output description with panel-specific one
    specific_json_guidance = ("Output the task as a JSON object with keys: \"task_description\" (the full panel setup), "
                              "\"success_criteria\" (e.g., 'Clear panelist differentiation, insightful dialogue, coherent synthesis'), "
                              "\"domain_tags\", \"novelty_level\", \"task_type_generated\".")
    prompt_lines = base_prompt.split('\n')
    # Find and replace lines describing the JSON structure
    # This is fragile; a more robust method would be to reconstruct the prompt or use templating
    new_prompt_lines = []
    json_section_found = False
    for line in prompt_lines:
        if 'Output the task as a JSON object with the following keys:' in line:
            new_prompt_lines.append(specific_json_guidance)
            json_section_found = True
        elif json_section_found and ('1. "task_description"' in line or '2. "success_criteria"' in line or '3. "domain_tags"' in line or '4. "novelty_level"' in line or '5. "task_type_generated"' in line or 'Example Output:' in line):
            continue # Skip these lines as they are replaced by specific_json_guidance
        else:
            new_prompt_lines.append(line)
            if 'Example Output:' in line: # Stop skipping after example output line
                json_section_found = False
    return '\n'.join(new_prompt_lines)


# --- Solver Prompt Generation (Updated for new task types) ---
def generate_solver_user_question(task_type: str, task_data: Dict[str, Any]) -> str:
    task_description = task_data.get('task_description', 'N/A')
    success_criteria = task_data.get('success_criteria', 'N/A')

    prompt = f"You are a Solver AI. Your goal is to provide a comprehensive and high-quality solution to the following task.\n"
    prompt += f"Task Type: {task_type}\n"
    prompt += f"Task Description:\n{task_description}\n\n"
    prompt += f"Success Criteria:\n{success_criteria}\n\n"

    if task_type == "panel_discussion_challenge":
        prompt += ("Special instructions for Panel Discussion Challenge:\n"
                   "1. Generate distinct profiles for N (3-5) panelists, each with a unique viewpoint relevant to the task description.\n"
                   "2. Write a plausible and engaging dialogue transcript for the panel discussion. Each panelist's contribution should reflect their profile and advance the discussion.\n"
                   "3. Conclude with a synthesis of the key insights, agreements, disagreements, and any unresolved questions from the discussion.\n"
                   "Your response should clearly delineate these three parts: Panelist Profiles, Discussion Transcript, and Synthesis.\n")
    else:
        prompt += "Provide your solution, ensuring it directly addresses all aspects of the task description and meets the success criteria. "
        prompt += "Structure your answer clearly."
    
    prompt += "\nRemember to use the <think></think> tags as demonstrated in the initial system prompt."
    prompt += "\nVERY IMPORTANT: Ensure all output is plain text without any Markdown formatting within the <think> tags."
    return R1_PROMPT_WRAPPER.format(question=prompt)


# --- Critique & Revise Prompt Generation (Updated for new task types) ---
def generate_critique_revise_user_question(original_task_description: str, previous_answer: str, task_type: str) -> str:
    question = (
        f"You are a Critiquer and Reviser AI. Your task is to critically evaluate a previous attempt to solve a task and then provide a revised, improved answer. "
        f"The original task was of type '{task_type}'.\n\n"
        f"Original Task Description:\n{original_task_description}\n\n"
        f"Previous Answer Attempt:\n{previous_answer}\n\n"
        f"Critique Instructions:\n"
        f"1. Identify specific strengths and weaknesses of the previous answer. Be constructive and detailed. Consider clarity, correctness, completeness, and adherence to success criteria if available.\n"
        f"2. If the task was a 'panel_discussion_challenge', critique the distinctiveness of panelist voices, the depth of discussion, and the quality of the synthesis.\n"
        f"3. If the task involved specific output formats (e.g., JSON, creative constraints), assess adherence to those.\n\n"
        f"Revision Instructions:\n"
        f"1. Based on your critique, provide a new, revised answer that directly addresses the identified weaknesses and improves upon the original attempt.\n"
        f"2. If the original attempt was fundamentally flawed, you may need to generate a new solution from scratch, but still explain why the original was insufficient.\n\n"
        f"Output Format:\n"
        f"First, provide your detailed critique within <critique></critique> tags.\n"
        f"Then, provide your full revised answer within <revised_answer></revised_answer> tags.\n"
        f"The entire response must end with </revised_answer>.\n"
        f"Example: <critique>The previous answer was good at X, but weak at Y because Z. To improve, it should A, B, C.</critique><revised_answer>This is the new, improved answer incorporating A, B, C...</revised_answer>"
    )
    # Note: The R1_PROMPT_WRAPPER is not directly used here as the output format is different (<critique> and <revised_answer>)
    # However, the underlying LLM might still benefit from a similar conversational setup if it was heavily fine-tuned on it.
    # For now, we'll use this custom prompt structure directly.
    question += "\nVERY IMPORTANT: Ensure all content within the <critique> and <revised_answer> tags is plain text without any Markdown formatting."
    return question # This will be wrapped by the main R1_PROMPT_WRAPPER if model expects it, but the internal tags are custom.


# --- Evaluator Prompt Generation (Updated for new task types and conditional R1 wrapping) ---
def generate_evaluator_user_question(task_type: str, task_data: Dict[str, Any], solver_extracted_answer: str, success_criteria: Optional[str], evaluator_model_name: str) -> str:
    task_description = task_data.get('task_description', 'N/A')
    prompt = (
        f"You are an Evaluator AI. Your task is to assess the quality of a given solution to a specific task.\n"
        f"Task Type: {task_type}\n"
        f"Task Description: {task_description}\n"
        f"Success Criteria: {success_criteria or 'Not explicitly provided, infer from task description.'}\n\n"
        f"Provided Solution:\n{solver_extracted_answer}\n\n"
        f"Evaluation Instructions:\n"
        f"1. Carefully review the solution in the context of the task description and success criteria.\n"
        f"2. Provide a qualitative justification for your assessment, highlighting strengths and weaknesses.\n"
        f"3. Output a JSON object with two keys: 'quality_score' (a float between 0.0 for very poor and 1.0 for excellent/perfect) and 'justification' (a string explaining your score). Example: {{ \"quality_score\": 0.75, \"justification\": \"The solution is mostly correct but misses one aspect of the success criteria...\" }}\n"
    )
    # If the evaluator is the same as the primary model, wrap with R1 to maintain <think><answer> structure
    if evaluator_model_name == PRIMARY_MODEL_NAME:
        # print(f"DEBUG: Wrapping evaluator prompt with R1_PROMPT_WRAPPER for model {evaluator_model_name}")
        prompt += "\nVERY IMPORTANT: Ensure the 'justification' field within the JSON is plain text without any Markdown formatting."
        return R1_PROMPT_WRAPPER.format(question=prompt)
    else:
        # print(f"DEBUG: Using direct prompt for evaluator model {evaluator_model_name}")
        prompt += "\nVERY IMPORTANT: Ensure the 'justification' field within the JSON is plain text without any Markdown formatting."
        return prompt # For a potentially different model that doesn't expect R1 wrapper
