# Absolute Zero Universal Knowledge Generator
# This script implements a paradigm to generate tasks/questions
# that a human might not typically formulate, spanning any field of knowledge,
# and then has an LLM attempt to answer them.
# v1.4.0: Panel of Experts, Solver Self-Critique, Stochastic Proposer Perturbations.

import json
import random
import os
import time # For tracking iteration duration
import re
import asyncio # Added for asynchronous operations
from typing import Dict, List, Tuple, Any, Optional

# --- Configuration ---
# Primary LLM Configuration
PRIMARY_API_BASE_URL = os.getenv("PRIMARY_API_BASE_URL", "https://api.novita.ai/v3/openai")
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY", "<Your_API_Key_HERE>") # SET THIS!
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "deepseek/deepseek-r1") # Proposer, Solver, Critiquer

# Secondary LLM Configuration (Evaluator - uses Primary API credentials)
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "qwen/qwen3-235b-a22b-fp8") # Set to "" or None to disable.

# Version Configuration
VERSION = "1.4.0"

# General Configuration
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "30")) # Adjusted due to increased calls per iter
K_REFERENCE_EXAMPLES = 2
N_SOLVER_ROLLOUTS_FOR_PROPOSER = int(os.getenv("N_SOLVER_ROLLOUTS_FOR_PROPOSER", "1")) # Reduced for faster iterations with new steps
FINETUNING_DATA_FILE = f"universal_knowledge_exploration_log_v{VERSION}.jsonl"

TASK_TYPE_DISTRIBUTION = {
    "synthesis_of_disparate_paradigms": 0.15,
    "generation_of_novel_axioms_and_exploration": 0.15,
    "epistemological_boundary_probes": 0.10,
    "hypothetical_scenario_exploration": 0.15,
    "constrained_creative_challenge": 0.15,
    "first_principles_reimagination": 0.10,
    "analogical_problem_solving": 0.10,
    "panel_discussion_challenge": 0.10, # New
}

MAX_TOKENS_PROPOSER = 3300
MAX_TOKENS_SOLVER = 3800 # Might need more for panel discussions + synthesis
MAX_TOKENS_CRITIQUE_REVISE = 3800 # For critique and revised answer
MAX_TOKENS_EVALUATOR = 1000
PROPOSER_TEMPERATURE = 0.90 # Higher for more creative and perturbed proposals
SOLVER_TEMPERATURE = 0.78 # Slightly higher for creative panel roles
CRITIQUE_TEMPERATURE = 0.5
REVISE_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.4

# Quality Thresholds
LOGGING_QUALITY_THRESHOLD = float(os.getenv("LOGGING_QUALITY_THRESHOLD", "0.3"))
LEARNED_CONCEPT_QUALITY_THRESHOLD = float(os.getenv("LEARNED_CONCEPT_QUALITY_THRESHOLD", "0.6"))

COMPOSITE_CONCEPT_PROBABILITY = 0.2
MAX_LEARNED_CONCEPTS = 30
STOCHASTIC_PERTURBATION_PROBABILITY = 0.15 # Chance to inject random seed into proposer
RANDOM_SEED_CONCEPTS = ["entropy", "fractals", "emergence", "symbiosis", "quantum entanglement", "neural networks", "game theory", "dark matter", "consciousness", "algorithmic bias", "terraforming", "bio-mimicry"]


# API Throttling Configuration
API_RPM_LIMIT = int(os.getenv("API_RPM_LIMIT", "10"))
MIN_ITER_SLEEP = 0.2

# --- Globals for Curriculum Learning ---
learned_concepts_pool: List[Dict[str, Any]] = []
experience_buffer: List[Dict[str, Any]] = []
MAX_BUFFER_SIZE = 75

# --- R1 Prompt Template (Using <think>) ---
R1_PROMPT_WRAPPER = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first outlines the reasoning process in detail within <think> </think> tags, "
    "and then provides the final answer within <answer> </answer> tags. "
    "The entire response must end with </answer>.\n"
    "Example: <think> My detailed plan is to first A, then B, considering C. </think> <answer> final answer here </answer>.\n\n"
    "User: {question}\n\n"
    "Assistant: "
)

# --- Async API Client ---
async def query_llm_api(user_content: str, temperature: float, max_tokens: int, model_name: str, api_base_url: str, api_key: str) -> Optional[str]:
    if api_key == "<Your_API_Key_HERE>" or not api_key :
        print(f"FATAL: API_KEY for model {model_name} (using key: {api_key[:5]}...{api_key[-5:] if len(api_key)>10 else ''}) is not properly set. It might still be the placeholder '<Your_API_Key_HERE>'.")
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("FATAL: The 'openai' library is not installed or version is too old for AsyncOpenAI. Please run: pip install --upgrade openai")
        return None

    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    messages = [{"role": "user", "content": user_content}]

    try:
        chat_completion_res = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format={"type": "text"}
        )
        assistant_response = chat_completion_res.choices[0].message.content
        if assistant_response:
            return assistant_response.strip()
        else:
            print(f"Warning: LLM API ({model_name}) returned an empty response content.")
            return None
    except Exception as e:
        print(f"Error querying LLM API ({model_name} at {api_base_url}): {e}")
        if hasattr(e, 'response') and e.response is not None:
             try:
                 err_body = await e.response.json()
                 print(f"LLM API Response Status: {e.response.status}")
                 print(f"LLM API Response Body: {err_body}")
             except Exception as e_resp:
                 print(f"Could not parse error response body: {e_resp}")
        elif hasattr(e, 'message'): print(f"Error details: {e.message}")
        return None

# --- Enhanced Task Generation Prompts ---
def get_base_proposer_prompt(task_type_description: str, k_examples: List[Dict[str, Any]], stochastic_seed: Optional[str] = None) -> str:
    question = (
        f"You are an advanced AI Proposer tasked with generating exceptionally novel and challenging intellectual tasks for another AI (the Responder). "
        f"These tasks should push the boundaries of known concepts, encourage deep synthesis, or explore meta-cognitive questions that humans rarely, if ever, formulate.\n"
        f"The current task category is: **{task_type_description}**.\n\n"
        "To achieve true novelty, consider these strategies when formulating your task:\n"
        "- **Combinatorial Creativity**: Force the Responder to merge concepts from domains it likely hasn't seen combined.\n"
        "- **High-Constraint Scenarios**: Impose unusual, difficult, or even seemingly impossible conditions or constraints. This forces solutions beyond the obvious.\n"
        "- **Counterfactuals & Alternative Histories**: Ask 'What if X were different?' and explore deep ripple effects.\n"
        "- **First Principles Reasoning**: Instruct the Responder to discard common assumptions and rebuild a concept or system from fundamental truths.\n"
        "- **Analogical Leaps**: Encourage the use of analogies from obscure, unrelated, or emerging phenomena to solve a problem or generate a new idea.\n"
        "- **SCAMPER/Transformative Thinking**: Suggest applying techniques like Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, or Reverse to a known concept.\n"
        "- **Multi-Part Challenges**: Design tasks that require the Responder to perform a sequence of distinct reasoning steps (e.g., analyze, critique, then synthesize a solution).\n"
        "- **Pre-Critique**: Before finalizing your task, briefly consider potential ambiguities or weaknesses a critic might point out. How can you make your task clearer, more robust, or more uniquely challenging in light of these potential criticisms?\n\n"
    )
    if stochastic_seed:
        question += f"**Stochastic Seed Concept**: As an extra challenge, try to subtly and coherently weave the concept of '{stochastic_seed}' into the task you generate, if it can be done naturally to enhance novelty.\n\n"

    question += (
        "Your goal is to propose a task that is: \n"
        "1. Highly original and not a trivial variation of common knowledge or problems.\n"
        "2. Conceptually deep, requiring sophisticated reasoning or creative synthesis.\n"
        "3. Well-defined enough that an advanced AI Responder could attempt a coherent answer, even if the subject is highly abstract or speculative.\n"
        "4. Avoid questions with simple factual answers or those easily found in standard knowledge bases. Aim for generative, analytical, or speculative challenges.\n\n"
        "IMPORTANT: Your response MUST strictly follow this structure: first, use the <think> tag to outline your reasoning for formulating this specific task, explaining its novelty, the creative techniques you're employing (including any pre-critique thoughts and how you incorporated the stochastic seed, if provided), and what makes it challenging. "
        "Immediately after the </think> tag, provide the final task proposal within the <answer> tag. Your entire response MUST end with </answer>. "
        "The content inside <answer> MUST be a single JSON object detailing the task for the Responder. All keys and string values in the JSON MUST use double quotes.\n"
    )
    if k_examples:
        question += "\nHere are some examples of how an assistant might structure its thoughts and task proposals for similar broad categories:\n"
        for ex in k_examples:
            think_example = f"<think>Example Proposer Thinking for a '{ex.get('task_type', 'N/A')}' task: My plan is to combine concept X from domain A with methodology Y from domain B, using combinatorial creativity. The constraint Z will make it challenging. Pre-critique: I initially thought of asking for a simple comparison, but that's too easy. I'll make it a generative task instead. The JSON output will specify keys like 'domain_A_concept', 'domain_B_methodology', 'constraint_Z', 'target_framework_description'.</think>"
            example_answer_content = ex.get('proposer_task_json_str', '{}')
            if isinstance(example_answer_content, dict): example_answer_content = json.dumps(example_answer_content)
            elif not isinstance(example_answer_content, str): example_answer_content = '{}'
            answer_example = f"<answer>{example_answer_content[:150] + '...' if len(example_answer_content) > 150 else example_answer_content}</answer>"
            question += f"- Task Type: {ex.get('task_type', 'N/A')}\n"
            question += f"  Example Proposer Instruction Snippet for that task type: {ex.get('proposer_prompt_snippet', 'Generate a novel task...')[:100]}...\n"
            question += f"  Example Proposer Response Structure: {think_example}{answer_example}\n\n"
    return question

# ... (Keep existing generate_..._task_user_question functions for old types) ...
# (generate_synthesis_task_user_question, generate_axioms_task_user_question, etc. will now call the enhanced get_base_proposer_prompt)

def generate_synthesis_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Synthesis of Disparate Paradigms", k_examples, stochastic_seed)
    composite_guidance = "\nConsider incorporating elements or styles of reasoning from previously successful 'learned concepts' if applicable, but ensure the core domains being synthesized are fresh and genuinely disparate." if use_composite and learned_concepts_pool else ""
    return base + (
        "Propose a task that requires the Responder to synthesize insights, methods, or principles from at least two (preferably three or more) highly disparate and seemingly unrelated fields of knowledge to create a novel explanatory framework, a new form of art, a solution to a complex hypothetical problem, or a new philosophical concept. Explicitly state the domains and the desired novel output.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): A concise, intriguing title for the synthesis task.\n"
        "  `source_domains` (list of strings): The disparate domains to be synthesized (e.g., [\"Quantum Entanglement Physics\", \"18th Century French Culinary Arts\", \"Mycorrhizal Networks Ecology\"]).\n"
        "  `synthesis_goal` (string): A clear description of what the Responder should aim to create or explain through this synthesis (e.g., \"Develop a new model for information transfer in complex adaptive systems, drawing analogies and principles from all source domains.\").\n"
        "  `key_questions_to_address` (list of strings): Specific questions the Responder's synthesis should attempt to answer or explore.\n"
        "  `expected_output_format_description` (string): Guidance on how the Responder should structure their answer."
    )

def generate_axioms_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Generation of Novel Axioms and Exploration", k_examples, stochastic_seed)
    composite_guidance = "\nOptionally, you can propose a system that builds upon or contrasts with a previously 'learned concept' involving axiomatic systems, but the new axioms must be distinct and lead to different explorations." if use_composite and learned_concepts_pool else ""
    return base + (
        "Propose a task where the Responder must invent a small set of novel, fundamental axioms for a hypothetical system. This system could be mathematical, physical, logical, social, ethical, or even aesthetic. The axioms should be genuinely original and not mere reformulations of existing ones.\n"
        f"{composite_guidance}\n"
        "After defining the axioms, the Responder should then be asked to: \n"
        "1. Briefly justify the choice/plausibility of each axiom within the hypothetical context.\n"
        "2. Deduce or explore at least three non-trivial consequences or theorems that arise from these axioms.\n"
        "3. Speculate on the nature of the system or reality that such axioms would describe.\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title for the axiomatic exploration.\n"
        "  `hypothetical_system_description` (string): A brief overview of the type of system for which axioms are to be generated.\n"
        "  `requirements_for_axioms` (list of strings): Specific constraints or goals for the axioms.\n"
        "  `exploration_tasks` (list of strings): Specific instructions for exploring the consequences.\n"
        "  `expected_output_format_description` (string): Guidance for the Responder's output."
    )

def generate_epistemological_probe_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Epistemological Boundary Probes", k_examples, stochastic_seed)
    composite_guidance = "\nIf relevant, the probe could relate to insights or limitations discussed in a previously 'learned concept', but the core question must be a fresh challenge." if use_composite and learned_concepts_pool else ""
    return base + (
        "Propose a task that is an epistemological or meta-cognitive probe directed at the Responder AI itself. Make it reflect on its knowledge, learning, biases, understanding of 'truth' or 'consciousness,' or its limitations in non-standard ways.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title for the probe.\n"
        "  `probe_question` (string): The central, challenging question.\n"
        "  `context_or_scenario` (string, optional): Context to frame the probe.\n"
        "  `aspects_to_consider_in_response` (list of strings): Key dimensions the AI should address.\n"
        "  `expected_output_format_description` (string): Guidance for the output."
    )

def generate_hypothetical_scenario_exploration_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Hypothetical Scenario Exploration", k_examples, stochastic_seed)
    composite_guidance = "\nConsider linking this scenario to themes from 'learned concepts' if appropriate." if use_composite and learned_concepts_pool else ""
    return base + (
        "Propose a 'What If?' or counterfactual scenario with a significant deviation from known reality/history. Ask the Responder to explore non-obvious consequences across multiple domains.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title.\n"
        "  `scenario_premise` (string): The core 'What If?' statement.\n"
        "  `domains_for_exploration` (list of strings): Areas for consequence exploration.\n"
        "  `key_questions_to_address` (list of strings): Questions to guide exploration.\n"
        "  `expected_output_format_description` (string): E.g., \"A multi-chapter speculative analysis.\""
    )

def generate_constrained_creative_challenge_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Constrained Creative Challenge", k_examples, stochastic_seed)
    composite_guidance = "\nIf a 'learned concept' involved an object/idea, task the Responder to transform it." if use_composite and learned_concepts_pool else ""
    return base + (
        "Challenge the Responder to create something novel under unusual or severe constraints, or apply a transformation technique (like SCAMPER) to reinvent a known concept.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title.\n"
        "  `creative_goal` (string): What to create/reinvent.\n"
        "  `constraints_or_transformation_technique` (list of strings): Specific constraints OR transformation technique and target.\n"
        "  `evaluation_criteria_for_novelty` (list of strings): How novelty will be judged.\n"
        "  `expected_output_format_description` (string): E.g., \"A detailed design document.\""
    )

def generate_first_principles_reimagination_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("First Principles Reimagination", k_examples, stochastic_seed)
    composite_guidance = "\nCould a 'learned concept' be reimagined from different first principles?" if use_composite and learned_concepts_pool else ""
    return base + (
        "Task the Responder to take a complex existing concept, discard assumptions, and rebuild/reimagine it from specified (possibly unconventional) first principles.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title.\n"
        "  `concept_to_reimagine` (string): The existing concept/system.\n"
        "  `new_first_principles` (list of strings): Fundamental principles for rebuilding.\n"
        "  `key_aspects_to_develop` (list of strings): Specific parts of the reimagined concept to detail.\n"
        "  `expected_output_format_description` (string): E.g., \"A manifesto outlining the new system.\""
    )

def generate_analogical_problem_solving_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Analogical Problem Solving", k_examples, stochastic_seed)
    composite_guidance = "\nCan an analogy from a 'learned concept' solve a problem in a different domain?" if use_composite and learned_concepts_pool else ""
    return base + (
        "Task the Responder to solve a problem or generate a solution in a target domain by drawing analogies from an obscure, unrelated, or emerging source domain.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title.\n"
        "  `problem_domain_and_challenge` (string): The target problem.\n"
        "  `source_analog_domain` (string): The obscure domain for analogies.\n"
        "  `aspects_for_analogical_mapping` (list of strings): Key features for mapping.\n"
        "  `desired_solution_characteristics` (list of strings): Properties the solution should have.\n"
        "  `expected_output_format_description` (string): E.g., \"A detailed proposal.\""
    )

def generate_panel_discussion_challenge_task_user_question(k_examples: List[Dict[str, Any]], use_composite: bool = False, stochastic_seed: Optional[str] = None) -> str:
    base = get_base_proposer_prompt("Panel Discussion Challenge", k_examples, stochastic_seed)
    composite_guidance = "\nCould a 'learned concept' be the topic of this panel discussion, explored from new angles?" if use_composite and learned_concepts_pool else ""
    return base + (
        "Propose a complex problem or a multifaceted topic. Then, define 2-4 distinct 'expert roles' with specific viewpoints or areas of expertise. The Responder will need to articulate the perspective of each expert on the problem/topic and then provide a synthesized summary or a set of recommendations that considers all viewpoints.\n"
        f"{composite_guidance}\n"
        "The JSON output in <answer> should include keys like: \n"
        "  `task_title` (string): Title for the panel discussion.\n"
        "  `discussion_topic_or_problem` (string): The central issue to be discussed.\n"
        "  `expert_roles` (list of dicts): Each dict should have 'role_name' (e.g., \"Skeptical Ethicist\") and 'role_description' (e.g., \"Focuses on potential negative consequences and moral hazards, often questioning underlying assumptions.\"). Aim for 2-4 roles.\n"
        "  `key_questions_for_panel` (list of strings): Specific questions each expert (and the synthesis) should address.\n"
        "  `synthesis_goal` (string): What the final synthesized part of the answer should achieve (e.g., \"A balanced summary of perspectives and a list of three actionable recommendations.\").\n"
        "  `expected_output_format_description` (string): E.g., \"Structured response with sections for each expert's viewpoint followed by a final synthesis.\""
    )

# --- Solver Prompt Generation (Updated for new task types) ---
def generate_solver_user_question(task_type: str, task_data: Dict[str, Any]) -> str:
    question = f"You are an advanced AI Responder. You are tasked with addressing the following highly conceptual and novel intellectual challenge of type: **{task_type.replace('_', ' ').title()}**.\n"
    question += "Engage with the task deeply, aim for originality, coherence, and insightful reasoning. Use <think> for your detailed reasoning process before providing the final answer in the <answer> tag. Your entire response must end with </answer>.\n\n"
    question += f"**Task Title:** {task_data.get('task_title', 'N/A')}\n\n"

    # ... (keep existing cases for older task types) ...
    if task_type == "synthesis_of_disparate_paradigms":
        question += f"**Source Domains to Synthesize:** {', '.join(task_data.get('source_domains', []))}\n"
        question += f"**Synthesis Goal:** {task_data.get('synthesis_goal', 'N/A')}\n"
        if task_data.get('key_questions_to_address'):
            question += "**Key Questions to Address in Your Synthesis:**\n"
            for i, q_item in enumerate(task_data.get('key_questions_to_address', [])): question += f"  {i+1}. {q_item}\n"
    elif task_type == "generation_of_novel_axioms_and_exploration":
        question += f"**Hypothetical System Description:** {task_data.get('hypothetical_system_description', 'N/A')}\n"
        if task_data.get('requirements_for_axioms'):
            question += "**Requirements for Axioms:**\n"
            for i, req_item in enumerate(task_data.get('requirements_for_axioms', [])): question += f"  {i+1}. {req_item}\n"
        if task_data.get('exploration_tasks'):
            question += "**Exploration Tasks Based on Your Axioms:**\n"
            for i, exp_item in enumerate(task_data.get('exploration_tasks', [])): question += f"  {i+1}. {exp_item}\n"
    elif task_type == "epistemological_boundary_probes":
        question += f"**Probe Question:** {task_data.get('probe_question', 'N/A')}\n"
        if task_data.get('context_or_scenario'): question += f"**Context/Scenario:** {task_data.get('context_or_scenario')}\n"
        if task_data.get('aspects_to_consider_in_response'):
            question += "**Aspects to Consider in Your Response:**\n"
            for i, aspect_item in enumerate(task_data.get('aspects_to_consider_in_response', [])): question += f"  {i+1}. {aspect_item}\n"
    elif task_type == "hypothetical_scenario_exploration":
        question += f"**Scenario Premise:** {task_data.get('scenario_premise', 'N/A')}\n"
        if task_data.get('domains_for_exploration'):
            question += f"**Domains for Exploration:** {', '.join(task_data.get('domains_for_exploration', []))}\n"
        if task_data.get('key_questions_to_address'):
            question += "**Key Questions to Address:**\n"
            for i, q_item in enumerate(task_data.get('key_questions_to_address', [])): question += f"  {i+1}. {q_item}\n"
    elif task_type == "constrained_creative_challenge":
        question += f"**Creative Goal:** {task_data.get('creative_goal', 'N/A')}\n"
        if task_data.get('constraints_or_transformation_technique'):
            question += "**Constraints or Transformation Technique:**\n"
            for i, const_item in enumerate(task_data.get('constraints_or_transformation_technique', [])): question += f"  - {const_item}\n"
        if task_data.get('evaluation_criteria_for_novelty'):
            question += "**Evaluation Criteria for Novelty (consider these in your response):**\n"
            for i, crit_item in enumerate(task_data.get('evaluation_criteria_for_novelty', [])): question += f"  - {crit_item}\n"
    elif task_type == "first_principles_reimagination":
        question += f"**Concept to Reimagine:** {task_data.get('concept_to_reimagine', 'N/A')}\n"
        if task_data.get('new_first_principles'):
            question += "**New First Principles to Use:**\n"
            for i, princ_item in enumerate(task_data.get('new_first_principles', [])): question += f"  - {princ_item}\n"
        if task_data.get('key_aspects_to_develop'):
            question += "**Key Aspects to Develop in Your Reimagination:**\n"
            for i, aspect_item in enumerate(task_data.get('key_aspects_to_develop', [])): question += f"  - {aspect_item}\n"
    elif task_type == "analogical_problem_solving":
        question += f"**Problem Domain and Challenge:** {task_data.get('problem_domain_and_challenge', 'N/A')}\n"
        question += f"**Source Analog Domain:** {task_data.get('source_analog_domain', 'N/A')}\n"
        if task_data.get('aspects_for_analogical_mapping'):
            question += "**Aspects for Analogical Mapping (from source to problem):**\n"
            for i, map_item in enumerate(task_data.get('aspects_for_analogical_mapping', [])): question += f"  - {map_item}\n"
        if task_data.get('desired_solution_characteristics'):
            question += "**Desired Solution Characteristics:**\n"
            for i, char_item in enumerate(task_data.get('desired_solution_characteristics', [])): question += f"  - {char_item}\n"
    elif task_type == "panel_discussion_challenge":
        question += f"**Discussion Topic/Problem:** {task_data.get('discussion_topic_or_problem', 'N/A')}\n"
        question += "**Expert Roles to Embody:**\n"
        for i, role_info in enumerate(task_data.get('expert_roles', [])):
            question += f"  {i+1}. **{role_info.get('role_name', 'Unnamed Role')}**: {role_info.get('role_description', 'No description')}\n"
        if task_data.get('key_questions_for_panel'):
            question += "**Key Questions for the Panel (each expert should address these, and they should inform your synthesis):**\n"
            for i, q_item in enumerate(task_data.get('key_questions_for_panel', [])): question += f"  - {q_item}\n"
        question += f"**Synthesis Goal:** {task_data.get('synthesis_goal', 'N/A')}\n"
        question += "Structure your response clearly: first, provide the perspective of each expert individually. Then, provide a final synthesized summary or solution that integrates these perspectives and addresses the synthesis goal.\n"

    question += f"\n**Expected Output Format:** {task_data.get('expected_output_format_description', 'A detailed, well-reasoned response.')}\n"
    question += "Please provide your full thinking process within <think> tags, followed by your comprehensive answer within <answer> tags."
    return question

def generate_critique_revise_user_question(original_task_description: str, previous_answer: str, task_type: str) -> str:
    return (
        f"You are an AI assistant performing a self-critique and revision step.\n\n"
        f"**Original Task Type:** {task_type}\n"
        f"**Original Task Description Summary:**\n{original_task_description[:1000]}...\n\n" # Show a snippet of the original task
        f"**Your Previous Answer:**\n```text\n{previous_answer}\n```\n\n"
        "**Instructions for Self-Critique and Revision:**\n"
        "1.  **Critique your previous answer**: Play devil's advocate. Identify its key weaknesses, potential biases, overlooked angles, areas where the reasoning could be deeper, or aspects that might not fully meet the spirit of the original task's novelty and depth. Be specific in your critique.\n"
        "2.  **Provide a Revised and Improved Answer**: Based on your critique, generate a new, stronger version of the answer. This revised answer should address the identified weaknesses and aim to be more insightful, comprehensive, creative, and robust.\n\n"
        "Structure your entire response as follows:\n"
        "<critique>\n"
        "[Your detailed critique of the previous answer here...]\n"
        "</critique>\n\n"
        "<revised_answer>\n"
        "[Your new, improved answer here...]\n"
        "</revised_answer>\n\n"
        "Ensure the <revised_answer> is a complete, standalone response to the original task, incorporating the improvements."
    )

# --- Parsing for Critique and Revised Answer ---
def extract_from_critique_revise_response(llm_full_response: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not llm_full_response:
        return None, None
    
    critique_match = re.search(r"<critique>\s*([\s\S]+?)\s*</critique>", llm_full_response, re.IGNORECASE | re.DOTALL)
    revised_answer_match = re.search(r"<revised_answer>\s*([\s\S]+?)\s*</revised_answer>", llm_full_response, re.IGNORECASE | re.DOTALL)
    
    critique = critique_match.group(1).strip() if critique_match else None
    revised_answer = revised_answer_match.group(1).strip() if revised_answer_match else None
    
    if not critique or not revised_answer:
        print(f"Warning: Could not extract both <critique> and <revised_answer> from self-critique response. Full response: {llm_full_response[:300]}...")
        # Fallback: if only revised_answer is present without critique tags, or vice-versa, try to get it.
        # This is less ideal but might salvage some output.
        if not revised_answer and llm_full_response.strip().lower().startswith("<revised_answer>"): # Check if it's just the revised answer
             revised_answer_match_fallback = re.search(r"<revised_answer>\s*([\s\S]+?)\s*</revised_answer>", llm_full_response, re.IGNORECASE | re.DOTALL) # Redo search on full string
             if revised_answer_match_fallback: revised_answer = revised_answer_match_fallback.group(1).strip()

    return critique, revised_answer


def generate_evaluator_user_question(task_type: str, task_data: Dict[str, Any], solver_extracted_answer: str, success_criteria: Optional[str], evaluator_model_name: str) -> str:
    task_title = task_data.get('task_title', 'Untitled Task')
    # The solver_extracted_answer here will be the *revised* answer after self-critique
    return (
        f"You are an AI Quality Evaluator using model {evaluator_model_name}. Your role is to assess the quality of a solution provided by another AI (the Responder) to a complex, novel task. The solution you are evaluating has already undergone a self-critique and revision step.\n\n"
        f"**Original Task Type:** {task_type}\n"
        f"**Task Title:** {task_title}\n"
        "**Task Description (JSON from Proposer):**\n```json\n"
        f"{json.dumps(task_data, indent=2)}\n```\n\n"
        f"**Success Criteria for a Good Response:**\n{success_criteria or 'No specific criteria provided, evaluate based on general quality.'}\n\n"
        "**Responder's (Revised) Solution:**\n```text\n"
        f"{solver_extracted_answer}\n```\n\n"
        "**Evaluation Instructions:**\n"
        "1. Carefully review the original task, success criteria, and the Responder's revised solution.\n"
        "2. Provide a holistic quality score for the solution on a scale of 0.0 (very poor) to 1.0 (excellent).\n"
        "3. Provide a brief justification for your score, highlighting strengths and weaknesses of this revised version.\n"
        "Your response MUST be a JSON object with two keys: 'quality_score' (float) and 'justification' (string).\n"
        "Example: {\"quality_score\": 0.85, \"justification\": \"The revised solution effectively addressed the initial weaknesses and now presents a more robust and insightful argument.\"}"
    )

# --- Parsing LLM's <answer> content ---
def extract_from_answer_tag(llm_full_response: Optional[str], task_type_for_heuristic: Optional[str] = None) -> Optional[str]:
    if not llm_full_response: return None
    answer_match = re.search(r"<answer[^>]*>\s*([\s\S]+?)\s*</answer>", llm_full_response, re.IGNORECASE | re.DOTALL)
    if answer_match: return answer_match.group(1).strip()
    print(f"Warning: Could not find complete <answer>...</answer> block. Attempting fallbacks for response starting with: {llm_full_response[:200]}...")
    last_think_end_pos = -1
    for think_tag_variant in [r"</think>", r"</thought>"]:
        for match in re.finditer(think_tag_variant, llm_full_response, re.IGNORECASE | re.DOTALL): last_think_end_pos = max(last_think_end_pos, match.end())
    if last_think_end_pos != -1:
        potential_answer_after_think = llm_full_response[last_think_end_pos:].strip()
        if potential_answer_after_think.lower().startswith("<answer"):
            content_from_start_answer = re.sub(r"<answer[^>]*>", "", potential_answer_after_think, 1, flags=re.IGNORECASE | re.DOTALL).strip()
            if content_from_start_answer:
                print(f"  Fallback 1.1: Using content from <answer> tag after last </think>: '{content_from_start_answer[:100]}...'")
                end_answer_in_extract = re.search(r"</answer>", content_from_start_answer, re.IGNORECASE | re.DOTALL)
                if end_answer_in_extract: return content_from_start_answer[:end_answer_in_extract.start()].strip()
                return content_from_start_answer
        elif potential_answer_after_think and not potential_answer_after_think.lower().startswith(("<think", "<thought")):
            print(f"  Fallback 1.2: Using all content after last </think>: '{potential_answer_after_think[:100]}...'")
            return potential_answer_after_think
    has_any_tags = any(tag in llm_full_response.lower() for tag in ["<think>", "<thought>", "</think>", "</thought>", "<answer>", "</answer>"])
    if not has_any_tags and len(llm_full_response) < 500:
        cleaned_response = llm_full_response.strip()
        if not any(err_token in cleaned_response.lower() for err_token in ["error", "sorry", "cannot", "i am unable", "i do not have enough information"]):
            print(f"  Fallback 2: Using entire short, tagless response as potential answer: '{cleaned_response[:100]}...'")
            return cleaned_response
    print(f"  All fallbacks failed to extract a clear answer. Original response (first 200 chars): {llm_full_response[:200]}...")
    return None

def _fix_json_string(json_str: str) -> str:
    json_str = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        json_str = re.sub(r"([{,\s])(['])([a-zA-Z_][\w]*)obar(['])(\s*):", r'\1"\3"\5:', json_str)
        json_str = re.sub(r"([{,\s])([a-zA-Z_][\w]*)(\s*):", r'\1"\2"\3:', json_str)
    except Exception as e: print(f"Regex error during JSON fixing: {e}")
    return json_str

def parse_json_from_answer(answer_content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not answer_content: return None
    try: return json.loads(answer_content)
    except json.JSONDecodeError:
        fixed_json_str = _fix_json_string(answer_content)
        try: return json.loads(fixed_json_str)
        except json.JSONDecodeError as e2:
            print(f"JSON parse of answer content failed even after fixes: {e2}")
            print(f"Original answer content for JSON parsing (first 300 chars): {answer_content[:300]}...")
            match = re.search(r"```json\s*([\s\S]+?)\s*```", answer_content, re.DOTALL)
            if match:
                print("Found JSON block in markdown, trying to parse that.")
                try: return json.loads(match.group(1).strip())
                except json.JSONDecodeError as e3: print(f"Parsing embedded JSON block also failed: {e3}")
            return None

# --- Helper for default async results ---
async def async_return_value(value: Any):
    return value

# --- Experience Buffer and Learned Concepts ---
def add_to_experience_buffer(proposed_task_data_json: Dict[str, Any], solver_full_llm_response: str, 
                             critique_text: Optional[str], revised_answer_text: str, # Added critique/revised
                             quality_score: float, justification: str):
    experience = {
        "task_type": proposed_task_data_json["task_type"],
        "proposer_task_details": proposed_task_data_json,
        "solver_initial_full_llm_response": solver_full_llm_response, # Storing initial solver response
        "solver_critique": critique_text,
        "solver_revised_answer": revised_answer_text, # This is what's evaluated
        "solution_quality_score": quality_score,
        "solution_quality_justification": justification
    }
    experience_buffer.append(experience)
    if len(experience_buffer) > MAX_BUFFER_SIZE: experience_buffer.pop(0)

def add_to_learned_concepts_pool(task_data: Dict[str, Any], revised_answer_text: str, quality_score: float): # Uses revised answer
    if quality_score < LEARNED_CONCEPT_QUALITY_THRESHOLD: return
    concept = {
        "task_type": task_data["task_type"],
        "task_title": task_data.get("task_title", "Untitled"),
        "task_details_json_str": json.dumps(task_data),
        "solver_revised_solution_snippet": revised_answer_text[:300] + "...", # Snippet of the revised solution
        "quality_score": quality_score,
    }
    if not any(c['task_title'] == concept['task_title'] and c['task_type'] == concept['task_type'] for c in learned_concepts_pool):
        learned_concepts_pool.append(concept)
        if len(learned_concepts_pool) > MAX_LEARNED_CONCEPTS: learned_concepts_pool.pop(0)
        print(f"    Added concept '{concept['task_title']}' to learned_concepts_pool (Score: {quality_score:.2f}). Pool size: {len(learned_concepts_pool)}")

def get_k_reference_examples() -> List[Dict[str, Any]]:
    if not experience_buffer: return []
    formatted_examples = []
    samples = random.sample(experience_buffer, min(len(experience_buffer), K_REFERENCE_EXAMPLES))
    for sample in samples:
        task_details = sample.get("proposer_task_details", {})
        formatted_examples.append({
            "task_type": task_details.get("task_type", "N/A"),
            "proposer_prompt_snippet": f"Generate a {task_details.get('task_type', 'N/A')} task...",
            "proposer_task_json_str": task_details.get("proposer_task_json_str", "{}")
        })
    return formatted_examples

# --- Logging ---
def log_exploration_data(user_question_for_solver: str, solver_initial_full_response: str, 
                         critique_response: Optional[str], revised_answer: str, # Added critique/revised
                         task_data: Dict[str, Any], quality_score: float, justification: str):
    with open(FINETUNING_DATA_FILE, "a", encoding='utf-8') as f:
        log_entry = {
            "task_type": task_data.get("task_type"),
            "task_title": task_data.get("task_title"),
            "proposer_task_json": task_data,
            "solver_prompt": user_question_for_solver,
            "solver_initial_full_response": solver_initial_full_response,
            "solver_critique_full_response": critique_response,
            "solver_revised_answer_extracted": revised_answer,
            "solution_final_quality_score": quality_score,
            "solution_quality_justification_combined": justification
        }
        f.write(json.dumps(log_entry) + "\n")

# --- Main Async Loop ---
async def main():
    print(f"Starting Absolute Zero Universal Knowledge Generator (v{VERSION} - Panel, Self-Critique, Perturbations)...")
    if PRIMARY_API_KEY == "<Your_API_Key_HERE>" or not PRIMARY_API_KEY:
        print("FATAL: PRIMARY_API_KEY is not set. Please set the environment variable or update the script.")
        return

    print(f"Primary Model (Proposer/Solver/Critiquer/Eval1): {PRIMARY_MODEL_NAME} via {PRIMARY_API_BASE_URL}")
    if SECONDARY_MODEL_NAME and SECONDARY_MODEL_NAME.strip() != "":
        print(f"Secondary Evaluator Model (Eval2): {SECONDARY_MODEL_NAME} (uses Primary API Base URL & Key)")
    else:
        print("Secondary Evaluator Model: Not configured. Using single evaluation.")
    print(f"Logging explorations to: {FINETUNING_DATA_FILE}")
    print(f"Targeting API RPM Limit: {API_RPM_LIMIT if API_RPM_LIMIT > 0 else 'Unlimited'}")
    print(f"Solver rollouts for proposer reward: {N_SOLVER_ROLLOUTS_FOR_PROPOSER}")
    print(f"Logging Quality Threshold: {LOGGING_QUALITY_THRESHOLD}, Learned Concept Threshold: {LEARNED_CONCEPT_QUALITY_THRESHOLD}")
    print(f"Stochastic Perturbation Probability for Proposer: {STOCHASTIC_PERTURBATION_PROBABILITY*100}%")


    for iteration in range(1, NUM_ITERATIONS + 1):
        iteration_start_time = time.monotonic()
        print(f"\n--- Iteration {iteration}/{NUM_ITERATIONS} ---")
        
        api_calls_this_iteration = 0

        # --- Stage 1: Propose Task (Sequential) ---
        task_type = random.choices(list(TASK_TYPE_DISTRIBUTION.keys()), weights=list(TASK_TYPE_DISTRIBUTION.values()), k=1)[0]
        k_examples_for_prompt = get_k_reference_examples()
        use_composite_task = random.random() < COMPOSITE_CONCEPT_PROBABILITY and learned_concepts_pool
        
        stochastic_seed_for_proposer = None
        if random.random() < STOCHASTIC_PERTURBATION_PROBABILITY:
            if learned_concepts_pool and random.random() < 0.5: # Prefer seeds from learned concepts if available
                stochastic_seed_for_proposer = random.choice(learned_concepts_pool).get("task_title", random.choice(RANDOM_SEED_CONCEPTS))
            else:
                stochastic_seed_for_proposer = random.choice(RANDOM_SEED_CONCEPTS)
            print(f"  🎲 Applying stochastic seed to proposer: '{stochastic_seed_for_proposer}'")


        proposer_prompt_text = ""
        # Updated to include new task generation functions and pass stochastic_seed
        if task_type == "synthesis_of_disparate_paradigms":
            proposer_prompt_text = generate_synthesis_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "generation_of_novel_axioms_and_exploration":
            proposer_prompt_text = generate_axioms_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "epistemological_boundary_probes":
            proposer_prompt_text = generate_epistemological_probe_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "hypothetical_scenario_exploration":
            proposer_prompt_text = generate_hypothetical_scenario_exploration_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "constrained_creative_challenge":
            proposer_prompt_text = generate_constrained_creative_challenge_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "first_principles_reimagination":
            proposer_prompt_text = generate_first_principles_reimagination_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "analogical_problem_solving":
            proposer_prompt_text = generate_analogical_problem_solving_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        elif task_type == "panel_discussion_challenge":
            proposer_prompt_text = generate_panel_discussion_challenge_task_user_question(k_examples_for_prompt, use_composite=use_composite_task, stochastic_seed=stochastic_seed_for_proposer)
        else:
            print(f"  Unknown task type for proposal: {task_type}. Skipping iteration.")
            continue
        
        print(f"🤖 Proposing {task_type} task{' (composite attempt)' if use_composite_task else ''}{' with seed: '+stochastic_seed_for_proposer if stochastic_seed_for_proposer else ''}...")
        proposer_full_llm_response = await query_llm_api(proposer_prompt_text, temperature=PROPOSER_TEMPERATURE, max_tokens=MAX_TOKENS_PROPOSER,
                                                         model_name=PRIMARY_MODEL_NAME, api_base_url=PRIMARY_API_BASE_URL, api_key=PRIMARY_API_KEY)
        api_calls_this_iteration += 1

        if not proposer_full_llm_response:
            print("  Proposer LLM failed to respond. Skipping iteration."); await asyncio.sleep(1); continue
        
        proposer_answer_content = extract_from_answer_tag(proposer_full_llm_response, task_type_for_heuristic=task_type)
        if not proposer_answer_content:
            print(f"  Proposer: No usable <answer> for {task_type}. Skipping."); await asyncio.sleep(1); continue
            
        current_task_core_data = parse_json_from_answer(proposer_answer_content)
        if not current_task_core_data:
            print(f"  Proposer: <answer> not valid JSON for {task_type}. Skipping."); await asyncio.sleep(1); continue

        if "task_title" not in current_task_core_data: 
             print(f"  Proposer: {task_type} JSON missing 'task_title'. Skipping."); await asyncio.sleep(1); continue
        
        current_task_core_data["task_type"] = task_type
        proposer_task_package = {
            "task_type": task_type, "proposer_full_llm_response": proposer_full_llm_response,
            "proposer_task_json_str": proposer_answer_content, "task_title": current_task_core_data.get("task_title", "Untitled")
        }
        print(f"  Proposer LLM proposed: {current_task_core_data.get('task_title', 'Untitled Task')[:80]}")
        success_criteria = f"A successful response for this '{task_type}' task should be coherent, deeply reasoned, directly address all aspects of the task description, demonstrate originality, and adhere to the expected output format. The thinking process should be transparent."
        current_task_core_data["success_criteria_for_solver"] = success_criteria

        # --- Stage 2: Solver (Initial Answer) + Self-Critique/Revision (Main + Rollouts) ---
        print(f"  🤖 Preparing solver and self-critique attempts...")
        
        async def solve_and_critique_revise_task(task_data_dict: Dict[str, Any], solver_temp: float, critique_temp: float, revise_temp: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            """ Helper to encapsulate solve -> critique -> revise logic """
            # Initial Solve
            solver_user_q = generate_solver_user_question(task_data_dict["task_type"], task_data_dict)
            initial_solver_resp = await query_llm_api(solver_user_q, temperature=solver_temp, max_tokens=MAX_TOKENS_SOLVER,
                                                      model_name=PRIMARY_MODEL_NAME, api_base_url=PRIMARY_API_BASE_URL, api_key=PRIMARY_API_KEY)
            initial_extracted_ans = extract_from_answer_tag(initial_solver_resp, task_data_dict["task_type"])

            if not initial_extracted_ans:
                print(f"    Solver (temp {solver_temp:.2f}) failed to produce initial usable answer.")
                return initial_solver_resp, None, None # Return full initial response, no critique, no revised

            # Self-Critique and Revision
            critique_revise_q = generate_critique_revise_user_question(
                json.dumps(task_data_dict, indent=2), # Pass full task data for context
                initial_extracted_ans,
                task_data_dict["task_type"]
            )
            critique_revise_resp = await query_llm_api(critique_revise_q, temperature=critique_temp, max_tokens=MAX_TOKENS_CRITIQUE_REVISE, # Using critique_temp, can be same as revise_temp
                                                        model_name=PRIMARY_MODEL_NAME, api_base_url=PRIMARY_API_BASE_URL, api_key=PRIMARY_API_KEY)
            
            critique_text, revised_answer_text = extract_from_critique_revise_response(critique_revise_resp)

            if not revised_answer_text:
                print(f"    Self-Critique/Revision failed to produce a revised answer. Using initial answer if available.")
                # Return initial response, the critique attempt, but no *new* revised answer (or initial if critique failed badly)
                return initial_solver_resp, critique_revise_resp, initial_extracted_ans # Fallback to initial if revision fails
            
            return initial_solver_resp, critique_revise_resp, revised_answer_text

        solve_critique_coroutines = []
        # Main attempt
        solve_critique_coroutines.append(solve_and_critique_revise_task(current_task_core_data, SOLVER_TEMPERATURE, CRITIQUE_TEMPERATURE, REVISE_TEMPERATURE))
        # Rollout attempts
        for _ in range(N_SOLVER_ROLLOUTS_FOR_PROPOSER):
            rollout_solver_temp = max(0.1, min(1.0, SOLVER_TEMPERATURE + random.uniform(-0.1, 0.1)))
            # For rollouts, critique/revise temps can also be varied or kept same
            solve_critique_coroutines.append(solve_and_critique_revise_task(current_task_core_data, rollout_solver_temp, CRITIQUE_TEMPERATURE, REVISE_TEMPERATURE))

        print(f"  🚀 Launching {len(solve_critique_coroutines) * 2} LLM calls (solve + critique/revise) for {len(solve_critique_coroutines)} solution attempts concurrently...")
        all_solve_critique_results = await asyncio.gather(*solve_critique_coroutines)
        api_calls_this_iteration += len(solve_critique_coroutines) * 2 # 1 for solve, 1 for critique/revise

        main_initial_solver_response, main_critique_response, main_revised_answer = all_solve_critique_results[0]
        rollout_results = all_solve_critique_results[1:] # List of (initial_resp, critique_resp, revised_ans)

        if not main_revised_answer:
            print("  Main solver's self-critique/revision failed to produce a final answer.")
            # If main_revised_answer is None, it means even the fallback to initial_extracted_ans failed.
            # This case should be rare if initial solve produced something.

        # --- Stage 3: All Evaluators (Main + Rollouts for TWO LLMs if configured) Concurrently ---
        # Evaluators will now assess the REVISED answers.
        print(f"  🔎 Preparing evaluator attempts for revised answers...")
        evaluator_tasks_coroutines = []
        num_actual_evaluator_api_calls = 0

        solutions_to_evaluate_revised = [(main_revised_answer, "Main (Revised)")]
        for i, (_, _, r_revised_ans) in enumerate(rollout_results): # Iterate through revised answers from rollouts
            solutions_to_evaluate_revised.append((r_revised_ans, f"Rollout {i+1} (Revised)"))

        for sol_revised_answer, sol_type_desc in solutions_to_evaluate_revised:
            if sol_revised_answer: # If a revised answer exists (even if it's fallback to initial)
                eval_prompt_primary = generate_evaluator_user_question(task_type, current_task_core_data, sol_revised_answer, success_criteria, PRIMARY_MODEL_NAME)
                evaluator_tasks_coroutines.append(query_llm_api(eval_prompt_primary, temperature=EVALUATOR_TEMPERATURE, max_tokens=MAX_TOKENS_EVALUATOR,
                                                                model_name=PRIMARY_MODEL_NAME, api_base_url=PRIMARY_API_BASE_URL, api_key=PRIMARY_API_KEY))
                num_actual_evaluator_api_calls +=1
                if SECONDARY_MODEL_NAME and SECONDARY_MODEL_NAME.strip() != "":
                    eval_prompt_secondary = generate_evaluator_user_question(task_type, current_task_core_data, sol_revised_answer, success_criteria, SECONDARY_MODEL_NAME)
                    evaluator_tasks_coroutines.append(query_llm_api(eval_prompt_secondary, temperature=EVALUATOR_TEMPERATURE, max_tokens=MAX_TOKENS_EVALUATOR,
                                                                    model_name=SECONDARY_MODEL_NAME, api_base_url=PRIMARY_API_BASE_URL, api_key=PRIMARY_API_KEY))
                    num_actual_evaluator_api_calls +=1
                else: 
                    evaluator_tasks_coroutines.append(async_return_value(None)) 
            else: # No answer (initial solve and critique/revise failed)
                evaluator_tasks_coroutines.append(async_return_value(json.dumps({"quality_score": 0.0, "justification": f"{sol_type_desc} solver/critique failed to produce any answer."})))
                if SECONDARY_MODEL_NAME and SECONDARY_MODEL_NAME.strip() != "": 
                    evaluator_tasks_coroutines.append(async_return_value(json.dumps({"quality_score": 0.0, "justification": f"{sol_type_desc} solver/critique failed (no secondary eval)."})))
                else:
                    evaluator_tasks_coroutines.append(async_return_value(None))

        print(f"  🚀 Launching {num_actual_evaluator_api_calls} actual evaluator LLM calls (up to {len(evaluator_tasks_coroutines)} total slots) concurrently...")
        all_evaluator_json_responses = await asyncio.gather(*evaluator_tasks_coroutines)
        api_calls_this_iteration += num_actual_evaluator_api_calls
        
        combined_eval_results = [] 
        eval_response_cursor = 0 

        for sol_revised_answer, sol_type_desc in solutions_to_evaluate_revised: 
            score1, just1 = 0.0, f"{sol_type_desc} primary eval failed or N/A."
            score2, just2 = 0.0, f"{sol_type_desc} secondary eval failed or N/A (or not configured)."
            
            primary_eval_json_str = all_evaluator_json_responses[eval_response_cursor]
            eval_response_cursor += 1 
            if primary_eval_json_str: 
                eval_data1 = parse_json_from_answer(primary_eval_json_str) 
                if eval_data1 and "quality_score" in eval_data1:
                    score1 = max(0.0, min(1.0, float(eval_data1["quality_score"])))
                    just1 = str(eval_data1.get("justification", "No justification from primary."))
                elif primary_eval_json_str: 
                     just1 = f"{sol_type_desc} primary eval malformed: {str(primary_eval_json_str)[:50]}"
            
            secondary_eval_json_str = all_evaluator_json_responses[eval_response_cursor]
            eval_response_cursor += 1 
            
            if SECONDARY_MODEL_NAME and SECONDARY_MODEL_NAME.strip() != "":
                if secondary_eval_json_str: 
                    eval_data2 = parse_json_from_answer(secondary_eval_json_str)
                    if eval_data2 and "quality_score" in eval_data2:
                        score2 = max(0.0, min(1.0, float(eval_data2["quality_score"])))
                        just2 = str(eval_data2.get("justification", "No justification from secondary."))
                    elif secondary_eval_json_str: 
                        just2 = f"{sol_type_desc} secondary eval malformed: {str(secondary_eval_json_str)[:50]}"
                
                avg_score = (score1 + score2) / 2.0
                combined_just = f"Primary ({PRIMARY_MODEL_NAME} S={score1:.2f}): {just1} | Secondary ({SECONDARY_MODEL_NAME} S={score2:.2f}): {just2}"
            else: 
                avg_score = score1
                combined_just = f"Primary ({PRIMARY_MODEL_NAME} S={score1:.2f}): {just1}"
            
            combined_eval_results.append((avg_score, combined_just))

        main_final_quality_score, main_final_quality_justification = combined_eval_results[0]
        rollout_final_quality_scores_tuples = combined_eval_results[1:]
        
        print(f"  Main Solution Final Quality (avg of revised): {main_final_quality_score:.2f}. Justification: {main_final_quality_justification[:150]}...")

        # --- Stage 4: Calculate Proposer Reward, Log, Learn (Sequential) ---
        rollout_scores_for_reward = [score for score, just in rollout_final_quality_scores_tuples]
        if N_SOLVER_ROLLOUTS_FOR_PROPOSER > 0 and rollout_scores_for_reward:
            avg_rollout_quality = sum(rollout_scores_for_reward) / len(rollout_scores_for_reward)
            proposer_reward = avg_rollout_quality
        elif N_SOLVER_ROLLOUTS_FOR_PROPOSER == 0 :
             proposer_reward = 0.5 
        else: 
            proposer_reward = 0.0
        print(f"  Proposer reward (r_propose based on {len(rollout_scores_for_reward)} avg rollout scores): {proposer_reward:.2f}")

        if main_revised_answer and main_final_quality_score >= LOGGING_QUALITY_THRESHOLD: 
            log_exploration_data(main_solver_user_question, main_initial_solver_response, # Log initial response
                                 main_critique_response, main_revised_answer, # Log critique and revised
                                 current_task_core_data, main_final_quality_score, main_final_quality_justification)
            add_to_experience_buffer(proposer_task_package, main_initial_solver_response, # Use initial for buffer context
                                     main_critique_response, main_revised_answer,
                                     main_final_quality_score, main_final_quality_justification)
            print(f"  ✅ Main solution (Final Quality: {main_final_quality_score:.2f}) logged and added to experience buffer.")
            add_to_learned_concepts_pool(current_task_core_data, main_revised_answer, main_final_quality_score)
        elif main_revised_answer: # Revised answer exists but quality too low
             print(f"  ❌ Main solution final quality ({main_final_quality_score:.2f}) too low (Threshold: {LOGGING_QUALITY_THRESHOLD}). Not logged.")
        else: # No revised answer was produced
             print(f"  ❌ Main solver/critique did not produce a final answer. Nothing to log or learn from this attempt.")

        # --- Iteration Throttling ---
        iteration_duration = time.monotonic() - iteration_start_time
        print(f"  Iteration {iteration} processed {api_calls_this_iteration} API calls in {iteration_duration:.2f} seconds.")
        if API_RPM_LIMIT > 0:
            target_calls_per_second = API_RPM_LIMIT / 60.0
            min_time_per_iteration_for_api_limit = api_calls_this_iteration / target_calls_per_second
            if iteration_duration < min_time_per_iteration_for_api_limit:
                sleep_duration = min_time_per_iteration_for_api_limit - iteration_duration
                print(f"  Throttling: Sleeping for {sleep_duration:.2f}s to maintain ~{API_RPM_LIMIT} API RPM.")
                await asyncio.sleep(sleep_duration)
            else: await asyncio.sleep(MIN_ITER_SLEEP)
        else: await asyncio.sleep(MIN_ITER_SLEEP)

    print("\n--- Finished ---")
    print(f"Exploration data saved to {FINETUNING_DATA_FILE}")
    print(f"Total successful experiences in buffer: {len(experience_buffer)}")
    print(f"Total concepts in learned_concepts_pool: {len(learned_concepts_pool)}")

if __name__ == "__main__":
    print("********************************************************************************")
    print("DISCLAIMER: This script generates highly speculative and abstract content using LLMs.")
    print("The generated tasks and solutions are for research and exploration into AI capabilities.")
    print("Interpret outputs with caution; they are not validated facts or established knowledge.")
    print("Ensure API Keys (e.g., PRIMARY_API_KEY) are set or updated.")
    print("The 'openai' library is required (pip install --upgrade openai for AsyncOpenAI).")
    print("********************************************************************************\n")
    
    try:
        import openai
        if not hasattr(openai, 'AsyncOpenAI'):
            print("Warning: Your 'openai' library version might be too old for AsyncOpenAI. Consider 'pip install --upgrade openai'.")
        print(f"OpenAI library version: {openai.__version__}")
    except ImportError:
        print("FATAL: 'openai' library not installed. Please run: pip install --upgrade openai"); exit(1)
    
    asyncio.run(main())
