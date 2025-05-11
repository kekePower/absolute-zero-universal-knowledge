import os
from typing import Dict, List

# --- Configuration ---
# Primary LLM Configuration
PRIMARY_API_BASE_URL = os.getenv("PRIMARY_API_BASE_URL", "https://api.novita.ai/v3/openai")
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY", "<Your_API_Key_HERE>") # SET THIS!
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "deepseek/deepseek-r1") # Proposer, Solver, Critiquer

# Secondary LLM Configuration (Evaluator - uses Primary API credentials)
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "qwen/qwen3-235b-a22b-fp8") # Set to "" or None to disable.

# Version Configuration
VERSION = "1.4.3"

# General Configuration
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "30")) # Adjusted due to increased calls per iter
K_REFERENCE_EXAMPLES = 2
N_SOLVER_ROLLOUTS_FOR_PROPOSER = int(os.getenv("N_SOLVER_ROLLOUTS_FOR_PROPOSER", "1")) # Reduced for faster iterations with new steps
FINETUNING_DATA_FILE = f"universal_knowledge_exploration_log_v{VERSION}.jsonl"

TASK_TYPE_DISTRIBUTION: Dict[str, float] = {
    "synthesis_of_disparate_paradigms": 0.15,
    "generation_of_novel_axioms_and_exploration": 0.15,
    "epistemological_boundary_probes": 0.10,
    "hypothetical_scenario_exploration": 0.15,
    "constrained_creative_challenge": 0.15,
    "first_principles_reimagination": 0.10,
    "analogical_problem_solving": 0.10,
    "panel_discussion_challenge": 0.10, # New
}

MAX_TOKENS_PROPOSER = 1000
MAX_TOKENS_SOLVER = 4500 # Might need more for panel discussions + synthesis
MAX_TOKENS_CRITIQUE_REVISE = 4500 # For critique and revised answer
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
RANDOM_SEED_CONCEPTS: List[str] = ["entropy", "fractals", "emergence", "symbiosis", "quantum entanglement", "neural networks", "game theory", "dark matter", "consciousness", "algorithmic bias", "terraforming", "bio-mimicry"]

# API Throttling Configuration
API_RPM_LIMIT = int(os.getenv("API_RPM_LIMIT", "10"))
MIN_ITER_SLEEP = 0.2

# New Ollama/Gemma-3 Configuration
OLLAMA_ENABLED = True  # Enable/disable Ollama integration
OLLAMA_API_BASE_URL = "http://localhost:11434/api"  # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "gemma3:4b-it-q8_0"          # Your specified Gemma-3 model

# System prompt for Gemma-3 to refine instructions
GEMMA_SYSTEM_PROMPT_FOR_REFINEMENT = (
    "You are a text editing AI. You will be given two pieces of text:"
    "1. 'Original Instructions': A set of instructions for another AI."
    "2. 'Task Description': Context about what the other AI is supposed to do."
    "\n"
    "YOUR ONLY JOB IS TO: Re-write and improve the 'Original Instructions' to make them clearer and more effective for the 'Task Description'."
    "\n"
    "RULES FOR YOUR OUTPUT (THE RE-WRITTEN 'ORIGINAL INSTRUCTIONS'):"
    "- You MUST preserve any specific formatting requirements mentioned in the 'Original Instructions' (e.g., XML-like tags such as <think></think>, <answer></answer>, requirements for JSON, and prohibitions on Markdown). Make these requirements very clear."
    "- Your output MUST be *ONLY* the re-written 'Original Instructions'."
    "- DO NOT output an example of what the other AI might say."
    "- DO NOT add any explanations, apologies, or conversational text."
    "- The re-written 'Original Instructions' MUST be plain text, with no Markdown used by you."
    "\n"
    "The user will provide the texts like this (DO NOT copy this part, just use the content):"
    "TEXT_TO_REWRITE_AND_OUTPUT (Original Instructions):"
    "---"
    "[Content of Original Instructions will be here]"
    "---"
    "CONTEXT_FOR_REWRITING (Task Description):"
    "---"
    "[Content of Task Description will be here]"
    "---"
    "\n"
    "Your entire response will be used as the new set of instructions for the other AI. Output only the re-written 'Original Instructions'."
)

# LLM API Client settings
# --- Model Names ---
