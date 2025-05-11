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
VERSION = "1.4.1"

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
RANDOM_SEED_CONCEPTS: List[str] = ["entropy", "fractals", "emergence", "symbiosis", "quantum entanglement", "neural networks", "game theory", "dark matter", "consciousness", "algorithmic bias", "terraforming", "bio-mimicry"]

# API Throttling Configuration
API_RPM_LIMIT = int(os.getenv("API_RPM_LIMIT", "10"))
MIN_ITER_SLEEP = 0.2

# New Ollama/Gemma-3 Configuration
OLLAMA_ENABLED = True  # Enable/disable Ollama integration
OLLAMA_API_BASE_URL = "http://localhost:11434/api"  # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "gemma3:4b-it-q8_0"          # Your specified Gemma-3 model

# System prompt for Gemma-3 to refine instructions
GEMMA_SYSTEM_PROMPT_FOR_REFINEMENT = """You are an expert instruction enhancer. 
Your task is to take a set of initial instructions and a target task, and then generate a more comprehensive, clear, and focused set of instructions tailored for a large language model (LLM) to perform that task. 
The refined instructions should guide the LLM to produce high-quality, relevant, and accurate results. 
Consider potential ambiguities in the original instructions and try to clarify them. 
Ensure the refined instructions are actionable and provide any necessary context or constraints.
IMPORTANT: The refined instructions must explicitly tell the target LLM to avoid using any Markdown formatting in its response. For example, include a sentence like 'Do not use Markdown formatting in your response.'
Output only the refined instructions, without any preamble or explanation."""

# --- Model Names ---
