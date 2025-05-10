import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import json

from .config import (
    K_REFERENCE_EXAMPLES,
    MAX_LEARNED_CONCEPTS,
    COMPOSITE_CONCEPT_PROBABILITY,
    STOCHASTIC_PERTURBATION_PROBABILITY,
    RANDOM_SEED_CONCEPTS,
    LEARNED_CONCEPT_QUALITY_THRESHOLD 
)

# --- Globals for Curriculum Learning ---
experience_buffer: List[Dict[str, Any]] = []
learned_concepts_cache: List[Dict[str, Any]] = [] 
concept_quality_history: Dict[str, List[float]] = {} 

# --- Experience Buffer Management ---
class Experience(TypedDict):
    iteration: int
    task_type: str
    task_description: str # From proposer_task_details_parsed
    proposer_model: str
    proposer_temperature: float
    proposer_response: Optional[str] # Full proposer LLM response
    proposer_task_details_parsed: Optional[Dict[str, Any]] # Parsed JSON from proposer_response
    solver_model: Optional[str]
    solver_temperature: Optional[float]
    solver_user_question: Optional[str] # Full question/prompt given to solver
    solver_response: Optional[str]
    critique_revise_model: Optional[str]
    critique_revise_temperature: Optional[float]
    critique_revise_response: Optional[str]
    evaluator_model: Optional[str]
    evaluator_temperature: Optional[float]
    evaluator_response: Optional[str]
    evaluator_score: Optional[float]
    evaluator_feedback: Optional[str]
    final_answer: Optional[str]
    revised_answer: Optional[str]
    critique: Optional[str]
    proposed_concept_name: Optional[str]
    proposed_concept_description: Optional[str]
    is_new_concept_learned: bool
    learned_concept_details: Optional[Dict[str, Any]]

def add_to_experience_buffer(experience: Experience):
    experience_buffer.append(experience)
    # Optional: Could cap the buffer size here if needed
    # if len(experience_buffer) > MAX_BUFFER_SIZE: experience_buffer.pop(0)

def sample_from_experience_buffer(k: int = K_REFERENCE_EXAMPLES) -> List[Experience]:
    if not experience_buffer: return []
    return random.sample(experience_buffer, min(k, len(experience_buffer)))

# --- Concept Learning & Management ---
def log_concept_quality(concept_name: str, quality_score: float):
    if concept_name not in concept_quality_history:
        concept_quality_history[concept_name] = []
    concept_quality_history[concept_name].append(quality_score)
    
    # Update quality in learned_concepts_cache if it exists
    for concept_info in learned_concepts_cache:
        if concept_info['concept'] == concept_name:
            # Use a running average or simply the latest score for now
            # For simplicity, let's use a running average of the last N scores (e.g., 5)
            avg_score = np.mean(concept_quality_history[concept_name][-5:]) 
            concept_info['quality_score'] = avg_score
            print(f"Updated quality for concept '{concept_name}' to {avg_score:.2f}")
            break

def get_concept_to_propose() -> Optional[Dict[str, Any]]:
    """Selects a concept for the Proposer to use or generate based on curriculum strategy."""
    if not learned_concepts_cache or random.random() < STOCHASTIC_PERTURBATION_PROBABILITY: 
        print("Concept Proposer: Attempting to generate a new seed concept.")
        return None 

    # Filter for high-quality concepts
    high_quality_concepts = [c for c in learned_concepts_cache if c.get('quality_score', 0) >= LEARNED_CONCEPT_QUALITY_THRESHOLD]
    if not high_quality_concepts:
        print("Concept Proposer: No high-quality concepts available yet. Attempting to generate a new seed concept.")
        return None

    # Decide whether to use an existing concept or attempt to generate a composite concept
    if len(high_quality_concepts) >= 2 and random.random() < COMPOSITE_CONCEPT_PROBABILITY and len(learned_concepts_cache) < MAX_LEARNED_CONCEPTS:
        # Attempt to propose a composite concept
        # For now, we'll just signal this intent to the Proposer by returning a special marker or None
        # The actual combination logic will be in the Proposer's prompt generation based on this signal.
        print("Concept Proposer: Attempting to generate a composite concept from existing high-quality concepts.")
        # Select two distinct high-quality concepts
        concept1, concept2 = random.sample(high_quality_concepts, 2)
        # Return a dictionary indicating this is a composite concept suggestion
        return {
            "type": "composite_suggestion",
            "concepts_to_combine": [concept1, concept2],
            "combined_description": f"Combine insights from '{concept1['concept']}' and '{concept2['concept']}'."
        }
    else:
        # Select a single existing concept to explore further
        # Prioritize concepts that haven't been used recently or have high variance in quality (exploration vs exploitation)
        # Simple random choice among high-quality for now
        chosen_concept = random.choice(high_quality_concepts)
        print(f"Concept Proposer: Selected existing concept '{chosen_concept['concept']}' for further exploration.")
        return {
            "type": "existing_concept",
            "concept_details": chosen_concept
        }

# This function is for the Proposer to suggest a concept to be learned, not to learn it directly
# The actual learning happens in 'learn_new_concept'
# This function is more about the Proposer identifying a potential concept based on its generation.
# It might be deprecated or merged if 'learn_new_concept' handles all explicit concept formation.
# For now, let's assume the Proposer might output a 'proposed_concept_name' and 'proposed_concept_description'
# as part of its task generation, which is then fed into 'learn_new_concept'.

def learn_new_concept(
    proposed_concept_name: Optional[str],
    proposed_concept_description: Optional[str],
    task_description_for_concept: str, # The task description that embodies or led to this concept
    example_solution: Optional[str], # The solution to the task_description_for_concept
    quality_score: float, # The quality score of the example_solution for the task_description_for_concept
    task_type_for_concept: str, # Task type of the task_description_for_concept
    iteration_num: int, # Iteration number when this concept was crystalized
    full_experience_log: Experience # The full experience log entry for this iteration
) -> Optional[Dict[str, Any]]:
    """Learns a new concept if it's novel and meets quality thresholds."""
    if not proposed_concept_name or not proposed_concept_description:
        # print("  LearnConcept: No concept name or description provided.")
        return None

    if quality_score < LEARNED_CONCEPT_QUALITY_THRESHOLD:
        # print(f"  LearnConcept: Quality score {quality_score:.2f} for '{proposed_concept_name}' is below threshold {LEARNED_CONCEPT_QUALITY_THRESHOLD}. Skipping.")
        return None

    # Check for novelty (simple check by name for now, could be semantic)
    if any(c['concept'].lower() == proposed_concept_name.lower() for c in learned_concepts_cache):
        # print(f"  LearnConcept: Concept '{proposed_concept_name}' already exists. Updating quality.")
        log_concept_quality(proposed_concept_name, quality_score) # Update quality of existing concept
        # Potentially add new example to existing concept here if desired
        # for c in learned_concepts_cache:
        #     if c['concept'].lower() == proposed_concept_name.lower():
        #         # Add example if it's different enough or meets certain criteria
        #         new_example = {
        #             "task_description": task_description_for_concept,
        #             "example_solution": example_solution,
        #             "quality_score": quality_score,
        #             "task_type": task_type_for_concept,
        #             "iteration_learned": iteration_num
        #         }
        #         # Avoid duplicate examples (simple check)
        #         if not any(ex['task_description'] == new_example['task_description'] and ex['example_solution'] == new_example['example_solution'] for ex in c.get('examples', [])):
        #             c.setdefault('examples', []).append(new_example)
        #             print(f"    Added new example to existing concept '{proposed_concept_name}'. Total examples: {len(c['examples'])}")
        #         break
        return None # Not a *newly* learned concept in this call

    if len(learned_concepts_cache) >= MAX_LEARNED_CONCEPTS:
        # Optional: Implement a strategy to replace older/lower-quality concepts
        learned_concepts_cache.sort(key=lambda c: c.get('quality_score', 0) * len(c.get('examples', []))) # Prioritize removing low quality/few examples
        removed_concept = learned_concepts_cache.pop(0)
        print(f"  LearnConcept: Max concepts reached. Removed '{removed_concept['concept']}' to make space.")

    concept_info = {
        'concept': proposed_concept_name,
        'description': proposed_concept_description,
        'quality_score': quality_score, # Initial quality score
        'examples': [{
            "task_description": task_description_for_concept,
            "example_solution": example_solution,
            "quality_score": quality_score,
            "task_type": task_type_for_concept,
            "iteration_crystallized": iteration_num,
            # Store a reference to the full experience or key parts of it
            "experience_details": {
                "iteration": full_experience_log.get("iteration"),
                "task_type": full_experience_log.get("task_type"),
                "proposer_response_snippet": full_experience_log.get("proposer_response", "")[:100],
                "solver_response_snippet": full_experience_log.get("solver_response", "")[:100],
                "final_answer_snippet": full_experience_log.get("final_answer", "")[:100]
            }
        }],
        'creation_iteration': iteration_num,
        'last_updated_iteration': iteration_num
    }
    learned_concepts_cache.append(concept_info)
    log_concept_quality(proposed_concept_name, quality_score) # Log initial quality
    print(f"  Successfully learned NEW concept: '{proposed_concept_name}' (Quality: {quality_score:.2f}). Total concepts: {len(learned_concepts_cache)}")
    return concept_info
