import json
from typing import Dict, Any

from .config import FINETUNING_DATA_FILE
from .learning_manager import Experience # For type hinting

def log_exploration_data(entry: Experience):
    """Logs key information from an experience to the finetuning data file."""
    
    log_entry: Dict[str, Any] = {
        # Iteration and Task Info
        "iteration": entry.get("iteration"),
        "task_type": entry.get("task_type"),
        "task_description": entry.get("task_description"), # Already in experience
        
        # Proposer Details
        "proposer_model": entry.get("proposer_model"),
        "proposer_temperature": entry.get("proposer_temperature"),
        "proposer_response_full": entry.get("proposer_response"),
        "proposer_task_details_parsed": entry.get("proposer_task_details_parsed"), # Contains task_title, etc.
        
        # Solver Details
        "solver_model": entry.get("solver_model"),
        "solver_temperature": entry.get("solver_temperature"),
        "solver_user_question": entry.get("solver_user_question"),
        "solver_initial_full_response": entry.get("solver_response"), # This is the first/main solver response
        
        # Critique/Revise Details
        "critique_revise_model": entry.get("critique_revise_model"),
        "critique_revise_temperature": entry.get("critique_revise_temperature"),
        "critique_revise_full_response": entry.get("critique_revise_response"),
        "extracted_critique": entry.get("critique"),
        "extracted_revised_answer": entry.get("revised_answer"),
        
        # Evaluator Details
        "evaluator_model": entry.get("evaluator_model"),
        "evaluator_temperature": entry.get("evaluator_temperature"),
        "evaluator_full_response": entry.get("evaluator_response"),
        "final_answer_evaluated": entry.get("final_answer"),
        "final_quality_score": entry.get("evaluator_score"),
        "final_quality_justification": entry.get("evaluator_feedback"),
        
        # Learning Details
        "is_new_concept_learned": entry.get("is_new_concept_learned"),
        "learned_concept_details": entry.get("learned_concept_details") # This is a dict if a concept was learned
    }

    # Clean out None values for cleaner logs, if desired, or keep for completeness
    # log_entry_cleaned = {k: v for k, v in log_entry.items() if v is not None}

    try:
        with open(FINETUNING_DATA_FILE, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to log file {FINETUNING_DATA_FILE}: {e}")
        print(f"Log entry that failed: {log_entry}")
