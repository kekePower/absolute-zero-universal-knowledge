# Absolute Zero Universal Knowledge Generator
# This script implements a paradigm to generate tasks/questions
# that a human might not typically formulate, spanning any field of knowledge,
# and then has an LLM attempt to answer them.
# v1.4.6: Updated to use OpenAI for question generation.

import json
import random
import os
import time # For tracking iteration duration
import re
import asyncio # Added for asynchronous operations
from typing import Dict, List, Tuple, Any, Optional
from typing_extensions import TypedDict

# Import configurations
from modules.config import (
    PRIMARY_API_BASE_URL, PRIMARY_API_KEY, PRIMARY_MODEL_NAME,
    SECONDARY_MODEL_NAME, VERSION, NUM_ITERATIONS, K_REFERENCE_EXAMPLES,
    N_SOLVER_ROLLOUTS_FOR_PROPOSER, FINETUNING_DATA_FILE,
    TASK_TYPE_DISTRIBUTION, MAX_TOKENS_PROPOSER, MAX_TOKENS_SOLVER,
    MAX_TOKENS_CRITIQUE_REVISE, MAX_TOKENS_EVALUATOR,
    PROPOSER_TEMPERATURE, SOLVER_TEMPERATURE, CRITIQUE_TEMPERATURE,
    REVISE_TEMPERATURE, EVALUATOR_TEMPERATURE, LOGGING_QUALITY_THRESHOLD,
    LEARNED_CONCEPT_QUALITY_THRESHOLD, COMPOSITE_CONCEPT_PROBABILITY,
    MAX_LEARNED_CONCEPTS, STOCHASTIC_PERTURBATION_PROBABILITY,
    API_RPM_LIMIT, MIN_ITER_SLEEP, 
    OLLAMA_ENABLED, OLLAMA_MODEL_NAME, GEMMA_SYSTEM_PROMPT_FOR_REFINEMENT,
    OPENAI_QUESTION_MODEL # For logging/debug if needed, client handles API Key
)

# Import the LLM API client function
from modules.llm_api_client import query_llm_api, get_ollama_completion, generate_openai_completion

# Import prompt generation functions
from modules.prompt_generators import (
    get_base_proposer_prompt, # Corrected import
    generate_solver_user_question,
    generate_critique_revise_user_question,
    generate_evaluator_user_question
)
# Import response parsing functions
from modules.response_parsers import (
    extract_from_critique_revise_response, extract_from_answer_tag,
    parse_json_from_answer # _fix_json_string is internal to response_parsers
)
# Import learning management functions and globals
from modules.learning_manager import (
    experience_buffer, learned_concepts_cache, concept_quality_history, # Globals
    add_to_experience_buffer, sample_from_experience_buffer, 
    get_concept_to_propose, log_concept_quality, learn_new_concept, Experience, initialize_learning_manager, update_experience_buffer # Functions and TypedDict
)
# Import logging function
from modules.file_logger import log_exploration_data

# --- Helper for default async results ---
async def async_return_value(value: Any):
    return value

# --- Helper function for Ollama refinement ---
def refine_prompt_with_gemma(original_prompt: str, task_description: str) -> str:
    if OLLAMA_ENABLED:
        # Construct the explicit input format Gemma now expects
        gemma_input_prompt = (
            f"ORIGINAL_INSTRUCTIONS_TEMPLATE_WITH_PLACEHOLDER:\n"
            f"---\n"
            f"{original_prompt}\n"
            f"---\n"
            f"TASK_CONTEXT_FOR_PLACEHOLDER_REPLACEMENT:\n"
            f"---\n"
            f"{task_description}\n"
            f"---"
        )
        
        print(f"\nAttempting to refine prompt for '{task_description}' with {OLLAMA_MODEL_NAME}...")
        refined_text = get_ollama_completion(
            prompt_text=gemma_input_prompt, # Send the combined task and initial instructions
            system_prompt=GEMMA_SYSTEM_PROMPT_FOR_REFINEMENT
        )
        if refined_text:
            print(f"Successfully refined prompt with {OLLAMA_MODEL_NAME}.")
            # Log original and refined for debugging/analysis if needed
            # log_interaction("GemmaRefinement", gemma_input_prompt, refined_text, OLLAMA_MODEL_NAME, 0, "system_refinement")
            return refined_text
        else:
            print(f"{OLLAMA_MODEL_NAME} refinement failed or returned empty. Using original prompt for '{task_description}'.")
    return original_prompt

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

    initialize_learning_manager()

    total_openai_prompt_tokens = 0
    total_openai_completion_tokens = 0
    total_openai_calls = 0
    total_novita_calls = 0
    successful_experiences = 0
    failed_experiences = 0
    script_start_time = time.monotonic()

    for iteration in range(1, NUM_ITERATIONS + 1):
        iteration_start_time = time.monotonic()
        print(f"\n--- Iteration {iteration}/{NUM_ITERATIONS} ---")
        
        api_calls_this_iteration = 0

        # Initialize current_experience for this iteration
        task_type = random.choices(list(TASK_TYPE_DISTRIBUTION.keys()), weights=list(TASK_TYPE_DISTRIBUTION.values()), k=1)[0]
        current_experience: Experience = {
            "iteration": iteration,
            "task_type": task_type,
            "task_description": "N/A", # Will be updated after proposer
            "proposer_model": OPENAI_QUESTION_MODEL,
            "proposer_temperature": PROPOSER_TEMPERATURE,
            "proposer_response": None,
            "proposer_task_details_parsed": None,
            "solver_model": None, 
            "solver_temperature": None,
            "solver_user_question": None,
            "solver_response": None,
            "critique_revise_model": None,
            "critique_revise_temperature": None,
            "critique_revise_response": None,
            "evaluator_model": None,
            "evaluator_temperature": None,
            "evaluator_response": None,
            "evaluator_score": None,
            "evaluator_feedback": None,
            "final_answer": None,
            "revised_answer": None,
            "critique": None,
            "proposed_concept_name": None, # Will be updated after proposer
            "proposed_concept_description": None, # Will be updated after proposer
            "is_new_concept_learned": False,
            "learned_concept_details": None,
            "proposer_prompt_tokens": 0, # Default values
            "proposer_completion_tokens": 0,
            "proposer_total_tokens": 0
        }

        # --- Stage 1: Propose Task (Now with OpenAI) ---
        print(f"  Selected task type for proposer: {task_type}")
        concept_for_proposer = get_concept_to_propose() # From learning_manager
        k_examples_for_prompt = sample_from_experience_buffer(K_REFERENCE_EXAMPLES) # From learning_manager
        
        stochastic_seed_for_proposer = None
        if random.random() < STOCHASTIC_PERTURBATION_PROBABILITY:
            # Prefer seeds from learned concepts if available and they have a task_title
            # Fallback to a generic seed if cache is empty or no title found
            potential_seeds = [lc.get("task_title") for lc in learned_concepts_cache if lc.get("task_title")]
            if potential_seeds and random.random() < 0.75: # Higher chance to use learned concept if available
                stochastic_seed_for_proposer = random.choice(potential_seeds)
            else:
                # If no learned concepts with titles or by random chance, use a generic seed from a predefined list
                generic_seeds = [
                    "exploring fundamental limits of AI understanding", 
                    "the nature of consciousness in artificial entities",
                    "novel frameworks for interdisciplinary knowledge synthesis",
                    "redefining creativity in the age of advanced AI",
                    "ethical dilemmas in a world with superintelligent machines"
                ]
                stochastic_seed_for_proposer = random.choice(generic_seeds)
            print(f"  Applying stochastic seed to proposer: '{stochastic_seed_for_proposer}'")

        # This function now effectively prepares the user_prompt for the OpenAI Proposer
        # Corrected to use get_base_proposer_prompt
        proposer_full_prompt_for_openai = get_base_proposer_prompt(
            task_type, # Pass the string task_type key
            k_examples_for_prompt,
            main_concept=concept_for_proposer,
            stochastic_seed=stochastic_seed_for_proposer
        )
        
        if OLLAMA_ENABLED: # This block will be skipped as OLLAMA_ENABLED is False
            # refine_prompt_with_gemma has its own logging for start/success/failure
            print("  Attempting to refine proposer prompt with Ollama/Gemma (this step should be skipped if OLLAMA_ENABLED is False)...")
            refined_proposer_prompt = refine_prompt_with_gemma(proposer_full_prompt_for_openai, "Propose a new challenging task/question")
            if refined_proposer_prompt and refined_proposer_prompt != proposer_full_prompt_for_openai:
                proposer_full_prompt_for_openai = refined_proposer_prompt
        
        # Log the exact prompt being sent to the OpenAI Proposer LLM
        print(f"\nDEBUG: Final prompt for OpenAI Proposer LLM ({OPENAI_QUESTION_MODEL}):\n---\n{proposer_full_prompt_for_openai}\n---")

        proposer_response_content, proposer_usage_info = await generate_openai_completion(
            user_prompt=proposer_full_prompt_for_openai,
            temperature=PROPOSER_TEMPERATURE,
            max_tokens=MAX_TOKENS_PROPOSER,
            model_name=OPENAI_QUESTION_MODEL
            # Default system prompt from generate_openai_completion will be used
        )
        current_experience["proposer_response"] = proposer_response_content
        total_openai_calls += 1
        api_calls_this_iteration += 1 # Counts calls within this iteration for throttling

        if proposer_usage_info:
            current_experience["proposer_prompt_tokens"] = proposer_usage_info.get("prompt_tokens", 0)
            current_experience["proposer_completion_tokens"] = proposer_usage_info.get("completion_tokens", 0)
            current_experience["proposer_total_tokens"] = proposer_usage_info.get("total_tokens", 0)
            total_openai_prompt_tokens += current_experience["proposer_prompt_tokens"]
            total_openai_completion_tokens += current_experience["proposer_completion_tokens"]
        
        # Check if proposer failed
        if not current_experience["proposer_response"]:
            print("  ERROR: Proposer LLM (OpenAI) failed to generate a response. Skipping this iteration.")
            failed_experiences += 1
            # Optional: Add to a list of failed experiences or log more details
            # Simple sleep before next iteration to avoid tight loop on persistent failure
            await asyncio.sleep(MIN_ITER_SLEEP * 5) 
            continue # Skip to next iteration

        current_task_core_data = parse_json_from_answer(extract_from_answer_tag(current_experience["proposer_response"]))
        if not current_task_core_data or not isinstance(current_task_core_data, dict):
            print("  Proposer failed to generate valid task JSON. Skipping iteration.")
            # Log minimal experience for failure analysis if desired
            add_to_experience_buffer(current_experience) 
            await asyncio.sleep(MIN_ITER_SLEEP)
            continue
        
        current_experience["proposer_task_details_parsed"] = current_task_core_data
        current_experience["task_description"] = current_task_core_data.get("task_description", "Task description missing")
        current_experience["proposed_concept_name"] = current_task_core_data.get("proposed_concept_name")
        current_experience["proposed_concept_description"] = current_task_core_data.get("proposed_concept_description")

        print(f"  Proposer task ({current_task_core_data.get('task_title', 'Untitled')}): {current_experience['task_description'][:100]}...")

        # --- Stage 2: Solve Task (Parallel with Rollouts if N_SOLVER_ROLLOUTS_FOR_PROPOSER > 1) ---
        user_question_for_solver = generate_solver_user_question(task_type, current_experience)
        user_question_for_solver = refine_prompt_with_gemma(user_question_for_solver, f"Solve the task: {current_experience['task_description']}")
        current_experience["solver_user_question"] = user_question_for_solver
        current_experience["solver_model"] = PRIMARY_MODEL_NAME # Assuming primary for now
        current_experience["solver_temperature"] = SOLVER_TEMPERATURE

        solver_tasks = []
        num_solver_attempts = N_SOLVER_ROLLOUTS_FOR_PROPOSER if task_type == "self_critique_and_revision" else 1
        
        for _ in range(num_solver_attempts):
            solver_tasks.append(query_llm_api(user_question_for_solver, SOLVER_TEMPERATURE, MAX_TOKENS_SOLVER, 
                                            PRIMARY_MODEL_NAME, PRIMARY_API_BASE_URL, PRIMARY_API_KEY))
        
        solver_responses = await asyncio.gather(*solver_tasks)
        current_experience["solver_response"] = solver_responses[0] # Store first response for simplicity in log
        api_calls_this_iteration += num_solver_attempts
        main_final_answer_from_solver = extract_from_answer_tag(solver_responses[0], task_type)

        if not main_final_answer_from_solver:
            print("  Solver failed to generate an answer. Skipping further stages.")
            # Log current state of experience
            add_to_experience_buffer(current_experience)
            await asyncio.sleep(MIN_ITER_SLEEP)
            continue
        print(f"  Solver initial answer: {main_final_answer_from_solver[:100]}...")

        # --- Stage 3: Critique and Revise (Main Path, if applicable) ---
        main_critique_text: Optional[str] = None
        main_revised_answer: Optional[str] = None
        current_experience["critique_revise_model"] = PRIMARY_MODEL_NAME # Assuming primary for now
        current_experience["critique_revise_temperature"] = CRITIQUE_TEMPERATURE

        if task_type in ["self_critique_and_revision", "concept_proposal", "instruction_following", "multi_turn_dialogue", "exploratory_qa"]:
            user_question_for_critique_revise = generate_critique_revise_user_question(
                current_experience["task_description"], 
                main_final_answer_from_solver, 
                task_type
            )
            user_question_for_critique_revise = refine_prompt_with_gemma(user_question_for_critique_revise, f"Refine the answer for: {current_experience['task_description']}")
            current_experience["critique_revise_response"] = await query_llm_api(
                user_question_for_critique_revise, CRITIQUE_TEMPERATURE, MAX_TOKENS_CRITIQUE_REVISE,
                PRIMARY_MODEL_NAME, PRIMARY_API_BASE_URL, PRIMARY_API_KEY
            )
            api_calls_this_iteration += 1
            main_critique_text, main_revised_answer = extract_from_critique_revise_response(current_experience["critique_revise_response"])
            current_experience["critique"] = main_critique_text
            current_experience["revised_answer"] = main_revised_answer
            if main_revised_answer: print(f"  Solver revised answer: {main_revised_answer[:100]}...")
            if main_critique_text: print(f"  Solver critique: {main_critique_text[:100]}...")

        # --- Stage 4: Evaluate (Main Path) ---
        answer_to_evaluate = main_revised_answer if main_revised_answer else main_final_answer_from_solver
        current_experience["final_answer"] = answer_to_evaluate # Store the answer that will be/was evaluated
        
        # Determine which model to use for evaluation
        evaluator_model_for_this_run = PRIMARY_MODEL_NAME # Default to primary
        if SECONDARY_MODEL_NAME and SECONDARY_MODEL_NAME.strip() != "":
            evaluator_model_for_this_run = SECONDARY_MODEL_NAME
            # print(f"DEBUG: Using Secondary Evaluator: {evaluator_model_for_this_run}") # Optional debug
        # else:
            # print(f"DEBUG: Using Primary Evaluator: {evaluator_model_for_this_run}") # Optional debug

        current_experience["evaluator_model"] = evaluator_model_for_this_run
        current_experience["evaluator_temperature"] = EVALUATOR_TEMPERATURE

        if not answer_to_evaluate:
            print("  No answer to evaluate. Skipping evaluation.")
            main_final_quality_score = 0.0
            main_final_quality_justification = "No answer provided by solver or critique/revise."
        else:
            user_question_for_evaluator = generate_evaluator_user_question(
                current_experience.get("task_type_generated", "N/A"),
                current_task_core_data, # This is current_experience["proposer_task_details_parsed"]
                answer_to_evaluate, 
                current_task_core_data.get("success_criteria"),
                evaluator_model_for_this_run # Pass the chosen evaluator model
            )
            if OLLAMA_ENABLED:
                # The refine_prompt_with_gemma function itself has print statements for start/success/failure
                refined_eval_prompt = refine_prompt_with_gemma(user_question_for_evaluator, f"Evaluate the solution for: {current_experience.get('task_description', 'N/A')}")
                if refined_eval_prompt and refined_eval_prompt != user_question_for_evaluator: # Check if refinement actually happened
                    user_question_for_evaluator = refined_eval_prompt
                # No 'else' needed here for printing, as refine_prompt_with_gemma handles its own failure print

            # Log the exact prompt being sent to the main Evaluator LLM
            print(f"\nDEBUG: Final prompt for Evaluator LLM ({evaluator_model_for_this_run}):\n---\n{user_question_for_evaluator}\n---")

            current_experience["evaluator_response"] = await query_llm_api(
                user_question_for_evaluator, 
                EVALUATOR_TEMPERATURE, 
                MAX_TOKENS_EVALUATOR,
                evaluator_model_for_this_run, # Pass the chosen evaluator model
                PRIMARY_API_BASE_URL, 
                PRIMARY_API_KEY
            )
            api_calls_this_iteration += 1
            evaluator_parsed_response = parse_json_from_answer(extract_from_answer_tag(current_experience["evaluator_response"], "evaluator"))

            if evaluator_parsed_response and isinstance(evaluator_parsed_response, dict):
                main_final_quality_score = float(evaluator_parsed_response.get("quality_score", 0.0))
                main_final_quality_justification = evaluator_parsed_response.get("justification", "No justification provided.")
            else:
                print("  Evaluator failed to provide valid JSON. Assigning low score.")
                main_final_quality_score = 0.1 # Penalize evaluator failure
                main_final_quality_justification = "Evaluator response parsing failed."
        
        current_experience["evaluator_score"] = main_final_quality_score
        current_experience["evaluator_feedback"] = main_final_quality_justification
        print(f"  Evaluator Quality Score: {main_final_quality_score:.2f}. Justification: {main_final_quality_justification[:100]}...")

        # --- Stage 5: Learn Concept (if applicable) ---
        if current_experience["evaluator_score"] is not None and \
           current_experience["evaluator_score"] >= LEARNED_CONCEPT_QUALITY_THRESHOLD and \
           current_experience["final_answer"] and \
           (current_experience.get("proposed_concept_name") or task_type == "concept_proposal"): # Only attempt to learn if a concept was proposed or it's a concept_proposal task
            
            learned_concept_details = learn_new_concept(
                proposed_concept_name=current_experience.get("proposed_concept_name"),
                proposed_concept_description=current_experience.get("proposed_concept_description"),
                task_description_for_concept=current_experience.get("task_description", "N/A"),
                example_solution=current_experience["final_answer"],
                quality_score=current_experience["evaluator_score"],
                task_type_for_concept=current_experience["task_type"],
                iteration_num=current_experience["iteration"],
                full_experience_log=current_experience
            )
            if learned_concept_details:
                current_experience["is_new_concept_learned"] = True
                current_experience["learned_concept_details"] = learned_concept_details
                log_concept_quality(learned_concept_details['concept'], learned_concept_details['quality_score']) # Also ensure quality is logged via learning_manager

        # --- Stage 6: Log Exploration Data & Add to Experience Buffer ---
        add_to_experience_buffer(current_experience) # Add fully populated experience

        if current_experience["evaluator_score"] is not None and current_experience["evaluator_score"] >= LOGGING_QUALITY_THRESHOLD:
            log_exploration_data(current_experience)
            print(f"  Main solution (Final Quality: {current_experience['evaluator_score']:.2f}) logged.")
        elif current_experience["final_answer"]: # Final answer exists but quality too low
             print(f"  Main solution final quality ({current_experience['evaluator_score']:.2f}) too low (Threshold: {LOGGING_QUALITY_THRESHOLD}). Not logged for finetuning.")
        else: # No final answer was produced
             print("  No final answer to log.")

        iter_time = time.monotonic() - iteration_start_time
        print(f"  Iteration {iteration} took {iter_time:.2f} seconds.")
        if API_RPM_LIMIT > 0:
            target_calls_per_second = API_RPM_LIMIT / 60.0
            min_time_per_iteration_for_api_limit = api_calls_this_iteration / target_calls_per_second
            if iter_time < min_time_per_iteration_for_api_limit:
                sleep_duration = min_time_per_iteration_for_api_limit - iter_time
                print(f"  Throttling: Sleeping for {sleep_duration:.2f}s to maintain ~{API_RPM_LIMIT} API RPM.")
                await asyncio.sleep(sleep_duration)
            else: await asyncio.sleep(MIN_ITER_SLEEP)
        else: await asyncio.sleep(MIN_ITER_SLEEP)

    # --- Save experience buffer to file ---
    try:
        with open(FINETUNING_DATA_FILE, 'w', encoding='utf-8') as f:
            for experience_item in experience_buffer: # Assuming experience_buffer is a list of dicts
                f.write(json.dumps(experience_item) + '\n')
        print(f"Exploration data successfully saved to {FINETUNING_DATA_FILE}")
    except IOError as e:
        print(f"Error: Could not write to file {FINETUNING_DATA_FILE}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving data: {e}")

    print("\n--- Script Execution Summary ---")
    script_end_time = time.monotonic()
    total_runtime_seconds = script_end_time - script_start_time
    print(f"Total Runtime: {total_runtime_seconds:.2f} seconds")
    print(f"Total Iterations: {NUM_ITERATIONS}")
    print(f"Successful Experiences (logged based on threshold): {successful_experiences}")
    print(f"Failed/Partial Experiences: {failed_experiences}")
    
    print("\n--- API Call & Token Usage Summary ---")
    print(f"OpenAI API Calls: {total_openai_calls}")
    print(f"  Total OpenAI Prompt Tokens: {total_openai_prompt_tokens}")
    print(f"  Total OpenAI Completion Tokens: {total_openai_completion_tokens}")
    print(f"  Total OpenAI Tokens: {total_openai_prompt_tokens + total_openai_completion_tokens}")
    print(f"Novita (or similar primary provider) API Calls: {total_novita_calls}")
    # Token details for Novita are not returned by its API, so not tracked here.

    print(f"\nTotal successful experiences in final buffer: {len(experience_buffer)}")
    print(f"Total concepts in final learned_concepts_cache: {len(learned_concepts_cache)}")

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
