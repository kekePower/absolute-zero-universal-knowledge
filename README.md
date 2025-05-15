# Absolute Zero Universal Knowledge Generator

This script generates questions that we, as humans, would or could never ask and then tries to answer them. This is then added to a JSONL file that can be used to fine-tune other LLMs that, in turn, becomes more knowledgeable and helps us learn new things. I guess some of the information will be completely useless and some could, eventually, be world changing.

I used Google Gemini 2.5 Pro Preview 05-06 to create the script based on the [Absolute Zero](https://arxiv.org/abs/2505.03335) research paper.

I'm also sure that some VERY smart people, people like you, will come along and update this script to make it even more awesome.

## Backend

I'm using [Novita.ai](https://novita.ai/) to get access to DeekSeek R1, however it's possible to use any provider. I thought using R1 would be good since it's the largest OSS LLM available at the moment.

With a few changes, I think it'd be possible to make the script more modular so that it gets easier to use any provider and model.

## System Requirements

This project requires a specific environment configuration to operate correctly. Ensure the following are set up:

*   **Python Environment**:
    *   Python 3.8 or higher.
    *   Required libraries: Install dependencies using `pip install -r requirements.txt`. Key libraries include `requests` (for API calls) and `python-dotenv` (for environment variable management). Create a `requirements.txt` if one doesn't exist, listing at least these.

*   **Ollama (for Instruction Refinement)**:
    *   **Installation**: A local Ollama instance must be running.
    *   **Model**: The system is configured to use a model like `gemma3:4b-it-q8_0` (or as specified in `modules/config.py` via `OLLAMA_MODEL_NAME`) for refining instructions. Ensure this model is pulled and available in your Ollama instance (`ollama pull gemma3:4b-it-q8_0`).
    *   **Configuration**: The Ollama API endpoint is typically `http://localhost:11434/api/generate`. This can be configured via `OLLAMA_API_BASE_URL` in your `.env` file or directly in `modules/config.py`.
    *   **Enablement**: The Ollama refinement step can be toggled using the `OLLAMA_ENABLED` boolean flag in the configuration.

*   **Primary LLM API Access**:
    *   **Provider**: The script uses Novita.ai as the primary provider, leveraging its increased 100 RPM limit for parallel API calls.
    *   **Model Configuration**: The default models are:
        *   Solver (PRIMARY_MODEL_NAME): Qwen3 (qwen/qwen3-235b-a22b-fp8)
        *   Evaluator (SECONDARY_MODEL_NAME): DeepSeek (deepseek/deepseek-v3-0324)
        *   Proposer: OpenAI gpt-4.1-mini
    *   **API Key**: A valid Novita.ai API key is required and should be set as the `PRIMARY_API_KEY` environment variable.
    *   **Performance Optimization**: For panel discussion tasks, the script now parallelizes Novita API calls using asyncio.gather, significantly improving efficiency.

*   **Environment Variables (`.env` file)**:
    *   It is highly recommended to use a `.env` file in the project root to manage sensitive information and configurations. The script uses `python-dotenv` to load these variables.
    *   Essential variables include:
        *   `PRIMARY_API_KEY`: For your main LLM provider.
        *   `OLLAMA_ENABLED` (e.g., `True` or `False`): To toggle Ollama-based instruction refinement.
        *   `OLLAMA_API_BASE_URL` (e.g., `http://localhost:11434/api/generate`): Ollama API endpoint.
        *   `OLLAMA_MODEL_NAME` (e.g., `gemma3:4b-it-q8_0`): The Ollama model for refinement.
        *   `PRIMARY_MODEL_NAME` (e.g., `deepseek-coder-33b-instruct`): Default primary LLM.
        *   Other model names and configuration parameters as defined in `modules/config.py`.

## Prompt Engineering & LLM Interaction

The core of this system relies on a sophisticated multi-agent architecture and carefully crafted prompt engineering techniques:

*   **Multi-Agent Architecture**:
    The system employs several specialized LLM "agents," each with a distinct role in the knowledge generation pipeline:
    *   **Proposer AI**: Generates novel and complex tasks based on a predefined distribution of task types and, optionally, existing examples or conceptual seeds. Its output is a structured JSON defining the task.
    *   **Solver AI**: Attempts to provide a comprehensive solution to the task proposed. It often follows a "think-then-answer" methodology.
    *   **Critique/Reviser AI**: (Optionally) Evaluates the Solver's output, provides a critique, and then generates a revised, improved answer.
    *   **Evaluator AI**: Assesses the quality, novelty, and adherence to success criteria of the generated task-solution pairs, outputting a structured JSON with a score and justification.

*   **Structured Prompts & Output Formats**:
    *   **Custom Tags**: Prompts are designed to elicit structured responses from LLMs using custom XML-like tags (e.g., `<think> </think>`, `<answer> </answer>`, `<critique> </critique>`, `<revised_answer> </revised_answer>`). This is critical for reliably parsing LLM outputs. The `R1_PROMPT_WRAPPER` in `modules/prompt_generators.py` standardizes the think/answer structure for certain models.
    *   **JSON Payloads**: Many agents are instructed to return their primary output as a well-formed JSON object within the `<answer>` (or equivalent) tags. This facilitates programmatic access to the generated data.
    *   **"No Markdown" Policy**: All prompts explicitly instruct the LLMs to **avoid using any Markdown formatting** within JSON string values or in plain text responses. This ensures that the output is clean, directly parsable, and avoids rendering issues or unexpected characters.

*   **Instruction Refinement via Ollama (Gemma)**:
    *   **Purpose**: To enhance the clarity, focus, and effectiveness of instructions sent to the primary (often larger and more expensive) LLMs.
    *   **Process**: If `OLLAMA_ENABLED` is true, prompts generated for the Proposer, Solver, etc., are first sent to a local Ollama instance running a smaller, faster model (e.g., Gemma).
    *   **Refinement Prompt**: Gemma's behavior is guided by `GEMMA_SYSTEM_PROMPT_FOR_REFINEMENT` (in `modules/config.py`), which instructs it to improve the given prompt while crucially ensuring that the "no Markdown" directive is preserved or added to the refined instructions.
    *   **Workflow**: Original Prompt -> Gemma (via Ollama) for Refinement -> Refined Prompt -> Primary LLM.

*   **Modularity**:
    *   Prompt generation logic is centralized in `modules/prompt_generators.py`.
    *   Core configurations, including model names, API settings, and system prompts, are managed in `modules/config.py` and through environment variables.
    *   LLM API interactions are handled by `modules/llm_api_client.py`.

This structured approach to prompt engineering and LLM interaction is key to the system's ability to generate, solve, and evaluate complex tasks in a semi-autonomous loop.

## Description

### What it Does

The "Absolute Zero Universal Knowledge Generator" (AZR-UKG) is a Python script that implements a paradigm to:
1.  **Generate Unconventional Tasks**: It formulates complex, abstract, and often esoteric tasks or questions that a human might not typically conceive. These can span any field of knowledge.
2.  **Attempt LLM-based Solutions**: It then employs Large Language Models (LLMs) to try and answer these generated tasks.
3.  **Explore AI Capabilities**: The primary goal is to probe the boundaries of LLM knowledge, reasoning, and creative problem-solving abilities, particularly with highly novel or abstract prompts.

### How it Works

The script operates through a sophisticated multi-stage process involving several LLM roles:

*   **Task Generation (Proposer)**: An LLM acts as a "Proposer" to create tasks. The generation process is guided by a distribution of predefined `TASK_TYPE_DISTRIBUTION`. For panel discussion tasks, the system now uses OpenAI's gpt-4.1-mini to generate initial/seed questions, replacing the previous static RANDOM_SEED_CONCEPTS.
*   **Solution Generation (Solver)**: Another LLM instance, the "Solver," attempts to provide a detailed answer to the generated task. This often involves a "think-then-answer" methodology, where the LLM first outlines its reasoning process.
*   **Self-Critique & Revision**: The Solver's initial response can be subjected to a self-critique process, where an LLM (or the same LLM in a different role) critiques the answer, and then a revised answer is generated based on this critique.
*   **Evaluation (Evaluator)**: A separate LLM, the "Evaluator," assesses the quality and novelty of the generated task-solution pair. This evaluation can influence what is considered a "learned concept."
*   **Curriculum Learning & Experience Buffer**: The system maintains an `experience_buffer` and a `learned_concepts_pool`. High-quality outputs (based on evaluation scores) are added to these pools. These can then be fed back into the Proposer as examples (`K_REFERENCE_EXAMPLES`) to guide future task generation, creating a self-improving or curriculum learning loop.
*   **Stochastic Perturbations**: To encourage novelty, the Proposer can be influenced by `STOCHASTIC_PERTURBATION_PROBABILITY` and `RANDOM_SEED_CONCEPTS`, injecting randomness or specific conceptual seeds into the task generation process.
*   **Panel of Experts Simulation**: For certain tasks, the Solver might simulate a "panel of experts" to generate a more comprehensive or multi-faceted answer.
*   **Asynchronous Operations**: Leverages `asyncio` for efficient, non-blocking calls to LLM APIs, along with API rate limit management.
*   **Data Logging for Finetuning**: All significant interactions, including the task, initial answer, critique, revised answer, and evaluation scores, are logged to a JSONL file (e.g., `universal_knowledge_exploration_log_v{VERSION}.jsonl`). This data is structured for potential use in fine-tuning LLMs.
*   **Configuration**: The script is highly configurable via environment variables for API keys, model names, number of iterations, token limits, temperatures for different LLM roles, and quality thresholds.

### Thoughts Behind It

*   **Pushing LLM Boundaries**: The core motivation is to move beyond standard LLM benchmarks and explore their capacity for generating and solving problems that require deep abstraction, creativity, and synthesis of potentially disparate ideas.
*   **"Absolute Zero" Knowledge Generation**: The name suggests an ambition to generate knowledge or explore conceptual spaces from fundamental principles or in novel ways, without heavy reliance on direct human input for specific problem domains.
*   **Research into AI Self-Improvement**: The curriculum learning and experience feedback mechanisms are experimental approaches to AI self-improvement, where the system learns from its own outputs to generate progressively more complex or insightful content.
*   **Understanding Complex Reasoning**: By prompting LLMs with tasks that are inherently difficult for humans to formulate or answer, the script aims to shed light on the advanced reasoning capabilities and potential failure modes of current AI models.
*   **Speculative Exploration**: As highlighted by the disclaimer in the script, the generated content is speculative and intended for research into AI capabilities. It's a tool for exploring what's possible rather than producing validated factual knowledge.

## License

MIT License
