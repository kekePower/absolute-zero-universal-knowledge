# Absolute Zero Universal Knowledge Generator

This script generates questions that we, as humans, would or could never ask and then tries to answer them. This is then added to a JSONL file that can be used to fine-tune other LLMs that, in turn, becomes more knowledgeable and helps us learn new things. I guess some of the information will be completely useless and some could, eventually, be world changing.

I used Google Gemini 2.5 Pro Preview 05-06 to create the script based on the [Absolute Zero](https://arxiv.org/abs/2505.03335) research paper.

I'm also sure that some VERY smart people, people like you, will come along and update this script to make it even more awesome.

## Backend

I'm using [Novita.ai](https://novita.ai/) to get access to DeekSeek R1, however it's possible to use any provider. I thought using R1 would be good since it's the largest OSS LLM available at the moment.

With a few changes, I think it'd be possible to make the script more modular so that it gets easier to use any provider and model.

## Description

### What it Does

The "Absolute Zero Universal Knowledge Generator" (AZR-UKG) is a Python script that implements a paradigm to:
1.  **Generate Unconventional Tasks**: It formulates complex, abstract, and often esoteric tasks or questions that a human might not typically conceive. These can span any field of knowledge.
2.  **Attempt LLM-based Solutions**: It then employs Large Language Models (LLMs) to try and answer these generated tasks.
3.  **Explore AI Capabilities**: The primary goal is to probe the boundaries of LLM knowledge, reasoning, and creative problem-solving abilities, particularly with highly novel or abstract prompts.

### How it Works

The script operates through a sophisticated multi-stage process involving several LLM roles:

*   **Task Generation (Proposer)**: An LLM acts as a "Proposer" to create tasks. The generation process is guided by a distribution of predefined `TASK_TYPE_DISTRIBUTION` (e.g., "synthesis of disparate paradigms," "generation of novel axioms and exploration," "epistemological boundary probes," "panel_discussion_challenge").
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

