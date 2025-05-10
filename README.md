# Absolute Zero Universal Knowledge Generator

This script generates questions that we, as humans, would or could never ask and then tries to answer them. This is then added to a JSONL file that can be used to fine-tune other LLMs that, in turn, becomes more knowledgeable and helps us learn new things. I guess some of the information will be completely useless and some could, eventually, be world changing.

I used Google Gemini 2.5 Pro Preview to create the script based on the [Absolute Zero](https://arxiv.org/abs/2505.03335) research paper.

I'm also sure that some VERY smart people, people like you, will come along and update this script to make it even more awesome.

## Backend

I'm using [Novita.ai](https://novita.ai/) to get access to DeekSeek R1, however it's possible to use any provider. I thought using R1 would be good since it's the largest OSS LLM available at the moment.

With a few changes, I think it'd be possible to make the script more modular so that it gets easier to use any provider and model.
