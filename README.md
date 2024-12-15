# CS-263-Wordle
See "Solving Wordle with LLMs.pdf" for the full report.

## Environment setup
```bash
conda create -n wordle python=3.10
conda activate wordle
pip install -r requirements.txt
```

### HuggingFace models
Make sure you request the access of Llama3 in this [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

Then login into HuggingFace with your *read* CLI
```bash
huggingface-cli login
```

### Code Executions:
- 4-bit quantization
```bash
python3 run_models.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--num_games 50 \
--shots 4 \
--max_turns 5 \
--load_4bit
```

- 8-bit quantization
```bash
python3 run_models.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--num_games 50 \
--shots 4 \
--max_turns 5 \
--load_8bit
```

- vLLM
```bash
python3 run_models.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--num_games 50 \
--shots 4 \
--max_turns 5 \
--use_vllm
```
### File Contents
A brief overview of the files in the repository:
1. Wordle logic is contained in wordle.py, using words from words.txt
2. Prompting is contained in prompting.py, with several Jupyter notebooks for individual experiments
3. Experiments can be run using run_models.py, and the results are generally stored in the outputs folder
4. Outputs are then analyzed using eval.py and eval.ipynb
