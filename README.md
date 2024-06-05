# CS-263-Wordle

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
Examples
```bash
python3 run_models.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--num_games 50 \
--shots 4 \
--max_turns 5
```
