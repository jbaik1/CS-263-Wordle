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
Run the simulator specifying a custom model and output file:
```
python3 run_prompts.py --model "your-model-path" --num_games 100 --outfile "results.json"
```

Limit the number of turns per game to 5, using a zero-shot strategy:
```
python3 run_prompts.py --shots 0 --max_turns 5
```

Run the simulator specifying a custom model and output file:

```
python3 run_prompts.py --model "your-model-path" --num_games 100 --output_path "results.json"
```



