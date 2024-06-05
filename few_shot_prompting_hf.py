import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from wordle import Wordle
from prompting import Prompt_Helper

def run_tests(args):
    if args.test_eval_utils:
        from src.tests.test_eval_utils import evaluate_standard
        result = evaluate_standard()
        print(f"Your `evaluate_standard` function is {result}!")

    elif args.test_phi_dataset:
        from src.tests.test_phi_dataset import test_phi_prompt_dataset
        result = test_phi_prompt_dataset(args.annotations_filepath, args.prompt_type)
        print(f"Your `__getitem__` and `zero_shot_eval_prompt_transform` function in `PhiPromptDataset` is {result}!")

def main(args):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Configure the model and quantization
    llama3_8b = "meta-llama/Meta-Llama-3-8B-Instruct"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(llama3_8b, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        llama3_8b,
        quantization_config=nf4_config,
        device_map="auto"
    )

    # Set pad token to eos token for tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.test_eval_utils or args.test_phi_dataset:
        run_tests(args)
    else:
        # Initialize the Wordle game and prompting helper
        game = Wordle()
        p = Prompt_Helper()

        # Get a few-shot instruction and begin interaction loop
        conversation = p.get_few_shot_instructions(2)
        play_wordle(game, p, model, tokenizer, conversation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Wordle game or tests based on the language model")
    parser.add_argument("--test_eval_utils", action="store_true", help="Run evaluation utils test")
    parser.add_argument("--test_phi_dataset", action="store_true", help="Run Phi dataset test")
    parser.add_argument("--annotations_filepath", type=str, help="Path to the annotations file for Phi dataset test")
    parser.add_argument("--prompt_type", type=str, help="Type of prompt for Phi dataset test")
    args = parser.parse_args()
    main(args)
