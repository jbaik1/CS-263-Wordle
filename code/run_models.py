import argparse
import re, json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from wordle import Wordle
from prompting import Prompt_Helper

def save_conversations(conversations, outfile):
    print("outfile:", outfile)
    with open(outfile, "w") as f:
        json.dump(conversations, f, indent=4)

def extract_json_guess(json_string):
    # Example
    # {
    #     "turn_type" : "Guess",
    #     "guess" : "?", 
    #     "reason" : "Random initial guess"
    # }
    guess_idx = json_string.find("guess")
    remaining = json_string[guess_idx+len('guess'):]
    end = remaining.find("\n")
    guess = remaining[:end]
    guess = re.sub(r"[^a-zA-Z]+","", guess)

    if len(guess) != 5:
        return json_string
    
    return guess

def extract_guess(string):
    if len(string) == 5:
        # Extracted Guess (Method 1)
        return string
    elif string[0] == '{':
        # Extracted Guess (Method 3):
        return extract_json_guess(string)
    else:
        # Extracted Guess (Method 2)
        pattern = r'[A-Z]{5}'
        return re.findall(pattern, string)[-1]

def merge_consecutive_roles(json_data):
    merged_data = [json_data[0]]
    
    i = 1
    while i < len(json_data):
        current_dict = json_data[i]
        current_role = current_dict['role']
        current_content = current_dict['content']
        
        if current_role == merged_data[-1]['role']:
            merged_data[-1]['content'] += f'\n{current_content}'
        else:
            merged_data.append(current_dict)
        i += 1
    
    return merged_data

def main(args):
    if args.use_vllm:
        print("Using vLLM!")
        gpu_memory_utilization = 0.8

        model = LLM(
            model=args.model, 
            tokenizer=args.model, 
            tensor_parallel_size=torch.cuda.device_count(), 
            dtype='bfloat16', 
            trust_remote_code=True, 
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        tokenizer = model.get_tokenizer()
    else:
        # Configure the model and quantization
        if args.load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif args.load_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto"
        )

    # Set pad token to eos token for tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id

    p = Prompt_Helper()
    if args.shots == 0:
        starting_conversation = p.get_zero_shot_instructions()
    else:
        starting_conversation = p.load_few_shot_examples(shots=args.shots)
    
    # print(starting_conversation)

    
    conversations = {}
    for game_idx in tqdm(range(args.num_games)):
        game = Wordle()
        conversation = starting_conversation.copy()

        victory = False
        for turn_idx in range(args.max_turns):
            if args.model == "meta-llama/Llama-2-13b-chat-hf":
                ## Llama 2 throws an error if there are two consecutive contents from the same role
                conversation = merge_consecutive_roles(conversation)
            
            if args.use_vllm:
                stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "### Instruction"]
                sampling_params = SamplingParams(
                    temperature=0.3, 
                    top_p=0.9, 
                    max_tokens=500, 
                    stop=stop_tokens)
                inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                outputs = model.generate(inputs, sampling_params)
                response_str = outputs[0].outputs[0].text
            else:
                input_ids = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                    
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=500,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )

                # process model output
                response = outputs[0][input_ids.shape[-1]:]
                response_str = tokenizer.decode(response, skip_special_tokens=True)

            # print("raw output:", response_str)

            # Evaluate model output and give feedback
            guess_eval = ""
            try:
                guess_str = extract_guess(response_str)
                guess_eval = game.turn(guess_str)
                user_response = p.jb_feedback(guess_str, guess_eval, turn_idx)["feedback"]
                conversation.extend([
                    {'role': 'assistant', 'content': response_str},
                    {'role': 'user', 'content': user_response}
                ])
            except ValueError as error:
                print("VALUE ERROR GAME:", error)
                print(response_str)
                break
            except IndexError as error:
                print("INDEX ERROR GAME:", error)
                print(response_str)
                break


            if guess_eval == 'GGGGG':
                conversation.append({'role': 'system', 'content': f"Correct! The answer is {game.get_answer()}"})
                victory = True
                break

            # print(conversation[-1])

        if not victory:
            conversation.append({'role': 'system', 'content': f"{args.max_turns} guesses exhausted! The correct answer is {game.get_answer()}"})
        conversations[game_idx] = conversation

    if args.outfile:
        model_name = args.model
    else:
        if args.model == "meta-llama/Meta-Llama-3-8B-Instruct":
            model_name = "llama_3"
        elif args.model == "meta-llama/Llama-2-13b-chat-hf":
            model_name = "llama_2"
        else:
            model_name = "notllama"
    outfile = f"outputs/conversations_{model_name}_{args.shots}shot_{args.max_turns}_turns_{args.num_games}games.json"
    save_conversations(conversations, outfile)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Wordle game or tests based on the language model")
    parser.add_argument("--model", help="HF model to run", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_games", help="the number of games to run", type=int, default=50)
    parser.add_argument("--shots", help="zero-shot, 1-shot, 2-shot, ... up to 4-shot", type=int, default=4)
    parser.add_argument("--max_turns", help="if you don't limit the model to 5, the amount of turns you allow it to keep guessing for", type=int, default=5)
    parser.add_argument("--outfile", help="The JSON file to which the results will be dumped")
    parser.add_argument("--use_vllm", action="store_true", help="Whether to use VLLM to speedup inference")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit quantization")
    # parser.add_argument("--test_eval_utils", action="store_true", help="Run evaluation utils test")
    # parser.add_argument("--test_phi_dataset", action="store_true", help="Run Phi dataset test")
    # parser.add_argument("--annotations_filepath", type=str, help="Path to the annotations file for Phi dataset test")
    # parser.add_argument("--prompt_type", type=str, help="Type of prompt for Phi dataset test")
    args = parser.parse_args()
    main(args)