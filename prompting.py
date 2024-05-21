from wordle import Wordle

class Prompt_Helper():
    def __init__(self):
        self.instructions = "Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position, an invalid letter or a correct letter present in the word but in a wrong position. After each guess, we get a result"
        self.examples = [
            {
                "answer": 'THREE',
                "guesses": ['BAGEL', 'COVER', 'THREE']

            },
            {
                "answer": 'STAFF',
                "guesses": ['WHILE', 'RADIO', 'START', 'STAGE', 'STAFF']
            },
            {
                "answer": 'PROWL',
                "guesses": ['TASTE', 'HOARD', 'PROVE', 'PROWL']
            },
            {
                "answer": 'NICER',
                "guesses": ['RAISE', 'THEIR', 'LIVER', 'NICER']
            }
        ]
    

    def explanatory_response(self, guess, response, guess_number):
        if response == 'GGGGG':
            return f"Congrats you found the correct word {guess} in {guess_number+1} guesses"

        human_readable_response = ""
        for i in range(5):
            if response[i] == 'G':
                human_readable_response += f"{i+1}: Letter {guess[i]} at Position {i+1} is in the word and at the correct position.\n"
            elif response[i] == 'Y':
                human_readable_response += f"{i+1}: Letter {guess[i]} at Position {i+1} is in the wrong position but is in the word.\n"
            elif response[i] == 'W':
                human_readable_response += f"{i+1}: Letter {guess[i]} is not anywhere present in the word.\n"
            else:
                raise ValueError(f"Invalid response format for {response}. Response only includes letters G, Y, or W.")
        return human_readable_response

    def jb_feedback(self, guess, response, guess_number):
        # response given from wordle object
        # guess generated by the model
        # guess_number is how many guesses the model has made
#       )  
        valid_letters = [guess[i] for i in range(5) if response[i] == 'G']
        invalid_letters = [guess[i] for i in range(5) if response[i] == 'W']


        if response == "GGGGG": 
            victory_speech = f"Congrats you found the correct word {guess} in {guess_number+1} guesses, because all letters were in the word and in the right position!"
            return {
                "feedback": victory_speech,
                "valid_letters": valid_letters,
                "invalid_letters": invalid_letters,
                "correct": True
            }
        
        feedback = f"""Feedback:
Your guess was: '{guess}'. The feedback for each letter is:
{{ 
{self.explanatory_response(guess, response, None)}
}}
Please enter your next guess:
"""  

        feedback_dict = {
            "feedback": feedback,
            "valid_letters": valid_letters,
            "invalid_letters": invalid_letters,
            "correct": False # might be helpful ??
        }

        return feedback_dict

    def get_zero_shot_instructions(self):
        message = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": "Enter your first guess in a format like COVER"},
        ]
        return message

    def get_few_shot_instructions_old(self, shots=2):
        if shots < 1:
            return self.create_zero_shot_prompt()
        if shots > 3:
            shots = 3

        message = [{"role": "system", "content": self.instructions},]
        for s in range(shots):
            message.append({"role": "system", "content": f"Example #{s}:"})
            w = Wordle()
            w.set_answer(self.examples[s]['answer'])
            for gnum in range(len(self.examples[s]['guesses'])):
                guess = self.examples[s]['guesses'][gnum]
                message.extend([
                    {"role": "user", "content": f"Enter guess #{gnum + 1}"},
                    {"role": "assistant", "content": guess},
                    {"role": "user", "content": self.explanatory_response(guess, w.turn(guess), gnum)},
                ])
        message.extend([
            {"role": "system", "content": "Now it is your turn"},
            {"role": "user", "content": f"Enter guess #1"},
        ])
        return message
    
    def load_few_shot_examples(self, shots=4):
        message = [{"role": "system", "content": self.instructions}]

        for s in range(shots):
            example = self.examples[s]
            message.append({"role": "system", "content": "Example #1:"})
            w = Wordle()
            w.set_answer(example["answer"])
            target = example["answer"]
            history = []
            valid_letters = []
            invalid_letters = []
                
            guesses = example["guesses"]
            message.append({"role": "user", "content": "Let's play a new game with a different word"})

 

            # Process first guess before giving context to next guesses
            init = guesses[0]

            init_guess_str = f"""Guess:
{{ 
    "guess: "{init}",
    "reasoning" : "
        We are starting a new game, and this is a random initial guess
        "
        
}}
            """
            message.append({"role": "assistant", "content": init_guess_str})

            init_dict = self.jb_feedback(guess=init, response=w.turn(init), guess_number=0)
            init_feedback_str = init_dict["feedback"]
            valid_letters = init_dict["valid_letters"]
            invalid_letters = init_dict["invalid_letters"]
            history.append(init)

            message.append({"role": "user", "content": init_feedback_str})

            # process rest of the examples
            for i in range(1,len(guesses)):
                prev_guess = guesses[i-1]

                prev_result = w.turn(prev_guess) # returns colors
                position_info = []
                for color in prev_result:
                    if color == "G":
                        position_info.append("in the string and in the correct position")
                    elif color == "Y":
                        position_info.append("in the string but in the wrong position")
                    elif color == "W":
                        position_info.append("not in the string")


                guess = guesses[i]
                guess_str = f"""Guess:
{{ 
    "guess: "{guess}",
    "reasoning" : "
        The list of previously guessed words are: {history}
        The list of valid letters are: {valid_letters}
        The list of invalid letters are: {invalid_letters}

        The previous guess was: {prev_guess}
        Given the previous feedback, we know that
        1. The letter at position 1 ({prev_guess[0]}) was {position_info[0]} 
        2. The letter at position 2 ({prev_guess[1]}) was {position_info[1]} 
        3. The letter at position 3 ({prev_guess[2]}) was {position_info[2]} 
        4. The letter at position 4 ({prev_guess[3]}) was {position_info[3]} 
        5. The letter at position 5 ({prev_guess[4]}) was {position_info[4]} 

        Improving on this feedback,
        the new guess is {guess}
        "
        
}}
                """
                history.append(guess)

                feedback_dict = self.jb_feedback(guess=guess, response=w.turn(guess), guess_number=i)
                feedback_str = feedback_dict["feedback"]

                new_valid_letters = feedback_dict["valid_letters"]
                new_invalid_letters = feedback_dict["invalid_letters"]
                
                valid_letters = list(set(valid_letters + new_valid_letters))
                invalid_letters = list(set(invalid_letters + new_invalid_letters))

                message.append({"role": "assistant", "content": guess_str})
                message.append({"role": "user", "content": feedback_str})

        
        message.append({"role": "user", "content": "let's play another game. Start with your random initial guess"})
        
        return message
        
        



