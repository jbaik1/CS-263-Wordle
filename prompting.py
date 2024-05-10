from wordle import Wordle

class Prompt_Helper():
    def __init__(self):
        self.instructions = "Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position, an invalid letter or a correct letter present in the word but in a wrong position. After each guess, we get a result",
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
            }
        ]
    

    def explanatory_response(self, guess, response, guess_number):
        if response == 'GGGGG':
            return f"Congrats you found the correct word {guess} in {guess_number+1} guesses"

        human_readable_response = ""
        for i in range(5):
            if response[i] == 'G':
                human_readable_response += f"Letter {guess[i]} at Position {i} is in the word and at the correct position, "
            elif response[i] == 'Y':
                human_readable_response += f"Letter {guess[i]} at Position {i} is in the wrong position but is in the word, "
            elif response[i] == 'W':
                human_readable_response += f"Letter {guess[i]} is not anywhere present in the word, "
            else:
                raise ValueError(f"Invalid response format for {response}. Response only includes letters G, Y, or W.")
        return human_readable_response


    def get_zero_shot_instructions(self):
        message = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": "Enter your first guess in a format like COVER"},
        ]
        return message

    def get_few_shot_instructions(self, shots=2):
        if shots < 1:
            return self.create_zero_shot_prompt()
        if shots > 3:
            shots = 3

        message = [{"role": "system", "content": self.instructions},]
        for s in range(shots):
            message.append({"role": "system", "content": "Example #1:"})
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
        



prompt1 = """
Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position (G) an invalid letter (X) or a correct letter present in the word but in a wrong position (Y). After each guess, we get a result

For example:
BAGEL
XXXGX

Here E is a valid letter in the final word and in the correct position.
You will get a list of invalid letters and valid letters.
Do not use invalid letters in the guess word.
The guess word should be a valid english word.
Guess the correct 5 letter word.

Invalid letters: {{$invalid_letters}}
Valid letters: {{$valid_letters}}
Guess word:  """


prompt2 = """
Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position (G) an invalid letter (X) or a correct letter present in the word but in a wrong position (Y). After each guess, we get a result

For example:
BAGEL
XXXGX

Here E is a valid letter in the final word and in the correct position.
You will get a list of previous guesses and its result. You will also get a word that has letters in correct position with unknown letters as '-'
The guess word should be a valid english word.
Guess the correct 5 letter word.

{{$history}}

Valid letters with positions: {{$valid_letter_word}}
Guess word:  """

prompt3 = """
Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position (G) an invalid letter (X) or a correct letter present in the word but in a wrong position (Y). After each guess, we get a result

For example:
BAGEL
XXXGX

Here E is a valid letter in the final word and in the correct position.
Do not use invalid letters in the guess word.
The guess word should be a valid english word.
You will get a list of previous guesses and its result, invalid letters and valid letters.
Write your reasoning for what should be the guess word, then write the guess word in this format: GUESS->
Guess the correct 5 letter word.

{{$history}}

Invalid letters: {{$invalid_letters}}
Valid letters: {{$valid_letters}}
Reasoning:  """