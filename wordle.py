import random
from collections import Counter

class Wordle():
    def __init__(self, idx=-1) -> None:
        word_list = open("words.txt", 'r')
        self.words = word_list.read().split()
        # we keep everything lowercase for now
        # we can change it later by explicitly telling the model to predict captical letters
        if 0 <= idx < len(self.words):
            self.answer = self.words[idx].lower()
        else:
            self.answer = random.choice(self.words).lower()
    
    def get_answer(self):
        return self.answer

    def set_answer(self, new_answer):
        self.answer = new_answer.lower()

    def turn(self, guess):
        # given guess, return an array of 'white', 'yellow', and 'green' colors
        # white is incorrect, replacing grey so that it doesn't overlap w green
        answer = self.answer
        # we keep everything lowercase for now
        # we can change it later by explicitly telling the model to predict captical letters
        guess = guess.lower()

        if len(guess) != len(answer): # not 5 letters
            raise ValueError(f"Your guess '{guess}' does not have {len(answer)} letters.")
        
        # avoid counting the same letter more than once
        letter_count = Counter(answer)
        
        out = []
        for i in range(len(answer)):
            letter_guess = guess[i]
            letter_answer = answer[i]

            if letter_guess == letter_answer:
                letter_count[letter_answer] -= 1
                out.append('G')
            elif (letter_guess in answer) and (letter_count[letter_guess] > 0): # letter in word but misplaced
                letter_count[letter_guess] -= 1
                out.append('Y')
            else:
                out.append('W')
        return ''.join(out)

    def answer_series(self, num_random=3):
        # for few shot learning. 
        # not sure to do words that are close or randomly sampled
        answers = random.choices(self.words, k=num_random)
        answers = [w.lower() for w in answers]
        return  answers
        

if __name__ == "__main__":
    wordle = Wordle()
    max_guess_count = 5
    
    #todo: api for input
    for i in range(max_guess_count):
        guess = input('guess '+ str(i)+'\n')
        o = wordle.turn(guess)
        print(o)
        if o == ['G', 'G', 'G', 'G', 'G']:
            print('correct guess!')
            break
    print("couldn't get it in 5 guesses, answer was:", wordle.get_answer())


