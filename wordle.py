import random
from collections import Counter

class Wordle():
    def __init__(self) -> None:
        word_list = open("words.txt", 'r')
        words = word_list.read().split()
        self.answer = random.choice(words)
    
    def get_answer(self):
        return self.answer

    def set_answer(self, new_answer):
        self.answer = new_answer

    def turn(self, guess):
        # given guess, return an array of 'white', 'yellow', and 'green' colors
        # white is incorrect, replacing grey so that it doesn't overlap w green
        answer = self.answer

        if len(guess) != len(answer): # not 5 letters
            return 'length error'
        
        letter_count = Counter(answer)
        
        out = []
        # todo: handle logic for multiple yellows
        for i in range(len(answer)):
            letter_guess = guess[i]
            letter_answer = answer[i]

            if letter_guess == letter_answer:
                letter_count[letter_answer] -= 1
                out.append('G')
            elif letter_guess == letter_answer and letter_guess[i] in answer: # letter in word but misplaced
                letter_count[letter_answer] -= 1
                out.append('Y')
            else:
                out.append('W')
        return out

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


