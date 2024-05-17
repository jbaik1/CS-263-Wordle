from wordle import Wordle
import re

# functions for evaluating performances
#
# input is for 1 "round". Can make it more general if necessary
# ex. input: guesses = ["cover", "comet", "apple", ...], answer = "avert"
#
# tried to make it so that big = good, but doesn't have to be if there are other good metrics

# some simple evaluations
def eval_num_guesses(guesses, answer):
    # how many guesses did it take to get the answer
    for i in range(len(guesses)):
        if guesses[i] == answer:
            return i+1
    
    return float('inf') # some large number if it never guessed it correctly

def eval_num_correct_letters(guesses, answer):
    final = guesses[-1]
    
    num_correct = 0
    for i in range(len(answer)):
        if final[i] == answer[i]:
            num_correct += 1
            
    return num_correct
    


def eval_avg_acc(guesses, answer):
    # how many letters are correct in each guess
    # now average this over the whole guess history
    # model may not get the answer within the guess limit
    N = len(guesses)
    acc = []
    
    for guess in guesses:
        num_correct = 0
        for i in range(N):
            if guess[i] == answer[i]:
                num_correct += 1
        acc.append(num_correct/N)
    
    average_acc = sum(acc)/N
    
    return average_acc
        



def eval_entropy(answer, guesses=[]):
    # quality of guess using entropy
    # https://www.youtube.com/watch?v=v68zYyaEmEA
    # unlikely that the llm considers this 
    # maybe not worth it, given that it's kinda hard to program this
    
    return

class Conversation():
    def __init__(self, conversation) -> None:
        self.raw_convo = conversation
        self.length = len(self.raw_convo)
        self.correct_answer = self.extract_correct_answer()
        self.guesses = self.extract_guesses()

        self.wordle = Wordle()
        self.wordle.set_answer(self.correct_answer)
        

    def extract_correct_answer(self):
        last = self.raw_convo[-1]['content'].split()
        if last[-3:-1] == ['answer', 'is']:
            return last[-1]
        else:
            raise("what other edge cases exist ?")

    def extract_guesses(self):
        guesses = []
        still_in_prompt = True
        for i in range(len(self.raw_convo)):
            if still_in_prompt and (self.raw_convo[i]['role'] == 'system') and (self.raw_convo[i]['content'] == 'Now it is your turn'):
                still_in_prompt = False
                self.end_of_prompt = i
            if not still_in_prompt and self.raw_convo[i]['role'] == 'assistant':
                pattern = r'[A-Z]{5}'
                guess = re.findall(pattern, self.raw_convo[i]['content'])[-1]
                guesses.append(guess)
        
        if len(guesses) > 5:
            raise("too many guesses")
        return guesses
        
    


def main():
    example_convo = [
        {'role': 'system', 'content': ('Wordle is a 5 letter word guessing game. Each letter in a guess will either be a correct letter in correct position, an invalid letter or a correct letter present in the word but in a wrong position. After each guess, we get a result',)},
        {'role': 'system', 'content': 'Example #1:'},
        {'role': 'user', 'content': 'Enter guess #1'},
        {'role': 'assistant', 'content': 'BAGEL'},
        {'role': 'user', 'content': 'Letter B is not anywhere present in the word, Letter A is not anywhere present in the word, Letter G is not anywhere present in the word, Letter E at Position 3 is in the word and at the correct position, Letter L is not anywhere present in the word, '},
        {'role': 'user', 'content': 'Enter guess #2'},
        {'role': 'assistant', 'content': 'COVER'},
        {'role': 'user', 'content': 'Letter C is not anywhere present in the word, Letter O is not anywhere present in the word, Letter V is not anywhere present in the word, Letter E at Position 3 is in the word and at the correct position, Letter R at Position 4 is in the wrong position but is in the word, '},
        {'role': 'user', 'content': 'Enter guess #3'},
        {'role': 'assistant', 'content': 'THREE'},
        {'role': 'user', 'content': 'Congrats you found the correct word THREE in 3 guesses'},
        {'role': 'system', 'content': 'Example #1:'},
        {'role': 'user', 'content': 'Enter guess #1'},
        {'role': 'assistant', 'content': 'WHILE'},
        {'role': 'user', 'content': 'Letter W is not anywhere present in the word, Letter H is not anywhere present in the word, Letter I is not anywhere present in the word, Letter L is not anywhere present in the word, Letter E is not anywhere present in the word, '},
        {'role': 'user', 'content': 'Enter guess #2'},
        {'role': 'assistant', 'content': 'RADIO'},
        {'role': 'user', 'content': 'Letter R is not anywhere present in the word, Letter A at Position 1 is in the wrong position but is in the word, Letter D is not anywhere present in the word, Letter I is not anywhere present in the word, Letter O is not anywhere present in the word, '},
        {'role': 'user', 'content': 'Enter guess #3'},
        {'role': 'assistant', 'content': 'START'},
        {'role': 'user', 'content': 'Letter S at Position 0 is in the word and at the correct position, Letter T at Position 1 is in the word and at the correct position, Letter A at Position 2 is in the word and at the correct position, Letter R is not anywhere present in the word, Letter T is not anywhere present in the word, '},
        {'role': 'user', 'content': 'Enter guess #4'},
        {'role': 'assistant', 'content': 'STAGE'},
        {'role': 'user', 'content': 'Letter S at Position 0 is in the word and at the correct position, Letter T at Position 1 is in the word and at the correct position, Letter A at Position 2 is in the word and at the correct position, Letter G is not anywhere present in the word, Letter E is not anywhere present in the word, '},
        {'role': 'user', 'content': 'Enter guess #5'},
        {'role': 'assistant', 'content': 'STAFF'},
        {'role': 'user', 'content': 'Congrats you found the correct word STAFF in 5 guesses'},
        {'role': 'system', 'content': 'Now it is your turn'},
        {'role': 'user', 'content': 'Enter guess #1'},
        {'role': 'assistant', 'content': 'BAGEL'},
        {'role': 'user', 'content': 'Letter B is not anywhere present in the word, Letter A at Position 1 is in the word and at the correct position, Letter G is not anywhere present in the word, Letter E at Position 3 is in the wrong position but is in the word, Letter L is not anywhere present in the word, '},
        {'role': 'assistant', 'content': 'BAGEL'},
        {'role': 'user', 'content': 'Letter B is not anywhere present in the word, Letter A at Position 1 is in the word and at the correct position, Letter G is not anywhere present in the word, Letter E at Position 3 is in the wrong position but is in the word, Letter L is not anywhere present in the word, '},
        {'role': 'assistant', 'content': 'COVER'},
        {'role': 'user', 'content': 'Letter C is not anywhere present in the word, Letter O is not anywhere present in the word, Letter V is not anywhere present in the word, Letter E at Position 3 is in the wrong position but is in the word, Letter R at Position 4 is in the wrong position but is in the word, '},
        {'role': 'assistant', 'content': 'COVER'},
        {'role': 'user', 'content': 'Letter C is not anywhere present in the word, Letter O is not anywhere present in the word, Letter V is not anywhere present in the word, Letter E at Position 3 is in the wrong position but is in the word, Letter R at Position 4 is in the wrong position but is in the word, '},
        {'role': 'assistant', 'content': 'THREE'},
        {'role': 'user', 'content': 'Letter T is not anywhere present in the word, Letter H is not anywhere present in the word, Letter R at Position 2 is in the wrong position but is in the word, Letter E at Position 3 is in the wrong position but is in the word, Letter E at Position 4 is in the word and at the correct position, '},
        {'role': 'system', 'content': '5 guesses exhausted! The correct answer is raise'}
    ]
    c = Conversation(example_convo)
    print(c.correct_answer, c.guesses)

if __name__ == "__main__":
    main()