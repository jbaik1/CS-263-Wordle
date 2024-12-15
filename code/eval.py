from wordle import Wordle
import re, random
import numpy as np
from collections import defaultdict, Counter

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
    

    return -1 # never guessed it correctly


def eval_num_correct_letters(guesses, answer):
    # how many letters in the final guess are correct
    if len(guesses) == 0:
        print("In eval_num_correct_letters: list of guesses is empty")
        return 0
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


def eval_wyg(guesses, answer):
    count_g = 0
    count_y = 0
    count_w = 0
    letter_count = Counter(answer)
    N = len(guesses)

    for guess in guesses:
        for idx in range(len(answer)):
            letter_guess = guess[idx]
            letter_answer = answer[idx]
            
            if letter_guess == letter_answer:
                letter_count[letter_answer] -= 1
                count_g += 1
            elif (letter_guess in answer) and (letter_count[letter_guess] > 0): # letter in word but misplaced
                letter_count[letter_answer] -= 1
                count_y += 1
            else:
                count_w += 1

    assert(N * 5 == count_g + count_y + count_w)
    return count_g, count_y, count_w


def eval_info_gain(guesses, answer, word_list):
    ''' 
    returns counts relating to prevalence of not leveraging feedback from previous guesses
    
    Counting 3 types of errors:
    1. 'Wrong Letter': The model uses a letter in the guess that it should know is not in the word
    2. 'Wrong Pos': The model puts a letter in a position that it should know is not right
    3. 'Undo Correct': The model guessed something other than the correct letter in the position it already figured out the correct letter for.
    '''
    wrong_letter_count, wrong_pos_count, undo_correct_count = 0,0,0
    wrong_letter_random, wrong_pos_random, undo_correct_random = 0,0,0
    wrong_letter_denom, wrong_pos_denom, undo_correct_denom = 0,0,0

    letters_known_not_in_word = []
    pos_to_right_letter = defaultdict(list)
    letter_to_wrong_pos = defaultdict(list)

    for guess in guesses:
        # Count the wrong letters in the current guess
        # that have already been shown in previous guesses

        # for each guess, this is the information they have
        wrong_letter_denom += len(letters_known_not_in_word) # number of letters it shouldnt guess
        wrong_pos_denom += sum(len(lst) for lst in letter_to_wrong_pos.values()) # numbers of (letter, position) pairs it shouldn't guess
        undo_correct_denom += len(pos_to_right_letter.keys()) # number of correct (letter, position) pairs it should guess
        
        wrong_pos_count = 0
        random_guess = random.sample(word_list, 1)[0].lower()
        for letter_idx, letter in enumerate(guess):
            random_letter = random_guess[letter_idx]
            if letter_idx in pos_to_right_letter.keys():
                if letter != pos_to_right_letter[letter_idx]:
                    # implies model should already know letter in this spot
                    undo_correct_count += 1
                if random_letter != pos_to_right_letter[letter_idx]:
                    undo_correct_random += 1

            if letter not in answer:
                if letter in letters_known_not_in_word:
                    wrong_letter_count += 1
                else:
                    letters_known_not_in_word.append(letter)
            elif letter != answer[letter_idx]:
                if letter in letter_to_wrong_pos.keys():
                    if letter_idx in letter_to_wrong_pos[letter]:
                        wrong_pos_count += 1
                    else:
                        letter_to_wrong_pos[letter].append(letter_idx)
                else:
                    letter_to_wrong_pos[letter].append(letter_idx)
            elif letter == answer[letter_idx]:
                pos_to_right_letter[letter_idx] = letter
            
            if random_letter not in answer:
                if random_letter in letters_known_not_in_word:
                    wrong_letter_random += 1
            elif random_letter != answer[letter_idx]:
                if random_letter in letter_to_wrong_pos.keys():
                    if letter_idx in letter_to_wrong_pos[random_letter]:
                        wrong_pos_random += 1


    return [[wrong_letter_count, wrong_letter_denom, wrong_letter_random], 
            [wrong_pos_count, wrong_pos_denom, wrong_pos_random], 
            [undo_correct_count, undo_correct_denom, undo_correct_random]]  


class Conversation():
    def __init__(self, conversation) -> None:
        self.raw_convo = conversation
        self.length = len(self.raw_convo)
        self.correct_answer = self.extract_correct_answer()
        self.guesses = self.extract_guesses()

        self.wordle = Wordle()
        self.wordle.set_answer(self.correct_answer)
        

    def extract_correct_answer(self):
        # print(self.raw_convo)
        last = self.raw_convo[-1]['content'].split()
        # print(last)

        if last[-3:-1] == ['answer', 'is']:
            return last[-1].lower()
        elif last[-3:-1] == ['correct', 'word']:
            return last[-1].lower()
        else:
            print(last)
            raise("what other edge cases exist ?")

    def extract_guesses(self):
        guesses = []
        still_in_prompt = True
        for i in range(len(self.raw_convo)):
            if still_in_prompt and (self.raw_convo[i]['role'] == 'user') and ("let's play another game." in self.raw_convo[i]['content']):
                still_in_prompt = False
                self.end_of_prompt = i
            if not still_in_prompt and self.raw_convo[i]['role'] == 'assistant':
                pattern = r'[A-Z]{5}'
                guess = re.findall(pattern, self.raw_convo[i]['content'])[-1]
                guesses.append(guess.lower())
        
        # if len(guesses) > 5:
        #     raise("too many guesses")
        return guesses
        
def get_aggregate_scores(conversation_list):
    GYW = np.zeros(3)
    info_gain_stats = np.zeros((3, 3))
    num_guesses = []
    end_letter_level_acc = []
    with open("words.txt", 'r') as f:
        word_list = f.read().split()
    
    for key, convo in conversation_list.items():
        c = Conversation(convo)
        # don't need to have accuracy cause this essentially replaces it ?
        guessnum = eval_num_guesses(c.guesses, c.correct_answer)
        if guessnum > 0:
            num_guesses.append(guessnum)
        GYW = GYW + np.array(eval_wyg(c.guesses, c.correct_answer)) #element-wise addition
        info_gain_stats = info_gain_stats + np.array(eval_info_gain(c.guesses, c.correct_answer, word_list)) #element-wise addition
        end_letter_level_acc.append(eval_num_correct_letters(c.guesses, c.correct_answer))

    G, Y, W = GYW[0], GYW[1], GYW[2]
    correct_pct = (G) / (G + Y + W)
    correct_letter_pct = (G + Y) / (G + Y + W)

    wrong_letter_prevalence = info_gain_stats[0][0] / info_gain_stats[0][1]
    wrong_pos_prevalence = info_gain_stats[1][0] / info_gain_stats[1][1]
    undo_correct_prevalence = info_gain_stats[2][0] / info_gain_stats[2][1]
    wrong_letter_random = info_gain_stats[0][2] / info_gain_stats[0][1]
    wrong_pos_random = info_gain_stats[1][2] / info_gain_stats[1][1]
    undo_correct_random = info_gain_stats[2][2] / info_gain_stats[2][1]

    end_letter_level_acc = sum(end_letter_level_acc) / len(conversation_list)
    avg_number_of_guesses = sum(num_guesses) / len(conversation_list)

    print(f"Correct percentage: {correct_pct}")
    print(f"Correct letter percentage: {correct_letter_pct}")
    print(f"Wrong letter prevalence: {wrong_letter_prevalence}")
    print(f"Wrong pos prevalence: {wrong_pos_prevalence}")
    print(f"Undo correct prevalence: {undo_correct_prevalence}")
    print(f"Wrong letter random: {wrong_letter_random}")
    print(f"Wrong pos random: {wrong_pos_random}")
    print(f"Undo correct random: {undo_correct_random}")
    print(f"End letter level acc: {end_letter_level_acc}")
    print(f"Average number of guesses: {avg_number_of_guesses}")


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