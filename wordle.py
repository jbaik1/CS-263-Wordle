import random

def wordle_init():
    word_list = open("words.txt", 'r')
    words = word_list.read().split()
    answer = random.choice(words)

    return answer

def output(guess_word, answer):
    # given guess, return an array of 'white', 'yellow', and 'green' colors
    # white is incorrect, replacing grey so that it doesn't overlap w green
    if len(guess_word) != len(answer): # not 5 letters
        return 'length error'
    
    out = []
    # todo: handle logic for multiple yellows
    for i in range(len(answer)):
        if guess_word[i] == answer[i]:
            out.append('G')
        elif guess_word[i] != answer and guess_word[i] in answer: # letter in word but misplaced
            out.append('Y')
        else:
            out.append('W')
    return out

if __name__ == "__main__":
    print('not functioning correctly yet')
    answer = wordle_init()
    max_guess_count = 5
    
    #todo: api for input
    for i in range(max_guess_count):
        guess = input('guess '+ str(i)+'\n')
        o = output(guess, answer)
        print(o)
        if o == ['G', 'G', 'G', 'G', 'G']:
            print('correct guess!')
            break
    print("couldn't get it in 5 guesses, answer was:", answer)


