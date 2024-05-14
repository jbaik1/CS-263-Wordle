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
        



def eval_entropy(guesses=[], answer):
    # quality of guess using entropy
    # https://www.youtube.com/watch?v=v68zYyaEmEA
    # unlikely that the llm considers this 
    # maybe not worth it, given that it's kinda hard to program this
    
    return