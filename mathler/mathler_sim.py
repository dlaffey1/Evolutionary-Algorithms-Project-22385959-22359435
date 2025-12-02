import random

digits = ["0","1","2","3","4","5","6","7","8","9"]
ops = ["+","-","*","/"]

'''
Grammar rules for valid functions:
- no division by zero
- no leading zeros
- no leading operators

grammar is max length 6

first character has to be a non-zero digit

A -> int[1-9] N
N -> int[0-9] N | ["+","-","*"] M | epsilon (only to be chosen after 6 characters)
M -> int[1-9] N 

'''

def generate_secret_function():
    secret = [] # 6-character function
    secret.append(random.choice(digits[1:]))  # First character: non-zero digit
    for i in range(5):
        if secret[-1] in ops:
            secret.append(random.choice(digits[1:]))  # Avoid leading zeros after an operator
        else:
            choice = random.choices(
                population=['digit', 'operator'],
                weights=[0.6, 0.4] if i < 4 else [1.0, 0.0],
                k=1
            )[0]
            if choice == 'digit':
                secret.append(random.choice(digits))
            elif choice == 'operator':
                secret.append(random.choice(ops))
            else:
                break
    return ''.join(secret)

def make_valid_function():
    secret_func_made = False
    fails=0
    while not secret_func_made:
        func = generate_secret_function()
        if type(eval(func)) != int:
            fails+=1
            continue
        elif sum(1 for ch in func if ch in ops) < 2:
            fails+=1
            continue
        elif eval(func)== 0:
            fails+=1
            continue
        else:
            secret_function = func
            secret_func_made = True
    return secret_function

def evaluate_guess(secret, guess):
    results = []
    for i in range(6):
        if guess[i] == secret[i]:
            results.append(1)
        elif guess[i] in secret:
            results.append(-1)
        else:
            results.append(0)
    return results

print("HERE IS GAMEPLAY")
game_secret = make_valid_function()
target_value = eval(game_secret)
print("value of secret function: ", target_value)  # For testing purposes only
incorrect = True
while incorrect:
    user_guess = input("Enter your 6-character function guess: ")
    if len(user_guess) != 6:
        print("Guess must be 6 characters long.")
        continue
    try:
        evaluated_guess = eval(user_guess)
    except Exception as e:
        print("Invalid expression:", e)
        continue
    if evaluated_guess != target_value:
        print("Guess must evaluate to the target!")
        continue
    feedback = evaluate_guess(game_secret, user_guess)
    print("Feedback:", feedback)
    if all(f == 1 for f in feedback):
        incorrect = False
        print("Congratulations! You've guessed the secret function:", game_secret)
    
