import pandas as pd
import random

df = pd.read_csv('sudoku_cluewise.csv')

# Extract puzzles by difficulty
easy = df[df['clue_numbers'] >= 36].sample(n = 50)
easy['difficulty'] = 'easy'

medium = df[(df['clue_numbers'] >= 27) & (df['clue_numbers'] < 36)].sample(n = 50)
medium['difficulty'] = 'medium'

hard = df[df['clue_numbers'] < 27].sample(n = 50)
hard['difficulty'] = 'hard'

# Create invalid puzzles
invalid = df[~df.index.isin(pd.concat([easy, medium, hard]).index)].sample(n = 50)
invalid['difficulty'] = 'invalid'

def corrupt(puzzle):
    puzzle = list(puzzle)
    
    # Select random positions to corrupt
    positions = random.sample(range(len(puzzle)), random.randint(3, 5))
    
    # Replace with random digits
    for position in positions:
        puzzle[position] = str(random.randint(0, 9))
    
    return ''.join(puzzle)

invalid['quizzes'] = invalid['quizzes'].apply(corrupt)

# Save the puzzles
puzzles = pd.concat([easy, medium, hard, invalid])
puzzles = puzzles[['quizzes', 'difficulty']]
puzzles.columns = ['puzzle', 'difficulty']
puzzles.to_csv('dataset.csv', index=False)