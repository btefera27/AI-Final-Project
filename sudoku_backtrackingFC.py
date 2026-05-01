# Sudoku Solver using Backtracking with Forward Checking

import pandas as pd
import time

#these variables are to measure the search effort
recursive_calls = 0
assignments = 0


#this function is to convert the puzzle string from the csv file into a 9x9 sudoku board
def string_to_board(puzzle):
    board = []

    for i in range(0, 81, 9):
        row = [int(x) for x in puzzle[i:i+9]]
        board.append(row)

    return board


#this function is to find the empty cells in the puzzle. The empty cells are represented by 0
def empty_cell_finder(board):

    #this is to iterate through each row
    for row in range(9):
        #this is to iterate through each column
        for col in range(9):
            #this is to check if the cell is empty
            if board[row][col] == 0:
                #return the coordinates of that empty cell
                return row, col

    #this happens in the case when the board is solved and there are no empty cells remaining
    return None


#this function is to initialize the domains for each cell
#domains store the possible values each cell can take
def initialize_domains(board):
    domains = {}

    for row in range(9):
        for col in range(9):
            #if the cell is empty, it can take values from 1-9
            if board[row][col] == 0:
                domains[(row, col)] = set(range(1, 10))
            else:
                #if the cell is already filled, its domain is just that value
                domains[(row, col)] = {board[row][col]}

    return domains


#this function gets all neighboring cells (same row, column, and box)
def get_neighbors(row, col):
    neighbors = set()

    #same row and column
    for i in range(9):
        if i != col:
            neighbors.add((row, i))
        if i != row:
            neighbors.add((i, col))

    #same 3x3 box
    row_box = (row // 3) * 3
    col_box = (col // 3) * 3

    for r in range(row_box, row_box + 3):
        for c in range(col_box, col_box + 3):
            if (r, c) != (row, col):
                neighbors.add((r, c))

    return neighbors


#this function performs forward checking by removing invalid values from neighbors
def forward_check(domains, row, col, number):

    #check all neighboring cells
    for (r, c) in get_neighbors(row, col):

        #if the number exists in the neighbor's domain, remove it
        if number in domains[(r, c)]:
            domains[(r, c)].discard(number)

            #if any neighbor has no possible values left, return False
            if len(domains[(r, c)]) == 0:
                return False

    #if no issues, return True
    return True


#this is the implementation of the backtracking algorithm with forward checking
def solve(board, domains):
    global recursive_calls
    global assignments

    #this counts each time the solve function is called
    recursive_calls += 1

    #this is to find the next empty cell
    emptycell = empty_cell_finder(board)

    #this is to return solved in the case there are no empty cells remaining
    if emptycell is None:
        return True
    else:
        row, col = emptycell

    #try numbers from the domain of that cell
    for number in domains[(row, col)].copy():

        #this counts each temporary assignment made during search
        assignments += 1

        #we set that coordinate to equal the number temporarily
        board[row][col] = number

        #we copy the domains so we don't mess up other recursive calls
        new_domains = {k: v.copy() for k, v in domains.items()}
        new_domains[(row, col)] = {number}

        #we apply forward checking
        if forward_check(new_domains, row, col, number):

            #we use recursion to solve the remaining board
            if solve(board, new_domains):
                return True

        #we set that cell equal to 0 if that number does not give a valid solution
        board[row][col] = 0

    #we return false in the case no number works and we continue backtracking
    return False


#this function runs the forward checking solver on all puzzles in the csv file
def run_experiment(csv_file):
    global recursive_calls
    global assignments

    #this reads the dataset from the csv file
    df = pd.read_csv(csv_file)

    #this keeps only the easy, medium, and hard puzzles
    df = df[df["difficulty"].isin(["easy", "medium", "hard"])]

    #this list stores the result for each puzzle
    results = []

    #this loop goes through each puzzle in the dataset
    for index, row in df.iterrows():

        #this gets the puzzle string from the csv file
        puzzle = row["puzzle"]

        #this gets the difficulty level of the puzzle
        difficulty = row["difficulty"]

        #this converts the puzzle string into a 9x9 board
        board = string_to_board(puzzle)

        #this initializes the domains for the puzzle
        domains = initialize_domains(board)

        #this resets the counters before solving each new puzzle
        recursive_calls = 0
        assignments = 0

        #this records the start time
        start_time = time.time()

        #this solves the sudoku board using backtracking with forward checking
        solved = solve(board, domains)

        #this records the end time
        end_time = time.time()

        #this stores the results for this puzzle
        results.append({
            "puzzle_number": index,
            "difficulty": difficulty,
            "solved": solved,
            "recursive_calls": recursive_calls,
            "assignments": assignments,
            "runtime_seconds": end_time - start_time
        })

    #this converts all results into a pandas dataframe
    return pd.DataFrame(results)


#this runs the experiment on the dataset
results_df = run_experiment("dataset.csv")

#this prints the result for each individual puzzle
print("Individual Puzzle Results:")
print(results_df)

#this calculates the average results for each difficulty level
summary = results_df.groupby("difficulty").agg({
    "solved": "mean",
    "recursive_calls": "mean",
    "assignments": "mean",
    "runtime_seconds": "mean"
})

#this prints the average results by difficulty
print("\nAverage Results by Difficulty:")
print(summary)
