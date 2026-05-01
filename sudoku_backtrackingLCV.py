# Sudoku Solver using Backtracking with Least Constraining Value (LCV)

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


#this function is to check whether the placement of a number in the empty cell is a valid placement or not
def isValidPlacement(board, row, col, number):

    #this is to check if the number is already present in the same row
    for c in range(9):
        if board[row][c] == number:
            return False

    #this is to check if the number is already present in the same column
    for r in range(9):
        if board[r][col] == number:
            return False

    #this is to find the coordinates of the subgrid
    row_box = (row // 3)
    col_box = (col // 3)

    #this is to check if the number is present in the subgrid
    for r in range(row_box * 3, (row_box * 3) + 3):
        for c in range(col_box * 3, (col_box * 3) + 3):
            if board[r][c] == number:
                return False

    #if all the conditions are met, then it returns True which means it is a valid placement
    return True


#this function gets all neighboring cells (same row, column, and box)
def get_neighbors(row, col):
    neighbors = set()

    for i in range(9):
        if i != col:
            neighbors.add((row, i))
        if i != row:
            neighbors.add((i, col))

    row_box = (row // 3) * 3
    col_box = (col // 3) * 3

    for r in range(row_box, row_box + 3):
        for c in range(col_box, col_box + 3):
            if (r, c) != (row, col):
                neighbors.add((r, c))

    return neighbors


#this function orders values using the least constraining value heuristic
def order_values(board, row, col):

    values = []

    #try all numbers from 1-9
    for number in range(1, 10):
        if isValidPlacement(board, row, col, number):

            impact = 0

            #LCV logic follows
            #for each valid number, we measure how "constraining" it is
            #we do this by counting how many options it would eliminate from neighboring cells
            for (r, c) in get_neighbors(row, col):
                if board[r][c] == 0:
                    #if this number could also go in the neighbor, placing it here removes that option
                    if isValidPlacement(board, r, c, number):
                        impact += 1

            #we store (number, impact score)
            values.append((number, impact))

    #sort values so that the number with the LEAST impact comes first
    values.sort(key=lambda x: x[1])

    #return just the numbers in the new LCV order
    return [num for num, _ in values]


#this is the implementation of the backtracking algorithm with LCV
def solve(board):
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

    #LCV used here
    #instead of trying numbers from 1-9 in order,
    #we now try numbers in LCV order (least constraining first)
    for number in order_values(board, row, col):

        #this counts each temporary assignment made during search
        assignments += 1

        #we set that coordinate to equal the number temporarily
        board[row][col] = number

        #we use recursion to solve the remaining board
        if solve(board):
            return True

        #we set that cell equal to 0 if that number does not give a valid solution
        board[row][col] = 0

    #we return false in the case no number works and we continue backtracking
    return False


#this function runs the LCV solver on all puzzles in the csv file
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

        #this resets the counters before solving each new puzzle
        recursive_calls = 0
        assignments = 0

        #this records the start time
        start_time = time.time()

        #this solves the sudoku board using backtracking with LCV
        solved = solve(board)

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











