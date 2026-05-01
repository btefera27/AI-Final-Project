# Sudoku Solver using Backtracking Algorithm

import pandas as pd
import time

#these variables are to measure the search effort of the backtracking algorithm
recursive_calls = 0
assignments = 0

#this function is to convert the puzzle string from the csv file into a 9x9 sudoku board
def string_to_board(puzzle):
    board = []

    #this goes through the puzzle string 9 characters at a time
    for i in range(0, 81, 9):
        #this converts each group of 9 characters into one row of integers
        row = [int(x) for x in puzzle[i:i+9]]
        board.append(row)

    #this returns the 9x9 board
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


#this is the implementation of the backtracking algorithm
def solve(board):
    global recursive_calls
    global assignments

    #this counts each time the solve function is called
    recursive_calls += 1

    #this is to find the next empty cell
    emptycell = empty_cell_finder(board)

    #this is to return solved in the case there are no empty cells remaining or return the coordinates if there exists an empty cell
    if emptycell is None:
        return True
    else:
        row, col = emptycell

    for number in range(1, 10):
        #this is to check if it is a valid placement of that number
        if isValidPlacement(board, row, col, number):

            #this counts each time a number is placed temporarily on the board
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


#this function runs the backtracking solver on all puzzles in the csv file
def run_experiment(csv_file):
    global recursive_calls
    global assignments

    #this reads the dataset from the csv file
    df = pd.read_csv(csv_file)

    #this keeps only the easy, medium, and hard puzzles
    df = df[df["difficulty"].isin(["easy", "medium", "hard"])]
    df = df.head(150)

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

        #this solves the sudoku board
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
        
        
