# Sudoku Solver using Backtracking Algorithm

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
    row_box = (row //3)
    col_box = (col//3)

    #this is to check if the number is present in the subgrid
    for r in range(row_box*3, (row_box*3)+3):
        for c in range(col_box*3, (col_box*3) + 3):
            if board[r][c] == number:
                return False

    #if all the conditions are met, then it returns True which means it is a valid placement
    return True

#this is the implementation of the backtracking algorithm (logic explained in the beginning of the code)
def solve(board):
    #this is to find the next empty cell
    emptycell = empty_cell_finder(board)
        
    #this is to return solved in the case there are no empty cells remaining or return the coordinates if there exists an empty cell
    if emptycell is None:
        return True
    else:
        row, col = emptycell
        
    for number in range(1,10):
        #this is to check if it is a valid placement of that number
        if isValidPlacement(board, row, col, number):
            #we set that coordinate to equal the number temporarily
            board[row][col] = number

            #we use recursion to solve the remaining board
            if solve(board):
                return True
                
            #we set that cell equal to 0 if that number does not give a valid solution
            board[row][col] = 0

    #we return false in the case no number works and we continue backtracking
    return False
    
#sample sudoku board
board = [
    [1, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]
]

#this is to solve the sample board above and print it in the same format
if solve(board):
    for row in board:
        print(row)
else:
    print("No Solution Exists")
        
        
