# Sudoku Solver using Backtracking with Least Constraining Value (LCV)

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

   
    #sort values so that the number with the LEAST impact (least constraining) comes first
    values.sort(key=lambda x: x[1])
  

    #return just the numbers in the new LCV order
    return [num for num, _ in values]


#this is the implementation of the backtracking algorithm with LCV
def solve(board):

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

        #we set that coordinate to equal the number temporarily
        board[row][col] = number

        #we use recursion to solve the remaining board
        if solve(board):
            return True

        #we set that cell equal to 0 if that number does not give a valid solution
        board[row][col] = 0

    #we return false in the case no number works and we continue backtracking
    return False


#main function to test the solver
def main():
    print("Attempting to solve:")


    #the hardest sudoku puzzle
    puzzle = [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ]
    
    for row in puzzle:
        print(row)

    if solve(puzzle):
        print("Solution found:")
        for row in puzzle:
            print(row)
    else:
        print("No Solution Exists")




   

if __name__ == "__main__":
    main()






