# Sudoku Solver using Backtracking with Forward Checking

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

    #this is to find the next empty cell
    emptycell = empty_cell_finder(board)

    #this is to return solved in the case there are no empty cells remaining
    if emptycell is None:
        return True
    else:
        row, col = emptycell

    #try numbers from the domain of that cell
    for number in domains[(row, col)].copy():

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



#main function to run the solver
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
