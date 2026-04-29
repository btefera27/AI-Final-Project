from z3 import Solver, Bool, And, Or, Not, Implies, sat, unsat
import numpy as np
class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.solver = None
        self.variables = None

    def create_variables(self):
        """
        Set self.variables as a 3D list containing the Z3 variables. 
        self.variables[i][j][k] is true if cell i,j contains the value k+1.
        """

        B = np.empty((9, 9, 9), dtype=object)
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    B[i][j][k] = Bool(f"b{i}{j}{k}")

        self.variables = B

    def encode_rules(self):
        """
        Encode the rules of Sudoku into the solver.
        The rules are:
        1. Each cell must contain a value between 1 and 9.
        2. Each row must contain each value exactly once.
        3. Each column must contain each value exactly once.
        4. Each 3x3 subgrid must contain each value exactly once.
        """

        #Each cell must contain a value between 1 and 9.
        for i in range(9):
            for j in range(9):
                #there is atleast one
                self.solver.add(Or(self.variables[i][j][0], self.variables[i][j][1], self.variables[i][j][2], self.variables[i][j][3], self.variables[i][j][4], self.variables[i][j][5], 
                                   self.variables[i][j][6], self.variables[i][j][7], self.variables[i][j][8]))
                
                #atmost one
                for k in range(9):
                    for l in range(k+1, 9):
                          self.solver.add(Not(And(self.variables[i][j][k], self.variables[i][j][l])))



        #Each row must contain each value exactly once.
        for i in range(9):
            for k in range(9):

                # At least once in the row
                self.solver.add(Or(self.variables[i][0][k], self.variables[i][1][k], self.variables[i][2][k],
                                self.variables[i][3][k], self.variables[i][4][k], self.variables[i][5][k],
                                self.variables[i][6][k], self.variables[i][7][k], self.variables[i][8][k]))

                # At most once in the row
                for j in range(9):
                    for l in range(j + 1, 9):
                        self.solver.add(Not(And(self.variables[i][j][k], self.variables[i][l][k])))



         #Each column must contain each value exactly once.
        for j in range(9):
            for k in range(9):

                # At least once in the row
                self.solver.add(Or(self.variables[0][j][k], self.variables[1][j][k], self.variables[2][j][k],
                                self.variables[3][j][k], self.variables[4][j][k], self.variables[5][j][k],
                                self.variables[6][j][k], self.variables[7][j][k], self.variables[8][j][k]))

                # At most once in the row
                for i in range(9):
                    for l in range(i + 1, 9):
                        self.solver.add(Not(And(self.variables[i][j][k], self.variables[l][j][k])))


        #Each 3x3 subgrid must contain each value exactly once.
        for box_i in range(3):
            for box_j in range(3):
                for k in range(9):

                    cells = []

                    for i in range(3):
                        for j in range(3):
                            cells.append(self.variables[3*box_i+i][3*box_j+j][k])

                    self.solver.add(Or(cells))

                    for a in range(9):
                        for b in range(a+1, 9):
                            self.solver.add(Not(And(cells[a], cells[b])))
                                
        

    def encode_puzzle(self):
        """
        Encode the initial puzzle into the solver.
        """
        for i in range(9):
            for j in range(9):
                val = self.puzzle[i][j]
                if val != 0:
                    #self.variables[i][j][k] is true if cell i,j contains the value k+1
                    self.solver.add(self.variables[i][j][val-1])

    def extract_solution(self, model):
        """
        Extract the satisfying assignment from the given model and return it as a 
        9x9 list of lists.
        Args:
            model: The Z3 model containing the satisfying assignment.
        Returns:
            A 9x9 list of lists of integers representing the Sudoku solution.
        Hint:
            To access the value of a variable in the model, you can use:
            value = model.evaluate(var)
            where `var` is the Z3 variable whose value you want to retrieve.
        """
        # create a 3x3 solution matrix and set to 0
        solution = [[0]*9 for _ in range(9)]

        #put the solution to the solution array created above
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if model.evaluate(self.variables[i][j][k]):
                        solution[i][j] = k+1
        
        
        return solution

        
    
    def solve(self):
        """
        Solve the Sudoku puzzle.
        
        :return: A 9x9 list of lists representing the solved Sudoku puzzle, or None if no solution exists.
        """
        self.solver = Solver()
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()
        
        if self.solver.check() == sat:
            model = self.solver.model()
            solution = self.extract_solution(model)
            return solution
        else:
            return None


#main function to test the solver
def main():
    print("Attempting to solve:")
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

    solver = SudokuSolver(puzzle)
    solution = solver.solve()

    if solution:
        print("Solution found:")
        for row in solution:
            print(row)
    else:
        print("No solution exists.")





   

if __name__ == "__main__":
    main()
