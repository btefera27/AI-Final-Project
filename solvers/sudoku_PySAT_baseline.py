from pysat.solvers import Cadical195, Glucose42, MapleChrono
import numpy as np

class SudokuSolver:
    def __init__(self, puzzle, solver):
        self.puzzle = puzzle
        self.solver = solver
        self.variables = None

    def create_variables(self):
        """
        Set self.variables as a 3D list containing the PySAT variables. 
        self.variables[i][j][k] is true if cell i,j contains the value k+1.
        """

        B = np.empty((9, 9, 9), dtype = object)

        variable = 1
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    B[i][j][k] = variable
                    variable += 1

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

        # Each cell must contain a value between 1 and 9.
        for i in range(9):
            for j in range(9):
                # Retrieve all variables associated to the value of interest
                cell_variables = [self.variables[i][j][k] for k in range(9)]

                # Check that each cell contains at least one value.
                self.solver.add_clause(cell_variables)

                # Check that each cell contains at most one value.
                for x in range(9):
                    for y in range(x + 1, 9):
                        self.solver.add_clause([-cell_variables[x], -cell_variables[y]])

        # Each row must contain each value exactly once.
        for i in range(9):
            for k in range(9):
                # Retrieve all variables associated to the value of interest
                row_variables = [self.variables[i][j][k] for j in range(9)]

                # Check that the value appears at least once in the row.
                self.solver.add_clause(row_variables)

                # Check that the value appears at most once in the row.
                for x in range(9):
                    for y in range(x + 1, 9):
                        self.solver.add_clause([-row_variables[x], -row_variables[y]])

        # Each column must contain each value exactly once.
        for j in range(9):
            for k in range(9):
                # Retrieve all variables associated to the value of interest
                column_variables = [self.variables[i][j][k] for i in range(9)]

                # Check that the value appears at least once in the column.
                self.solver.add_clause(column_variables)

                # Check that the value appears at most once in the column.
                for x in range(9):
                    for y in range(x + 1, 9):
                        self.solver.add_clause([-column_variables[x], -column_variables[y]])

        # Each 3x3 subgrid must contain each value exactly once.
        for i_ in range(3):
            for j_ in range(3):
                for k in range(9):
                    # Retrieve all variables associated to the value of interest
                    subgrid_variables = []
                    for i in range(3 * i_, 3 * (i_ + 1)):
                        for j in range(3 * j_, 3 * (j_ + 1)):
                            subgrid_variables.append(self.variables[i][j][k])

                    # Check that the value appears at least once in the subgrid.
                    self.solver.add_clause(subgrid_variables)

                    # Check that the value appears at most once in the subgrid.
                    for x in range(9):
                        for y in range(x + 1, 9):
                            self.solver.add_clause([-subgrid_variables[x], -subgrid_variables[y]])

    def encode_puzzle(self):
        """
        Encode the initial puzzle into the solver.
        """
        for i in range(9):
            for j in range(9):
                value = self.puzzle[i][j]
                if not 0 <= value <= 9 or not isinstance(self.puzzle[i][j], int):
                    raise ValueError(f"Invalid value {value} at cell ({i}, {j})")
                elif value > 0:
                    self.solver.add_clause([self.variables[i][j][value - 1]])

    def extract_solution(self, model):
        """
        Extract the satisfying assignment and return it.
        """
        output = [[0 for _ in range(9)] for _ in range(9)]
        solution = model.get_model()

        for i in range(9):
            for j in range(9):
                for k in range(9):
                    value = solution[self.variables[i][j][k] - 1]
                    if value > 0:
                        output[i][j] = k + 1
                        continue
        
        return output

    def solve(self):
        """
        Solve the Sudoku puzzle.
        
        :return: A 9x9 list of lists representing the solved Sudoku puzzle, or None if no solution exists.
        """
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()
        
        if self.solver.solve() == True:
            # Extract the satisfying assignment from the solver
            solution = [[0 for _ in range(9)] for _ in range(9)]
            model = self.solver.get_model()

            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        value = model[self.variables[i][j][k] - 1]
                        if value > 0:
                            solution[i][j] = k + 1
                            continue
            
            return solution
        else:
            return None

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

    # Choose ONE solver to use
    # solver = SudokuSolver(puzzle, Cadical195()) # CaDiCaL 1.9.5 SAT solver
    # solver = SudokuSolver(puzzle, Glucose42()) # Glucose 4.2.1 SAT solver
    solver = SudokuSolver(puzzle, MapleChrono()) # MapleLCMDistChronoBT SAT solver

    solution = solver.solve()

    if solution:
        print("Solution found:")
        for row in solution:
            print(row)
    else:
        print("No solution exists.")

if __name__ == "__main__":
    main()