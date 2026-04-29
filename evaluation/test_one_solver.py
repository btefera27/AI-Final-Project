from copy import deepcopy

from cv.parse_dat import parse_dat_file, print_board
from solvers.sudoku_backtracking import solve


def main():
    dat_path = "data/raw/sudoku_dataset/mixed_incomplete/image1002.dat"
    # Change this path if your incomplete .dat file is somewhere else.

    board = parse_dat_file(dat_path)

    print("Original puzzle:")
    print_board(board)

    board_to_solve = deepcopy(board)

    solved = solve(board_to_solve)

    print("\nSolved?", solved)

    if solved:
        print("\nSolved board:")
        print_board(board_to_solve)


if __name__ == "__main__":
    main()