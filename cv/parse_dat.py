from pathlib import Path


def parse_dat_file(dat_path):
    """
    Parse one Sudoku .dat metadata file.

    Expected .dat format:
    line 1: phone/camera model
    line 2: image metadata
    next 9 lines: Sudoku board, where 0 means empty cell

    Returns:
        board: 9x9 list of integers
    """
    dat_path = Path(dat_path)

    with open(dat_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 11:
        raise ValueError(f"{dat_path} does not contain enough lines.")

    board_lines = lines[2:11]
    board = []

    for line in board_lines:
        row = [int(x) for x in line.split()]

        if len(row) != 9:
            raise ValueError(f"Invalid row length in {dat_path}: {line}")

        for value in row:
            if value < 0 or value > 9:
                raise ValueError(f"Invalid Sudoku value {value} in {dat_path}")

        board.append(row)

    if len(board) != 9:
        raise ValueError(f"{dat_path} does not contain a 9x9 board.")

    return board


def print_board(board):
    """
    Nicely print a 9x9 Sudoku board.
    """
    for row in board:
        print(row)


if __name__ == "__main__":
    dat_path = "data/raw/sudoku_dataset/mixed_incomplete/image1002.dat"
    board = parse_dat_file(dat_path)
    print("Parsed Sudoku board:")
    print_board(board)