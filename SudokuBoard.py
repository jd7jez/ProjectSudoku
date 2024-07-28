import sys

from sudoku import Sudoku
import random


class SudokuBoard:

    def __init__(self, width: int=3):
        self.width = width

    def generate_board_pair(self, difficulty=None):

        # If no given difficulty then randomly select one
        if difficulty == None:
            difficulty = random.uniform(0.15, 0.7)

        # Create a board with specified difficulty
        board = Sudoku(self.width, seed=random.randrange(0, sys.maxsize)).difficulty(difficulty)

        # Save copy of initial board
        unsolved = list(board.board)

        # Solve the board then return the unsolved and solved boards in list form
        solved = list(board.solve(raising=True).board)
        return unsolved, solved


if __name__ == "__main__":
    # boards = generate_board_pair(0.95)
    # print_board(boards[0])
    # print_board(boards[1])
    # board = Sudoku(3).difficulty(0.95)
    # solved = list(board.solve(raising=True).board)
    # count = 1
    #
    # for i in range(5):
    #     if solved != list(board.solve(raising=True).board):
    #         count += 1
    #
    # print(count)
    sb = SudokuBoard()
    diffs = [random.uniform(0.15, 0.7) for _ in range(5)]
    for i in range(5):
        print(sb.generate_board_pair())
