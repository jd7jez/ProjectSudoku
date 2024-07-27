from sudoku import Sudoku
import random
import numpy as np

def create_sudoku(difficulty=0.5):
    # Generate a Sudoku board with specified difficulty
    board = Sudoku(3).difficulty(difficulty)
    return board


def print_sudoku_board(board):
    # Print the Sudoku board
    for row in board.board:
        print(type(row))
        print(row)

def generate_board_pair(difficulty):

    # Create a board with the provided or random difficulty
    board = Sudoku(3).difficulty(difficulty)

    # Save copy of initial board
    unsolved = list(board.board)

    # Attempt to solve the board
    try:
        # If successfully solved then return the unsolved and solved boards in list form
        solved = list(board.solve(raising=True).board)
        return (unsolved, solved)
    except:
        # Failed to solve the board so recursively try again
        print("Failed to generate board with desired difficulty")
        return generate_board_pair()

def make_guess(guess, loc, boards):
    # Ensure that the guess is within the valid bounds of the board
    if loc[0] >= len(boards[0]) or loc[1] >= len(boards[0][0]):
        print("Not a valid guess")

    return boards[1][loc[0]][loc[1]] == guess

def print_board(board):
    for row in board:
        print(row)

if __name__ == "__main__":
    boards = generate_board_pair(0.95)
    print_board(boards[0])
    print_board(boards[1])
    board = Sudoku(3).difficulty(0.95)
    solved = list(board.solve(raising=True).board)
    count = 1

    for i in range(5):
        if solved != list(board.solve(raising=True).board):
            count += 1

    print(count)
