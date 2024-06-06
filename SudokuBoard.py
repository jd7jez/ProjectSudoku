from sudoku import Sudoku
import random

def create_sudoku(difficulty=0.5):
    # Generate a Sudoku board with specified difficulty
    board = Sudoku(3).difficulty(difficulty)
    return board


def print_sudoku_board(board):
    # Print the Sudoku board
    for row in board.board:
        print(type(row))
        print(row)

def generate_board_pair():
    # Get random difficulty
    difficulty = random.uniform(0, 1)

    # Create a board with the random difficulty
    board = Sudoku(3).difficulty(difficulty)

    # Save copy of initial board
    unsolved = list(board.board)

    # Attempt to solve the board
    try:
        # If successfully solved then return the unsolved and solved boards
        solved = list(board.solve().board)
        return (unsolved, solved)
    except:
        # Failed to solve the board so recursively try again
        return generate_board_pair()


if __name__ == "__main__":
    boards = generate_board_pair()
    print(boards[0])
    print(boards[1])