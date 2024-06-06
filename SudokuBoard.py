from sudoku import Sudoku


def create_sudoku(difficulty=0.5):
    # Generate a Sudoku board
    board = Sudoku(3).difficulty(difficulty)
    return board


def print_sudoku_board(board):
    # Print the Sudoku board
    for row in board.board:
        print(type(row))
        print(row)


def play_sudoku():
    # Create a Sudoku board
    board = create_sudoku(difficulty=0.5)

    print("Welcome to Sudoku!")
    print("Here's your Sudoku board:")
    print_sudoku_board(board)
    board.show_full()
    board.solve().show_full()

    # while not board.is_solved():
    #     row = int(input("Enter row (1-9): ")) - 1
    #     col = int(input("Enter column (1-9): ")) - 1
    #     value = int(input("Enter value (1-9): "))
    #
    #     if board.is_valid_move(row, col, value):
    #         board.make_move(row, col, value)
    #         print("Move accepted.")
    #         print_sudoku_board(board)
    #     else:
    #         print("Invalid move. Try again.")

    print("Congratulations! You solved the Sudoku puzzle.")


if __name__ == "__main__":
    play_sudoku()