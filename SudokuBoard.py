import sys

from sudoku import Sudoku
import random
import time
import csv
import numpy as np


class SudokuBoard:

    def __init__(self, width: int=3):
        self.width = width


    def generate_board_pairs(self, num_boards=10000, difficulty=None, filename=None):
        # If a filename is given then just read out of that file
        if filename != None:
            print(f"Loading in {num_boards} sudoku boards.")
            start = time.time()
            result = self.read_board_file(filename, num_boards)
            end = time.time()
            print(f"Loaded in {num_boards} boards in {end-start} seconds.")
            return result

        # No filename is given, so we need to generate the boards from scratch, this may take some time
        print(f"No filename specified so boards will be generated from scratch, this will take a long time "
              f"and is not recommended when loading in more than 10 boards.")
        unsolved_boards = []
        solved_boards = []
        for _ in range(num_boards):
            # Generate a complete board then remove numbers to make a solvable puzzle with one unique solution
            unsolved, solved = self.make_puzzle(difficulty=difficulty, board=solved)
            unsolved_boards.append(unsolved)
            solved_boards.append(solved)

        # Return the unsolved and solved board pairs
        return unsolved_boards, solved_boards


    def num_is_valid(self, board, row, col, num):
        # If the value in this space is not None then return False
        if board[row][col] != None:
            return False

        # Check if the number exists in the row or column already, return false if it does
        for i in range(9):
            if board[i][col] == num or board[row][i] == num:
                return False

        # Check if the number exists in the box this the number is trying to be placed in, return false if it does
        box_row = 3 * (row // 3)
        box_col = 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        # The number is not invalid thus it is valid so return true
        return True

    def board_is_solvable(self, board):
        # Basically try every possible way of solving the board to see if there is a solution
        for row in range(9):
            for col in range(9):
                if board[row][col] == None:
                    # If this cell has no value in it then try every possible value in random order
                    nums = [val for val in range(1, 10)]
                    random.shuffle(nums)
                    for num in nums:
                        if self.num_is_valid(board, row, col, num):
                            # If the number for the cell is valid then try to solve the board with this number
                            board[row][col] = num
                            if self.board_is_solvable(board):
                                # If the board was solved then this was the right number and return true
                                return True
                            # Reset the cell to empty because this number did not work
                            board[row][col] = None
                    # Return false because there was no solvable board with the current values
                    return False
        # The board is solved already so return true, this is base case
        return True

    def count_solutions(self, board):
        # Initialize the solutions counter and iterate through every cell on the board
        solutions = 0
        for row in range(9):
            for col in range(9):
                if board[row][col] == None:
                    # If the cell has no value in it then try every possible value
                    for num in range(1, 10):
                        if self.num_is_valid(board, row, col, num):
                            # If the number is valid in this position then continue trying to find solutions with this number, add the solutions found to solutions
                            board[row][col] = num
                            solutions += self.count_solutions(board)
                            board[row][col] = None
                    # Return the number of solutions because if every value has been tried for this empty cell in every scenario then inherintly all possible solutions have been found
                    return solutions
        # Return 1 because the board is already filled out so only 1 possible solution right now
        return 1

    def generate_board(self):
        # Create an empty board and then attempt to solve it which should inherently create a board
        board = [[None for _ in range(9)] for _ in range(9)]
        if self.board_is_solvable(board):
            return board
        else:
            # This should never happen, but we will call generate_board again just to be safe
            return self.generate_board()

    def make_puzzle(self, difficulty=None, board=None):
        # If no given board then generate one
        if board == None:
            board = self.generate_board()

        # Copy the board
        solved = board.copy()

        # If no given difficulty then get a random one
        if difficulty == None:
            difficulty = random.randint(50, 60)

        # Get a list of all possible cells and then randomize the order
        cells = [(row, col) for col in range(9) for row in range(9)]
        random.shuffle(cells)

        # Remove the number of cells listed in difficulty ensuring there is always only 1 unique solution
        for cell in cells:
            if difficulty == 0:
                break
            row, col = cell
            val = board[row][col]
            board[row][col] = None
            if self.count_solutions(board) == 1:
                difficulty -= 1
            else:
                board[row][col] = val

        # If difficulty has reached 0 then all removals have happened and return, otherwise error
        if difficulty == 0:
            return solved, board
        else:
            print("Board difficulty not possible for board, using new board")
            return self.make_puzzle(difficulty=difficulty)

    def print_board(self, board):
        # Print the board in a nice format by looping over rows
        for row in board:
            row_string = "|"
            for val in row:
                if val in range(1,10):
                    row_string += f" {val} |"
                else:
                    row_string += "   |"
            print(row_string)

    def read_board_file(self, filename, numlines=10000):
        unsolved_boards = []
        solved_boards = []
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if i > numlines:
                    break
                unsolved_string = row[1]
                solved_string = row[2]
                unsolved_boards.append(self.string_to_board(unsolved_string))
                solved_boards.append(self.string_to_board(solved_string))
        return unsolved_boards, solved_boards

    def string_to_board(self, board_string):
        board = [[] for _ in range(9)]
        for i, char in enumerate(board_string):
            row = (i // 9)
            if char == '.':
                board[row].append(None)
            else:
                board[row].append(int(char))
        return board


if __name__ == "__main__":
    sb = SudokuBoard()

    # sb.read_board_file('sudoku-3m.csv')
    # unsolved, solved = sb.generate_board_pairs(num_boards=3, filename='sudoku-3m.csv',)
    # unsolved, solved = np.array(unsolved), np.array(solved)
    # print(solved[0][0])
    # print(type(solved[0][0]))

    start = time.time()
    unsolved, solved = sb.generate_board_pairs(num_boards=100000, filename='sudoku-3m.csv')
    end = time.time()
    print(f"Total Time to read 1000 boards:{end-start}")