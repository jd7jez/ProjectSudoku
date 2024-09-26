import random
import time
import csv
import numpy as np
import copy


class SudokuBoard:

    def __init__(self):
        # Maybe consider making this a utility file with no class object defined in it
        pass

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
        missing = []
        for _ in range(num_boards):
            # Generate a complete board then remove numbers to make a solvable puzzle with one unique solution
            unsolved, solved, curr_missing = self.make_puzzle(difficulty=difficulty)
            unsolved_boards.append(unsolved)
            solved_boards.append(solved)
            missing.append(curr_missing)

        # Return the unsolved and solved board pairs
        return unsolved_boards, solved_boards, missing

    def get_valid_actions(self, board):
        valid_actions = [1 for _ in range(81 * 9)]
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val != 0:
                    first_action = (y * 81) + (x * 9)
                    for i in range(first_action, first_action+9):
                        valid_actions[i] = 0
        return valid_actions

    def get_actual_rewards_mask(self, board, solved):
        reward_mask = [0 for _ in range(81 * 9)]
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val == 0:
                    first_action = (y * 81) + (x * 9)
                    correct_action = first_action + (solved[y][x] - 1)
                    for i in range(first_action, first_action+9):
                        reward_mask[i] = 2 if i != correct_action else 1
        return reward_mask

    def generate_numpy_boards(self, num_boards=100, filename=None):
        unsolved, solved, missing = self.generate_board_pairs(num_boards=num_boards, filename=filename)
        unsolved, solved = np.array(unsolved).reshape((num_boards, 81)), np.array(solved).reshape((num_boards, 81))
        return unsolved, solved, missing

    # I am one-hot encoding the sudoku board data because the values are basically just categorical
    # I don't want the model to interpret any sort of quantitative relationship between numbers, strictly ordinal
    def one_hot_encode(self, boards):
        # Create numpy array of proper shape
        enc_boards = np.zeros((boards.shape[0], 81, 10))
        # One hot encode the values
        for i, board in enumerate(boards):
            for j, val in enumerate(board):
                if val != 0:
                    enc_boards[i][j][val] = 1
                else:
                    enc_boards[i][j][0] = 1
        # Return one hot encoded boards
        return enc_boards

    def num_is_valid(self, board, row, col, num):
        # If the value in this space is not zero then return False
        if board[row][col] != 0:
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
                if board[row][col] == 0:
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
                            board[row][col] = 0
                    # Return false because there was no solvable board with the current values
                    return False
        # The board is solved already so return true, this is base case
        return True

    def count_solutions(self, board):
        # Initialize the solutions counter and iterate through every cell on the board
        solutions = 0
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    # If the cell has no value in it then try every possible value
                    for num in range(1, 10):
                        if self.num_is_valid(board, row, col, num):
                            # If the number is valid in this position then continue trying to find solutions with this number, add the solutions found to solutions
                            board[row][col] = num
                            solutions += self.count_solutions(board)
                            board[row][col] = 0
                    # Return the number of solutions because if every value has been tried for this empty cell in every scenario then inherintly all possible solutions have been found
                    return solutions
        # Return 1 because the board is already filled out so only 1 possible solution right now
        return 1

    def generate_board(self):
        # Create an empty board and then attempt to solve it which should inherently create a board
        board = [[0 for _ in range(9)] for _ in range(9)]
        if self.board_is_solvable(board):
            return board
        else:
            # This should never happen, but we will call generate_board again just to be safe
            return self.generate_board()

    def make_puzzle(self, difficulty=None, board=None):
        # If no given board then generate one
        if board == None:
            board = self.generate_board()

        # Copy the board to a solved and unsolved version to mitigate reference issues
        unsolved = copy.deepcopy(board)
        solved = copy.deepcopy(board)

        # If no given difficulty then get a random one
        if difficulty == None:
            difficulty = random.randint(50, 60)
        missing = difficulty

        # Get a list of all possible cells and then randomize the order
        cells = [(row, col) for col in range(9) for row in range(9)]
        random.shuffle(cells)

        # Remove the number of cells listed in difficulty ensuring there is always only 1 unique solution
        for cell in cells:
            if difficulty == 0:
                break
            row, col = cell
            val = unsolved[row][col]
            unsolved[row][col] = 0
            if self.count_solutions(unsolved) == 1:
                difficulty -= 1
            else:
                unsolved[row][col] = val

        # If difficulty has reached 0 then all removals have happened and return, otherwise error
        if difficulty == 0:
            return unsolved, solved, missing
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
        missing = []
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i > numlines:
                    break
                unsolved_string = row[0]
                solved_string = row[1]
                missing_string = row[2]
                unsolved_boards.append(self.string_to_board(unsolved_string))
                solved_boards.append(self.string_to_board(solved_string))
                missing.append(int(missing_string))
        return unsolved_boards, solved_boards, missing

    def write_board_file(self, unsolved_boards, solved_boards, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for unsolved, solved in zip(unsolved_boards, solved_boards):
                missing = 0
                for row in unsolved:
                    for val in row:
                        if val == 0:
                            missing += 1
                unsolved_str = self.board_to_string(unsolved)
                solved_str = self.board_to_string(solved)

                board_data = [unsolved_str, solved_str, str(missing)]
                writer.writerow(board_data)

    def string_to_board(self, board_string):
        board = [[] for _ in range(9)]
        for i, char in enumerate(board_string):
            row = (i // 9)
            if char == '.':
                board[row].append(0)
            else:
                board[row].append(int(char))
        return board

    def board_to_string(self, board):
        board_string = ""
        for row in board:
            for val in row:
                cell = str(val) if val != 0 else '.'
                board_string += cell
        return board_string


if __name__ == "__main__":
    start = time.time()
    sb = SudokuBoard()
    unsolved, solved, missing = sb.generate_board_pairs(num_boards=10000, difficulty=50)
    sb.write_board_file(unsolved, solved, 'sudoku-10k-50missing.csv')
    end = time.time()
    print(f"Total Time to generate and save 10000 boards:{end - start}")

    # sb.read_board_file('sudoku-3m.csv')
    # unsolved, solved = sb.generate_board_pairs(num_boards=3, filename='sudoku-3m.csv',)
    # unsolved, solved = np.array(unsolved), np.array(solved)
    # print(solved[0][0])
    # print(type(solved[0][0]))

    # start = time.time()
    # unsolved, solved = sb.generate_board_pairs(num_boards=100000, filename='sudoku-3m.csv')
    # end = time.time()
    # print(f"Total Time to read 1000 boards:{end-start}")