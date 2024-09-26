from SudokuBoard import SudokuBoard
from SudokuGraphics import SudokuGraphics
import pygame
import sys
import copy

class SudokuGame:
    # Reward Scheme, by index, this is also the same for the move codes:
    # 0: Correct Guess
    # 1: Correct Guess that finishes board
    # 2: Incorrect Guess
    # 3: Invalid Guess that defies rules

    def __init__(self, unsolved=None, solved=None, preload=False, filename='sudoku-3m-kaggle.csv', num_boards=10, visualize=False, help=False, reward=False, rewards=None, verbose=1):
        self.sb = SudokuBoard()
        self.verbose = verbose
        self.unsolved = unsolved
        self.solved = solved
        self.current = copy.deepcopy(unsolved) if unsolved != None else None
        self.preload = preload
        self.num_boards = num_boards
        self.prev_states = []
        self.colors = None
        self.graphics = None
        self.visualize = visualize
        self.help = visualize and help
        self.reward = reward
        self.rewards = None
        self.loaded_unsolved = []
        self.loaded_solved = []

        if self.reward:
            self.validate_rewards(rewards)

        if self.preload:
            self.loaded_unsolved, self.loaded_solved, self.loaded_difficulties = self.sb.generate_board_pairs(num_boards=self.num_boards, filename=filename)

        # Constants
        self.WIDTH, self.HEIGHT = 750, 450  # Size of the window
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 450, 450
        self.ROWS, self.COLS = 9, 9  # Number of rows and columns
        self.CELL_SIZE = self.BOARD_WIDTH // self.COLS  # Size of each cell

        # Colors
        self.LINE_COLOR = (0, 0, 0)
        self.HIGHLIGHT_COLOR = (173, 216, 230)

        self.screen = None
        self.selected_cell = None
        self.printv("Created Sudoku Game")

        if self.visualize:
            self.printv("Visualizing and playing SudokuGame")
            self.graphics = SudokuGraphics(self, help)

    def validate_rewards(self, rewards):
        if isinstance(rewards, list) and (isinstance(val, float) for val in rewards) and len(rewards) == 4:
            self.rewards = rewards
        else:
            self.printv("Invalid rewards given, cannot configure rewards")
            self.reward = False

    def setBoard(self, unsolved_board=None, solved_board=None):
        if self.preload and len(self.loaded_unsolved) > 0:
            unsolved_board = self.loaded_unsolved.pop(0)
            solved_board = self.loaded_solved.pop(0)

        if unsolved_board == None and solved_board == None:
            unsolved_list, solved_list = self.sb.generate_board_pairs(num_boards=1)
            unsolved_board = copy.deepcopy(unsolved_list[0])
            solved_board = copy.deepcopy(solved_list[0])

        if unsolved_board == None or solved_board == None:
            self.printv("Provided an unsolved or solved board without the counterpart."
                  "You must set the game board with both the unsolved and solved boards."
                  "Setting board failed.")
            return

        self.unsolved = unsolved_board
        self.solved = solved_board
        self.current = copy.deepcopy(unsolved_board)
        self.prev_states = []

    def makeMove(self, row, col, val):
        self.printv(f"Making move: {val} -> ({row}, {col})")
        if self.unsolved == None or self.solved == None or self.current == None:
            self.printv("This Sudoku Game does not currently have the boards properly configured."
                  "Call setBoard to reset the boards.")
            return 0, 0

        if self.unsolved[row][col] != 0:
            self.printv("Cannot perform a move on a square that was given.")
            if self.reward:
                return self.rewards[3], 3
            else:
                return 0, 0

        if val < 0 or val > 9:
            self.printv("The value guessed must be between 0 and 9 inclusive")
            return self.rewards[3], 3

        self.prev_states.append((row, col, self.current[row][col]))

        if val == 0:
            self.current[row][col] = 0
            if self.reward:
                return self.rewards[2], 2
        elif self.current[row][col] == val:
            if self.reward:
                return self.rewards[2], 2
        else:
            self.current[row][col] = val
            if self.reward:
                if self.solved[row][col] == val:
                    if self.current == self.solved:
                        return self.rewards[1], 1
                    else:
                        return self.rewards[0], 0
                else:
                    return self.rewards[2], 2

    def undo_move(self):
        self.printv("undoing move")
        if len(self.prev_states) == 0:
            self.printv("No last move to undo")
            return

        row, col, val = self.prev_states.pop(-1)
        self.current[row][col] = val

    def reset_board(self):
        self.current = copy.deepcopy(self.unsolved)
        self.prev_states = []

    def get_unsolved(self):
        return copy.deepcopy(self.unsolved) if self.unsolved != None else None

    def get_current(self):
        return copy.deepcopy(self.current) if self.current != None else None

    def get_solved(self):
        return copy.deepcopy(self.solved) if self.solved != None else None

    def get_rewards(self):
        return copy.deepcopy(self.rewards)

    def has_loaded_board(self):
        return len(self.loaded_unsolved) > 0

    def printCurrentBoard(self):
        self.sb.print_board(self.current)

    def printv(self, text):
        if self.verbose:
            print(text)

if __name__ == "__main__":
    sg = SudokuGame(preload=True, visualize=True)