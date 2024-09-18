from SudokuBoard import SudokuBoard
import pygame
import sys
import copy

class SudokuGame:
    # Color scheme:
    # 0: Given, gray
    # 1: Open/Guessed, white
    # 2: Correct, green
    # 3: Wrong, red

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
        self.current = unsolved.copy() if unsolved != None else None
        self.preload = preload
        self.num_boards = num_boards
        self.prev_states = []
        self.colors = None
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
            self.play()

    def validate_rewards(self, rewards):
        if isinstance(rewards, list) and (isinstance(val, float) for val in rewards) and len(rewards) == 4:
            self.rewards = rewards
        else:
            self.printv("Invalid rewards given, cannot configure rewards")
            self.reward = False

    def play(self):
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Sudoku")

        while True:
            if self.current != None:
                if self.current == self.solved:
                    self.winScreen()
                else:
                    self.playWithBoard()
            else:
                self.startScreen()

    def startScreen(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if self.box_rect.collidepoint(pos):
                    self.setBoard()

        self.screen.fill((255, 255, 255))  # Clear screen
        self.box_rect = self.draw_start_screen()

        pygame.display.flip()

    def winScreen(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[0] >= 150 and pos[0] <= 300 and pos[1] >= 200 and pos[1] <= 250:
                    self.setBoard()

        self.screen.fill((255, 255, 255))
        self.draw_win_screen()

        pygame.display.flip()


    def playWithBoard(self):
        reset_button_rect, undo_button_rect, new_game_button_rect = self.draw_buttons()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                if reset_button_rect.collidepoint(pos):
                    self.reset_board()
                    self.screen.fill((255, 255, 255))  # Clear the screen
                    self.draw_grid()
                    self.highlight_cell()
                    self.draw_numbers()
                    pygame.display.flip()

                    # Check if the mouse click was inside the undo button
                elif undo_button_rect.collidepoint(pos):
                    self.undo_move()

                    # Check if the mouse click was inside the new game button
                elif new_game_button_rect.collidepoint(pos):
                    self.setBoard()

                else:
                    x, y = pos
                    col, row = x // self.CELL_SIZE, y // self.CELL_SIZE

                    if col < self.COLS and row < self.ROWS:
                        if self.selected_cell == (row, col):
                            self.selected_cell = None  # Deselect if clicking the same cell
                        else:
                            self.selected_cell = (row, col)

            if event.type == pygame.KEYDOWN:
                if self.selected_cell:
                    row, col = self.selected_cell
                    if event.key in range(pygame.K_0, pygame.K_9 + 1):  # Keys 1-9
                        self.makeMove(row, col, event.key - pygame.K_0)

        self.screen.fill((255, 255, 255))  # Clear screen
        self.draw_buttons()
        self.draw_grid()
        self.highlight_cell()
        self.draw_numbers()

        pygame.display.flip()

    def draw_start_screen(self):
        title_font = pygame.font.Font(None, 100)
        title_text = title_font.render("Sudoku", True, self.LINE_COLOR)
        screen_width, screen_height = self.screen.get_size()
        title_rect = title_text.get_rect(center=(screen_width // 2, 100))
        self.screen.blit(title_text, title_rect)

        box_width = 200
        box_height = 70
        top_left_x = (screen_width - box_width) // 2
        top_left_y = (screen_height - box_height) // 2

        pygame.draw.line(self.screen, self.LINE_COLOR, (top_left_x, top_left_y), (top_left_x + box_width, top_left_y), 3)  # Top line
        pygame.draw.line(self.screen, self.LINE_COLOR, (top_left_x, top_left_y + box_height),(top_left_x + box_width, top_left_y + box_height), 3)  # Bottom line
        pygame.draw.line(self.screen, self.LINE_COLOR, (top_left_x, top_left_y), (top_left_x, top_left_y + box_height), 3)  # Left line
        pygame.draw.line(self.screen, self.LINE_COLOR, (top_left_x + box_width, top_left_y), (top_left_x + box_width, top_left_y + box_height), 3)  # Right line

        font = pygame.font.Font(None, 36)
        text = font.render("Click to Begin", True, self.LINE_COLOR)
        text_rect = text.get_rect(center=(top_left_x + box_width // 2, top_left_y + box_height // 2))
        self.screen.blit(text, text_rect)

        box_rect = pygame.Rect(top_left_x, top_left_y, box_width, box_height)

        return box_rect

    def draw_win_screen(self):
        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 200), (310, 200), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 250), (310, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 200), (140, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (310, 200), (310, 250), 3)
        font = pygame.font.Font(None, 36)
        text1 = font.render("Click to Replay", True, self.LINE_COLOR)
        text2 = font.render("Congratulations", True, self.LINE_COLOR)
        self.screen.blit(text1, (144, 212))
        self.screen.blit(text2, (144, 150))

    # Function to draw the grid
    def draw_grid(self):
        for i in range(self.ROWS + 1):
            line_width = 1 if i % 3 != 0 else 3
            pygame.draw.line(self.screen, self.LINE_COLOR, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.BOARD_HEIGHT), line_width)
            pygame.draw.line(self.screen, self.LINE_COLOR, (0, i * self.CELL_SIZE), (self.BOARD_WIDTH, i * self.CELL_SIZE), line_width)

    def draw_buttons(self):



        # Draw reset button
        reset_button_rect = pygame.Rect(500, 140, 100, 50)  # Define a rectangle for the reset button
        pygame.draw.rect(self.screen, self.LINE_COLOR, reset_button_rect, 3)

        reset_center_x = (500 + 600) // 2
        reset_center_y = (140 + 190) // 2

        font = pygame.font.Font(None, 36)
        reset_button_text = font.render("Reset", True, self.LINE_COLOR)
        reset_button_text_rect = reset_button_text.get_rect(center=(reset_center_x, reset_center_y))
        self.screen.blit(reset_button_text, reset_button_text_rect)

        # Draw undo button
        undo_button_rect = pygame.Rect(500, 200, 100, 50)
        pygame.draw.rect(self.screen, self.LINE_COLOR, undo_button_rect, 3)

        undo_center_x = (500 + 600) // 2
        undo_center_y = (200 + 250) // 2

        font = pygame.font.Font(None, 36)
        undo_button_text = font.render("Undo", True, self.LINE_COLOR)
        undo_button_text_rect = undo_button_text.get_rect(center=(undo_center_x, undo_center_y))
        self.screen.blit(undo_button_text, undo_button_text_rect)

        # Draw new board button
        new_game_button_rect = pygame.Rect(500, 260, 100, 50)
        pygame.draw.rect(self.screen, self.LINE_COLOR, new_game_button_rect, 3)

        new_center_x = (500 + 600) // 2
        new_center_y = (260 + 310) // 2

        font = pygame.font.Font(None, 26)
        new_button_text = font.render("New Game", True, self.LINE_COLOR)
        new_button_text_rect = new_button_text.get_rect(center=(new_center_x, new_center_y))
        self.screen.blit(new_button_text, new_button_text_rect)

        return reset_button_rect, undo_button_rect, new_game_button_rect

    # Function to highlight a cell
    def highlight_cell(self):
        if self.selected_cell:
            row, col = self.selected_cell
            pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    # Function to draw numbers (replace with actual Sudoku board data)
    def draw_numbers(self):
        font = pygame.font.Font(None, 36)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.current[row][col] != None:
                    text = font.render(str(self.current[row][col]), True, self.LINE_COLOR)
                    self.screen.blit(text, (col * self.CELL_SIZE + 15, row * self.CELL_SIZE + 10))

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

    def makeMove(self, row, col, val):
        self.printv("Making move")
        if self.unsolved == None or self.solved == None or self.current == None:
            self.printv("This Sudoku Game does not currently have the boards properly configured."
                  "Call setBoard to reset the boards.")
            return 0, 0

        if self.unsolved[row][col] != None:
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
            self.current[row][col] = None
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
        self.current = self.unsolved

    def get_unsolved(self):
        return copy.deepcopy(self.unsolved)

    def get_current(self):
        return copy.deepcopy(self.current)

    def get_solved(self):
        return copy.deepcopy(self.solved)

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