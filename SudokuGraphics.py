import pygame
import sys

class SudokuGraphics:

    def __init__(self, game, help=False):
        self.game = game
        self.help = help

        # Constants
        self.WIDTH, self.HEIGHT = 750, 450  # Size of the window
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 450, 450
        self.ROWS, self.COLS = 9, 9  # Number of rows and columns
        self.CELL_SIZE = self.BOARD_WIDTH // self.COLS  # Size of each cell

        # Colors
        self.LINE_COLOR = (0, 0, 0)
        self.HIGHLIGHT_COLOR = (173, 216, 230)

        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Sudoku")

        self.selected_cell = None

        self.start()

    def start(self):
        while True:
            if self.game.get_current() != None:
                if self.game.get_current() == self.game.get_solved():
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
                if pos[0] >= 150 and pos[0] <= 300 and pos[1] >= 200 and pos[1] <= 250:
                    self.game.setBoard()

        self.screen.fill((255, 255, 255))  # Clear screen
        self.draw_start_screen()
        pygame.display.flip()

    def winScreen(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[0] >= 150 and pos[0] <= 300 and pos[1] >= 200 and pos[1] <= 250:
                    self.game.setBoard()

        self.screen.fill((255, 255, 255))
        self.draw_win_screen()
        pygame.display.flip()

    def playWithBoard(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // self.CELL_SIZE, y // self.CELL_SIZE
                if self.selected_cell == (row, col) or col > 8:
                    self.selected_cell = None
                else:
                    self.selected_cell = (row, col)

                if x >= 500 and x <= 600:
                    if y >= 140 and y <= 190:
                        self.game.reset_board()
                    if y >= 200 and y <= 250:
                        self.game.undo_move()
                    if y >= 260 and y <= 310:
                        self.game.setBoard()

            if event.type == pygame.KEYDOWN:
                if self.selected_cell:
                    row, col = self.selected_cell
                    if event.key in range(pygame.K_0, pygame.K_9 + 1):  # Keys 1-9
                        self.game.makeMove(row, col, event.key - pygame.K_0)

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

        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 200), (310, 200), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 250), (310, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (140, 200), (140, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (310, 200), (310, 250), 3)
        font = pygame.font.Font(None, 36)
        text = font.render("Click to Begin", True, self.LINE_COLOR)
        self.screen.blit(text, (146, 212))

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
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 140), (600, 140), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 190), (600, 190), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 140), (500, 190), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (600, 140), (600, 190), 3)

        # Draw undo button
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 200), (600, 200), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 250), (600, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 200), (500, 250), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (600, 200), (600, 250), 3)

        # Draw next board button
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 260), (600, 260), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 310), (600, 310), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (500, 260), (500, 310), 3)
        pygame.draw.line(self.screen, self.LINE_COLOR, (600, 260), (600, 310), 3)

    # Function to highlight a cell
    def highlight_cell(self):
        if self.selected_cell:
            row, col = self.selected_cell
            pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    # Function to draw numbers (replace with actual Sudoku board data)
    def draw_numbers(self):
        font = pygame.font.Font(None, 36)
        board = self.game.get_current()
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if board[row][col] != None:
                    text = font.render(str(board[row][col]), True, self.LINE_COLOR)
                    self.screen.blit(text, (col * self.CELL_SIZE + 15, row * self.CELL_SIZE + 10))
