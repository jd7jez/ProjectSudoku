import pygame
import sys

class SudokuGraphics:
    # Color scheme:
    # 0: Given, gray
    # 1: Open/Guessed, white
    # 2: Correct, green
    # 3: Wrong, red
    # 4: Highlighted, light blue

    def __init__(self, game, help=False):
        self.game = game
        self.help = help

        # Constants
        self.WIDTH, self.HEIGHT = 750, 468  # Size of the window
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 468, 468
        self.ROWS, self.COLS = 9, 9  # Number of rows and columns
        self.CELL_SIZE = 50  # Size of each cell

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.HIGHLIGHT_COLOR = (173, 216, 230)

        pygame.init()

        # Initialize containers for the buttons and cell metadata
        self.buttons = dict()
        self.create_buttons()
        # Cell metadata is a (9,9) python list where each element is the metadata of the corresponding cell
        # The format is: [coordinates, top_left_corner, cell_rect, cell_color, num_color]
        # 0 - coordinates - what the position of this cell is in the format (row, col)
        # 1 - top_left_corner - defines where the top left corner of the cell is with format (x, y)
        # 2 - cell_rect - pygame rectangle object that represents the area of the cell
        # 3 - cell_color - the color the cell should be of format (r, g, b)
        # 3 - num_color - the color the number should be of format (r, g, b)
        self.cell_md = [[[(row, col), (x, y), pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE), self.WHITE, self.BLACK]
                        for col in range(9) for x, y in [self.get_cell_top_left(row, col)]] for row in range(9)]

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Sudoku")

        self.selected_cell = None

        self.start()

    def create_buttons(self):
        # Create start button
        start_width = 200
        start_height = 70
        start_x, start_y = (self.WIDTH - start_width) // 2, (self.HEIGHT - start_height) // 2
        start_rect = pygame.Rect(start_x, start_y, start_width, start_height)
        font = pygame.font.Font(None, 36)
        start_text = font.render("Click to Begin", True, self.BLACK)
        start_text_rect = start_text.get_rect(center=(start_x + start_width // 2, start_y + start_height // 2))
        self.buttons['start'] = [start_rect, start_text, start_text_rect]

        # Create reset button
        reset_button_rect = pygame.Rect(500, 140, 200, 50)
        reset_button_text = font.render("Reset", True, self.BLACK)
        reset_button_text_rect = reset_button_text.get_rect(center=((500 + 700) // 2, (140 + 190) // 2))
        self.buttons['reset'] = [reset_button_rect, reset_button_text, reset_button_text_rect]

        # Draw undo button
        undo_button_rect = pygame.Rect(500, 200, 200, 50)
        undo_button_text = font.render("Undo", True, self.BLACK)
        undo_button_text_rect = undo_button_text.get_rect(center=((500 + 700) // 2, (200 + 250) // 2))
        self.buttons['undo'] = [undo_button_rect, undo_button_text, undo_button_text_rect]

        # Draw new board button
        new_button_rect = pygame.Rect(500, 260, 200, 50)
        new_button_text = font.render("New Game", True, self.BLACK)
        new_button_text_rect = new_button_text.get_rect(center=((500 + 700) // 2, (260 + 310) // 2))
        self.buttons['new'] = [new_button_rect, new_button_text, new_button_text_rect]

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
        self.draw_start_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if self.buttons['start'][0].collidepoint(pos):
                    self.game.setBoard()

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
        self.screen.fill((0, 0, 0))  # Clear screen
        self.draw_buttons()
        self.draw_grid_new()
        self.draw_numbers()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if x <= 468:
                    self.click_cell(x, y)
                else:
                    if self.buttons['reset'][0].collidepoint((x, y)):
                        self.game.reset_board()
                    if self.buttons['undo'][0].collidepoint((x, y)):
                        self.game.undo_move()
                    if self.buttons['new'][0].collidepoint((x, y)):
                        self.game.setBoard()

            if event.type == pygame.KEYDOWN:
                if self.selected_cell:
                    row, col = self.selected_cell
                    if event.key in range(pygame.K_0, pygame.K_9 + 1):  # Keys 1-9
                        self.game.makeMove(row, col, event.key - pygame.K_0)

    def draw_start_screen(self):
        self.screen.fill((255, 255, 255))  # Clear screen

        title_font = pygame.font.Font(None, 100)
        title_text = title_font.render("Sudoku", True, self.BLACK)
        screen_width, screen_height = self.screen.get_size()
        title_rect = title_text.get_rect(center=(screen_width // 2, 100))
        self.screen.blit(title_text, title_rect)

        pygame.draw.rect(self.screen, self.BLACK, self.buttons['start'][0], 3)
        self.screen.blit(self.buttons['start'][1], self.buttons['start'][2])

        pygame.display.flip()

    def draw_win_screen(self):
        pygame.draw.line(self.screen, self.BLACK, (140, 200), (310, 200), 3)
        pygame.draw.line(self.screen, self.BLACK, (140, 250), (310, 250), 3)
        pygame.draw.line(self.screen, self.BLACK, (140, 200), (140, 250), 3)
        pygame.draw.line(self.screen, self.BLACK, (310, 200), (310, 250), 3)
        font = pygame.font.Font(None, 36)
        text1 = font.render("Click to Replay", True, self.BLACK)
        text2 = font.render("Congratulations", True, self.BLACK)
        self.screen.blit(text1, (144, 212))
        self.screen.blit(text2, (144, 150))

    # Function to draw the grid
    def draw_grid(self):
        for i in range(self.ROWS + 1):
            line_width = 1 if i % 3 != 0 else 3
            pygame.draw.line(self.screen, self.BLACK, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.BOARD_HEIGHT), line_width)
            pygame.draw.line(self.screen, self.BLACK, (0, i * self.CELL_SIZE), (self.BOARD_WIDTH, i * self.CELL_SIZE), line_width)

    def draw_grid_new(self):
        for row in self.cell_md:
            for md in row:
                rect = md[2]
                color = md[3]
                pygame.draw.rect(self.screen, color, rect)

    def get_cell_top_left(self, row, col):
        x = self.CELL_SIZE * col
        y = self.CELL_SIZE * row
        for x_i in range(col+1):
            x += 3 if x_i % 3 == 0 else 1
        for y_i in range(row+1):
            y += 3 if y_i % 3 == 0 else 1
        return x, y

    def draw_buttons(self):
        # Draw reset button
        pygame.draw.rect(self.screen, self.WHITE, self.buttons['reset'][0])
        self.screen.blit(self.buttons['reset'][1], self.buttons['reset'][2])

        # Draw undo button
        pygame.draw.rect(self.screen, self.WHITE, self.buttons['undo'][0])
        self.screen.blit(self.buttons['undo'][1], self.buttons['undo'][2])

        # Draw next board button
        pygame.draw.rect(self.screen, self.WHITE, self.buttons['new'][0])
        self.screen.blit(self.buttons['new'][1], self.buttons['new'][2])

    # Function to highlight a cell
    def highlight_cell(self):
        if self.selected_cell:
            row, col = self.selected_cell
            pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    def click_cell(self, x, y):
        for row in self.cell_md:
            for md in row:
                cell_rect = md[2]
                if cell_rect.collidepoint((x, y)):
                    row, col = md[0]
                    if self.selected_cell:
                        self.cell_md[self.selected_cell[0]][self.selected_cell[1]][3] = self.WHITE
                        if self.selected_cell != (row, col):
                            self.selected_cell = (row, col)
                            self.cell_md[row][col][3] = self.HIGHLIGHT_COLOR
                        else:
                            self.selected_cell = None
                    else:
                        self.selected_cell = (row, col)
                        self.cell_md[row][col][3] = self.HIGHLIGHT_COLOR
                    break



    # Function to draw numbers (replace with actual Sudoku board data)
    def draw_numbers(self):
        font = pygame.font.Font(None, 36)
        board = self.game.get_current()
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if board[row][col] != None:
                    md = self.cell_md[row][col]
                    x, y = md[1]
                    num_color = md[4]
                    text = font.render(str(board[row][col]), True, num_color)
                    self.screen.blit(text, (x + 17, y + 13))