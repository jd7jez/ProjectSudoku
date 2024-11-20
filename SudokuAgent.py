import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from SudokuBoard import SudokuBoard

class SudokuAgent:

    def __init__(self, agentfile, verbose=True):
        self.model = self.get_cnn_model()
        self.model.load_weights(agentfile)
        self.sb = SudokuBoard()
        self.verbose=verbose

    def get_cnn_model(self):
        model = Sequential([
            Conv2D(256, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
            BatchNormalization(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.25),
            BatchNormalization(),
            Dense(2048, activation='relu'),
            Dropout(0.25),
            BatchNormalization(),
            Dense(1024, activation='relu'),
            Dense(81 * 9, activation='linear')
        ])
        # model = Sequential([
        #     Conv2D(256, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
        #     Conv2D(512, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
        #     Flatten(),
        #     Dense(1024, activation='relu'),
        #     Dense(2048, activation='relu'),
        #     Dense(1024, activation='relu'),
        #     Dense(81*9, activation='linear')
        # ])
        model.compile(optimizer=Adam(learning_rate=0.001))
        return model

    def get_move(self, board):
        board = [board]
        valid_action_masks = self.sb.get_valid_actions(board)
        x, mask = np.array(board), np.array(valid_action_masks, dtype='float32')
        y = self.model.predict(x, verbose=self.verbose)
        action_index = np.argmax(y * mask, axis=1)[0]
        return self.index_to_move(action_index)

    def index_to_move(self, index):
        if 0 <= index < 729:
            row = index // 81
            col = (index % 81) // 9
            val = ((index % 81) % 9) + 1
            return row, col, val
        else:
            return None

if __name__ == "__main__":
    agent = SudokuAgent('sudoku_genie_1.8_15ep_5milboards_0.0001.h5')
    sb = SudokuBoard()
    board, solved, _ = sb.generate_board_pairs(num_boards=2, filename='sudoku-1m-kaggle-1.csv')
    board = [board[1]]
    solved = [solved[1]]
    print(solved)
    print(agent.get_move(board))
