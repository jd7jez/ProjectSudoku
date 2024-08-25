import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Reshape, Input, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from SudokuBoard import SudokuBoard

class SudokuLSTM:

    def __init__(self, width: int=3):
        self.width = width
        self.sb = SudokuBoard(self.width)

    def get_boards(self, num_boards=100, filename=None):
        unsolved, solved = self.sb.generate_board_pairs(num_boards=num_boards, filename=filename)
        unsolved, solved = np.array(unsolved).reshape((num_boards, 81)), np.array(solved).reshape((num_boards, 81))
        unsolved[unsolved == None] = 0

        # One hot encode the boards
        enc_unsolved = self.one_hot_encode(unsolved)
        enc_solved = self.one_hot_encode(solved)

        # Split into train and test data, 80% train and 20% test. Also convert to tensors
        pivot = int(num_boards * 0.8)
        x_train, y_train = tf.convert_to_tensor(enc_unsolved[:pivot], dtype=tf.float32), tf.convert_to_tensor(
            enc_solved[:pivot], dtype=tf.float32)
        x_test, y_test = tf.convert_to_tensor(enc_unsolved[pivot:], dtype=tf.float32), tf.convert_to_tensor(
            enc_solved[pivot:], dtype=tf.float32)

        # Return the collection of solved and unsolved converted to tensors
        return x_train, y_train, x_test, y_test

        # I am one-hot encoding the sudoku boar data because the values are basically just categorical
        # I don't want the model to interpret any sort of quantitative relationship between numbers, strictly ordinal
    def one_hot_encode(self, boards):
        # Create numpy array of proper shape
        enc_boards = np.zeros((boards.shape[0], 81, 9))
        # One hot encode the values
        for i, board in enumerate(boards):
            for j, box in enumerate(board):
                if box != 0:
                    enc_boards[i][j][box - 1] = 1
        # Return one hot encoded boards
        return enc_boards

    def get_model(self):
        model = Sequential([
            Input(shape=(81, 9)),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            LSTM(56, return_sequences=True),
            TimeDistributed(Dense(128, activation='relu')),
            TimeDistributed(Dense(9, activation='softmax'))
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, x_train, y_train, epochs=50, batch_size=64, validation_split=0.2):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        ]

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)

    def eval_model(self, model, x_test, y_test,):
        # Evaluates the model against the given data
        return model.evaluate(x_test, y_test)

if __name__ == "__main__":
    slstm = SudokuLSTM()
    model = slstm.get_model()
    x_train, y_train, x_test, y_test = slstm.get_boards(100000, 'sudoku-3m.csv')
    slstm.train_model(model=model, x_train=x_train, y_train=y_train, epochs=50, batch_size=64)
    loss, accuracy = slstm.eval_model(model=model, x_test=x_test, y_test=y_test)
    print(f"Achieved accuracy of {accuracy} and loss of {loss}")