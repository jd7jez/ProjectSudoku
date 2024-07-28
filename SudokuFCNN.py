import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from SudokuBoard import SudokuBoard


class SudokuFCNN:

    def __init__(self, width: int=3):
        self.width = width
        self.sb = SudokuBoard(self.width)

    def get_boards(self, num_boards=100):
        # Initialize the unsolved and solved boards
        unsolved = []
        solved = []

        # Get the number of desired boards
        for _ in range(num_boards):
            # Get the unsolved and solved boards in list form, replace None with 0's
            boards = self.sb.generate_board_pair()
            unsolved_board = [[val if val is not None else 0 for val in row] for row in boards[0]]
            # Append the unsolved and solved boards to running list
            unsolved.append(np.array(unsolved_board).flatten())
            solved.append(np.array(boards[1]).flatten())

        # Convert lists to numpy arrays
        unsolved = np.array(unsolved)
        solved = np.array(solved)

        # One hot encode the boards
        enc_unsolved = self.one_hot_encode(unsolved)
        enc_solved = self.one_hot_encode(solved)

        # Return the collection of solved and unsolved converted to tensors
        return tf.convert_to_tensor(enc_unsolved, dtype=tf.float32), tf.convert_to_tensor(enc_solved, dtype=tf.float32)

    # I am one-hot encoding the sudoku boar data because the values are basically just categorical
    # I don't want the model to interpret any sort of quantitative relationship between numbers, strictly ordinal
    def one_hot_encode(self, boards):
        # Create numpy array of proper shape
        enc_boards = np.zeros((boards.shape[0], 81, 9))
        # One hot encode the values
        for i, board in enumerate(boards):
            for j, box in enumerate(board):
                if box != 0:
                    enc_boards[i][j][box-1] = 1
        # Return one hot encoded boards
        return enc_boards

    def one_hot_decode(self, enc_boards):
        # Create numpy array of proper shape
        boards = np.zeros((enc_boards.shape[0], 81))
        # Decode the one-hot encoding
        for i, board in enumerate(enc_boards):
            for j, box in enumerate(board):
                for k, val in enumerate(box):
                    if val != 0.0:
                        boards[i][j] = k+1
        # Return one-hot decoded boards
        return boards

    def get_model(self, lr=0.001):
        # Return a Fully Connected Neural Network with relu activation, dropout, and softmax
        model = Sequential([
            Dense(256, activation='relu'),
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(9, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, num_boards, epochs=50, batch_size=32, validation_split=0.2):
        # Train the model with the desired number of boards and specified hyperparameters
        x_train, y_train = self.get_boards(num_boards)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)

    def eval_model(self, model, num_boards):
        # Evaluates the model against the desired number of boards
        x_test, y_test = self.get_boards(num_boards)
        return model.evaluate(x_test, y_test)


if __name__ == "__main__":
    sfcnn = SudokuFCNN()
    model = sfcnn.get_model(lr=0.01)
    sfcnn.train_model(model=model, num_boards=10000, epochs=50, batch_size=64)
    loss, accuracy = sfcnn.eval_model(model=model, num_boards=100)
    print(f"Achieved accuracy of {accuracy} and loss of {loss}")
    # x_train, y_train = sfcnn.get_boards(1)
    # print(x_train)
