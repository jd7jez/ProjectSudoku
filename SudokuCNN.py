import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from SudokuBoard import SudokuBoard


class SudokuCNN:

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
        # Return a Convolutional Neural Network with relu activation, batchnorm, dropout, and softmax
        # model = Sequential([
        #     Input((81, 9)),
        #     Reshape((9, 9, 9)),
        #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Flatten(),
        #     Dense(256, activation='relu'),
        #     BatchNormalization(),
        #     Dropout(0.3),
        #     Dense(256, activation='relu'),
        #     BatchNormalization(),
        #     Dropout(0.3),
        #     Dense(81 * 9, activation='softmax'),
        #     Reshape((81, 9))
        # ])
        model = Sequential([
            Input(shape=(81, 9)),  # assuming the input is a 9x9 board with a single channel
            Reshape((9, 9, 9)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(81 * 9, activation='softmax'),
            Reshape((81, 9))
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        # Train the model with the given data and specified hyperparameters
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
        ]
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)

    def eval_model(self, model, x_test, y_test,):
        # Evaluates the model against the given data
        return model.evaluate(x_test, y_test)


if __name__ == "__main__":
    scnn = SudokuCNN()
    model = scnn.get_model(lr=0.01)
    x_train, y_train, x_test, y_test = scnn.get_boards(500000, 'sudoku-3m.csv')
    print(x_train.shape)
    print(x_test.shape)
    scnn.train_model(model=model, x_train=x_train, y_train=y_train, epochs=50, batch_size=128)
    loss, accuracy = scnn.eval_model(model=model, x_test=x_test, y_test=y_test)
    print(f"Achieved accuracy of {accuracy} and loss of {loss}")