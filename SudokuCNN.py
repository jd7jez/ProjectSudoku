import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from SudokuBoard import SudokuBoard
from time import time


def valid_action_mse(y_true, y_pred):
    mask = y_true[:, 729:]
    y_true = y_true[:, :729]
    return tf.reduce_mean(tf.square(y_true - y_pred) * mask)

def valid_action_accuracy(y_true, y_pred):
    mask = y_true[:, 729:]
    y_true = y_true[:, :729]

    tolerance = 0.01

    y_correct = tf.abs(y_pred - y_true) < tolerance
    y_correct = tf.cast(y_correct, tf.float32) * mask

    num_valid_actions = tf.reduce_sum(mask)
    num_correct = tf.reduce_sum(y_correct)
    return num_correct / num_valid_actions

class SudokuCNN:

    def __init__(self, lr=0.001, model_name=None):
        self.sb = SudokuBoard()
        self.model = self.get_model(lr)
        self.model_name = model_name if model_name != None else f"no_model_name_lr{lr}"

    def get_data(self, num_boards=100, filename=None, rewards=[5, 10, -5, -10]):
        unsolved, solved, missing = self.sb.generate_board_pairs(num_boards=num_boards, filename=filename)
        start = time.time()
        print(f"Preprocessing board data from {filename}.")
        reward_masks = [self.sb.get_actual_rewards_mask(unsol, sol) for unsol, sol in zip(unsolved, solved)]
        valid_action_masks = [self.sb.get_valid_actions(unsol) for unsol in unsolved]
        x, y, y_mask = np.array(unsolved), np.array(reward_masks, dtype='float32'), np.array(valid_action_masks, dtype='float32')
        y[y==0] = rewards[3]
        y[y==1] = rewards[0] # Needs to be changed depending on if this board has 1 empty square or more than 1
        y[y==2] = rewards[2]

        # Append valid action mask to ground truth rewards for y, this takes advantage of the custom loss function
        y = np.concatenate([y, y_mask], axis=1)
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))

        pivot = int(num_boards * 0.8)
        self.x_train, self.y_train = x[:pivot], y[:pivot]
        self.x_test, self.y_test = x[pivot:], y[pivot:]
        end = time.time()
        print(f"Finished preprocessing board data in {end - start} seconds.")

    def get_model(self, lr=0.001):
        model = Sequential([
            Conv2D(256, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
            Conv2D(512, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(81*9, activation='linear')
        ])
        # model = Sequential([
        #     Input(shape=(81, 9)),  # assuming the input is a 9x9 board with a single channel
        #     Reshape((9, 9, 9)),
        #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        #     BatchNormalization(),
        #     Flatten(),
        #     Dense(1024, activation='relu'),
        #     BatchNormalization(),
        #     Dropout(0.5),
        #     Dense(512, activation='relu'),
        #     BatchNormalization(),
        #     Dropout(0.5),
        #     Dense(81 * 9, activation='softmax'),
        #     Reshape((81, 9))
        # ])
        model.compile(optimizer=Adam(learning_rate=lr), loss=valid_action_mse, metrics=[valid_action_accuracy])
        return model

    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        # Train the model with the given data and specified hyperparameters
        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        #     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
        # ]
        callbacks = [CSVLogger(self.model_name+"_log.csv", append=True)]
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)

    def eval_model(self):
        # Evaluates the model against the given data
        return self.model.evaluate(self.x_test, self.y_test)

    def save_model(self):
        self.model.save_weights(self.model_name+".h5")

    def load_model(self, filename):
        self.model.load_weights(filename)

if __name__ == "__main__":
    # model_name = 'sudoku_singlevalplacer_1.61_12ep_4milboards'
    # scnn = SudokuCNN(model_name=model_name, lr=0.0001) # Was 0.0001 for epochs 1-6 and 10-12 then 0.00001 for epochs 7-9
    # scnn.load_model(filename='sudoku_singlevalplacer_1.6_9ep_3milboards.h5')
    # scnn.get_data(1000000, 'sudoku-1m-2missing-1.csv', rewards=[5, 10, -5, -10])
    # scnn.train_model(3, 32, 0.2)
    # scnn.eval_model()
    # scnn.save_model()

    model_name = 'sudoku_beginner_1.8_3ep_1milboards'
    scnn = SudokuCNN(model_name=model_name, lr=0.0001)
    scnn.get_data(1000000, 'sudoku-1m-10missing-1.csv', rewards=[10, 10, -10, -10])
    scnn.train_model(3, 32, 0.2)
    scnn.eval_model()
    scnn.save_model()

    model_name = 'sudoku_beginner_1.8_6ep_2milboards'
    scnn = SudokuCNN(model_name=model_name, lr=0.0001)
    scnn.load_model(filename='sudoku_beginner_1.8_3ep_1milboards.h5')
    scnn.get_data(1000000, 'sudoku-1m-10missing-2.csv', rewards=[10, 10, -10, -10])
    scnn.train_model(3, 32, 0.2)
    scnn.eval_model()
    scnn.save_model()
