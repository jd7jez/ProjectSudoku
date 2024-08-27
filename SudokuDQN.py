from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class SudokuDQN:

    def __init__(self, env):
        self.env = env

    def get_model(self, lr=0.001):
        model = Sequential()
        model.add(Dense(128, input_shape=(81,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.env.action_space, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def train_model(self, model, ):