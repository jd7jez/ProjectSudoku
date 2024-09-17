from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from SudokuGame import SudokuGame
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class SudokuDQN:

    def __init__(self, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = (9, 9)
        self.actions = [(row, col, val) for row in range(9) for col in range(9) for val in range(1, 10)]
        self.action_size = 81 * 9
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.fresh_model(lr=self.learning_rate)
        self.reward_history = []
        self.steps_history = []

    def fresh_model(self, lr=0.001):
        model = Sequential()
        model.add(Flatten(input_shape=(9, 9)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def memorize(self, board, action, reward, next_state, done):
        self.memory.append((board, action, reward, next_state, done))

    def act(self, board):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(board, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for board, action, reward, next_board, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_board, verbose=0)[0]))
            target_f = self.model.predict(board, verbose=0)
            target_f[0][action] = target
            self.model.fit(board, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, filename=None, rewards=[10, 50, -10, -50], num_boards=50, batch_size=32):
        game = SudokuGame(preload=True, filename=filename, num_boards=num_boards, reward=True, rewards=rewards, verbose=0)

        for n in range(100, num_boards+100):
            game.setBoard()
            board = np.reshape(game.get_current(), (1, 9, 9)).astype('float32')
            total_reward = 0
            start = time.time()
            steps = 0

            for t in range(25):
                steps += 1
                action_index = self.act(board)
                action = self.actions[action_index]
                row, col, val = action
                reward, code = game.makeMove(row, col, val)
                next_board = np.reshape(game.get_current(), (1, 9, 9)).astype('float32')
                done = code == 1
                total_reward += reward
                self.memorize(board, action_index, reward, next_board, done)

                if code == 2:
                    game.undo_move()
                board = np.reshape(game.get_current(), (1, 9, 9)).astype('float32')

                if done:
                    print(f"Episode {n + 1}/{num_boards} finished in {t + 1} steps with total reward {total_reward}")
                    break

                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            self.reward_history.append(total_reward)
            self.steps_history.append(steps)

            end = time.time()
            print(f"Finished training on board {n + 1} in {end - start} seconds with a reward of {total_reward}")

            if (n + 1) % 10 == 0:
                self.save_model(f"sudoku-dqn-{n+1}.h5")

    def set_model(self, model):
        self.model = model

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

    def get_reward_history(self):
        return self.reward_history

    def get_steps_history(self):
        return self.steps_history

    def load_history(self, filename):
        rewards = []
        steps = []
        with open(filename, 'r') as file:
            rewards_line = file.readline()
            steps_line = file.readline()
            rewards = map(int, rewards_line.split(','))
            steps = map(int, steps_line.split(','))
        return list(rewards), list(steps)

    def save_history(self, filename):
        with open(filename, 'w') as file:
            file.write(','.join(map(str, self.reward_history)) + '\n')
            file.write(','.join(map(str, self.steps_history)) + '\n')


    def plot_reward_history(self):
        if len(self.reward_history) <= 0:
            print("Cannot plot reward history with no history")
            return
        x_vals = range(len(self.reward_history))
        plt.plot(x_vals, self.reward_history)
        plt.title("Reward per session History")
        plt.grid(True)
        plt.show()

    def plot_steps_history(self):
        if len(self.reward_history) <= 0:
            print("Cannot plot steps history with no history")
            return
        x_vals = range(len(self.steps_history))
        plt.plot(x_vals, self.steps_history)
        plt.title("Steps to finish session History")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    model = SudokuDQN(epsilon=0.5)
    model.load_model('sudoku-dqn-100.h5')
    model.train(filename='sudoku-10k-1missing.csv', num_boards=100)
    model.save_history('dqn-history-200ep.txt')
    model.plot_reward_history()
    model.plot_steps_history()
