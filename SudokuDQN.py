from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from SudokuGame import SudokuGame
from SudokuBoard import SudokuBoard
import numpy as np
import time
import matplotlib.pyplot as plt
from PrioritizedMemory import PrioritizedMemory

class SudokuDQN:

    def __init__(self, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = (9, 9)
        self.actions = [(row, col, val) for row in range(9) for col in range(9) for val in range(1, 10)]
        self.action_size = 81 * 9
        self.memory = PrioritizedMemory()
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
        model.add(Flatten(input_shape=(81, 10)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def act(self, board, valid_actions_mask):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.where(valid_actions_mask == 1)[0])
        predictions = self.model.predict(board, verbose=0)[0]
        predictions[valid_actions_mask == 0] = float('-inf')
        return np.argmax(predictions)

    def replay(self, batch_size):
        experiences, indices, weights = self.memory.sample(batch_size=batch_size)
        errors = np.zeros(batch_size)
        for i, (board, action, reward, next_board, done) in enumerate(experiences):
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_board, verbose=0)[0])
            target_f = self.model.predict(board, verbose=0)
            errors[i] = abs(target_f[0][action] - target)
            target_f[0][action] = target
            self.model.fit(board, target_f, sample_weight=np.reshape(weights[i], (1,)), epochs=1, verbose=0)

        self.memory.update_priorities(indices, errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, filename=None, rewards=[10, 50, -10, -50], num_boards=50, batch_size=32):
        game = SudokuGame(preload=True, filename=filename, num_boards=num_boards, reward=True, rewards=rewards, verbose=0)
        sb = SudokuBoard()

        for n in range(num_boards):
            game.setBoard()
            valid_actions_mask = self.get_valid_actions(game.get_current())
            board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81)))
            total_reward = 0
            start = time.time()
            steps = 0

            for t in range(5):
                steps += 1
                action_index = self.act(board, valid_actions_mask)
                action = self.actions[action_index]
                row, col, val = action
                reward, code = game.makeMove(row, col, val)
                next_board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81)))
                done = code == 1
                total_reward += reward
                self.memory.memorize((board, action_index, reward, next_board, done))

                if code == 2:
                    game.undo_move()
                board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81)))

                if self.memory.memory_size > batch_size:
                    self.replay(batch_size)

                if done:
                    print(f"SUCCESSFULLY SOLVED BOARD {n + 1}/{num_boards} finished in {t + 1} steps with total reward {total_reward}")
                    break
            self.reward_history.append(total_reward)
            self.steps_history.append(steps)

            end = time.time()
            print(f"Finished training on board {n + 1} in {end - start} seconds with a reward of {total_reward}")

            if (n + 1) % 50 == 0:
                self.save_model(f"sudoku-dqn1.1-lr{self.learning_rate}-{n+1}.h5")
                self.save_history(f"dqn-history1.1-lr{self.learning_rate}-{n+1}ep.txt")

    def get_valid_actions(self, board):
        valid_actions = np.ones((self.action_size))
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val != None:
                    first_action = (y * 81) + (x * 9)
                    for i in range(first_action, first_action+9):
                        valid_actions[i] = 0
        return valid_actions

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
        with open(filename, 'r') as file:
            rewards_line = file.readline()
            steps_line = file.readline()
            rewards = map(int, rewards_line.split(','))
            steps = map(int, steps_line.split(','))
            self.reward_history = list(rewards)
            self.steps_history = list(steps)

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
    for lr in [0.0001, 0.00005, 0.00001]:
        model = SudokuDQN(epsilon_decay=0.995, gamma=0, learning_rate=lr)
        model.train(filename='sudoku-10k-1missing.csv', num_boards=1000)
