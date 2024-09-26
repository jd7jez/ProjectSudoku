from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from SudokuGame import SudokuGame
from SudokuBoard import SudokuBoard
import numpy as np
import time
import matplotlib.pyplot as plt
from PrioritizedMemory import PrioritizedMemory

class SudokuDQN:

    def __init__(self, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, learning_rate=0.001, rewards=[500, 1000, -500, -10000]):
        self.state_size = (9, 9)
        self.actions = [(row, col, val) for row in range(9) for col in range(9) for val in range(1, 10)]
        self.action_size = 81 * 9
        self.memory = PrioritizedMemory()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.rewards = rewards
        self.model = self.cnn_model(lr=self.learning_rate)
        self.reward_history = []
        self.steps_history = []
        self.error_history = []
        self.success_history = []

    def fcc_model(self, lr=0.001):
        model = Sequential()
        model.add(Flatten(input_shape=(81, 10)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def cnn_model(self, lr=0.001):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
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

        boards = np.zeros((batch_size, 9, 9, 1))
        next_boards = np.zeros((batch_size, 9, 9, 1))
        target_fs = np.zeros((batch_size, self.action_size))
        sample_weights = np.zeros(batch_size)

        for i, (board, action, reward, next_board, done, rewards_mask) in enumerate(experiences):
            boards[i] = board
            next_boards[i] = next_board

            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_board, verbose=0)[0])
            target_f = self.model.predict(board, verbose=0)
            errors[i] = abs(target_f[0][action] - target)
            # target_f[0][action] = target     Commented this line out to try using reward mask
            target_f[rewards_mask==1] = self.rewards[1]
            target_f[rewards_mask==2] = self.rewards[2]
            target_fs[i] = target_f
            sample_weights[i] = weights[i]

        self.model.fit(boards, target_fs, sample_weight=sample_weights, epochs=1, verbose=0)

        self.memory.update_priorities(indices, errors)

        return np.sum(errors) / batch_size

    def get_actual_rewards_mask(self, board, solved):
        rewards = np.zeros((1, self.action_size))
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val == 0:
                    first_action = (y * 81) + (x * 9)
                    correct_action = first_action + (solved[y][x] - 1)
                    for i in range(first_action, first_action+9):
                        rewards[0][i] = 2 if i != correct_action else 1
        return rewards

    def train(self, filename=None, num_boards=50, batch_size=32, one_hot=False):
        game = SudokuGame(preload=True, filename=filename, num_boards=num_boards, reward=True, rewards=self.rewards, verbose=0)
        sb = SudokuBoard()

        # Tracks every move made whether it was a successful move (1.0) or unsuccessful (0.0)
        successes = np.array([])

        for n in range(num_boards):
            game.setBoard()
            valid_actions_mask = self.get_valid_actions(game.get_current())
            rewards_mask = self.get_actual_rewards_mask(game.get_current(), game.get_solved())
            board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81))) if one_hot else np.float32(np.reshape(game.get_current(), (1, 9, 9, 1)))

            # Delete these lines later by making all values 0 instead of None in SudokuBoard
            board = np.nan_to_num(board)

            total_reward = 0
            start = time.time()
            steps = 0

            for t in range(1):
                steps += 1
                action_index = self.act(board, valid_actions_mask)
                action = self.actions[action_index]
                row, col, val = action
                reward, code = game.makeMove(row, col, val)
                next_board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81))) if one_hot else np.float32(np.reshape(game.get_current(), (1, 9, 9, 1)))

                # Delete this line later by making all values 0 instead of None in SudokuBoard
                next_board = np.nan_to_num(next_board)

                done = code == 1
                total_reward += reward
                self.memory.memorize((board, action_index, reward, next_board, done, rewards_mask))

                if code == 2:
                    successes = np.append(successes, 0.0)
                    game.undo_move()
                else:
                    successes = np.append(successes, 1.0)
                board = sb.one_hot_encode(np.reshape(game.get_current(), (1, 81))) if one_hot else np.float32(np.reshape(game.get_current(), (1, 9, 9, 1)))

                # Delete this line later by making all values 0 instead of None in SudokuBoard
                board = np.nan_to_num(board)

                if self.memory.memory_size > batch_size:
                    error = self.replay(batch_size)
                    self.error_history.append(error)

                if done:
                    print(f"SUCCESSFULLY SOLVED BOARD {n + 1}/{num_boards} finished in {t + 1} steps with total reward {total_reward}")
                    break

            self.reward_history.append(total_reward)
            self.steps_history.append(steps)
            # The success rate is the rate of success from the last 300 moves
            success_cutoff = 0 if successes.shape[0] <= 300 else successes.shape[0] - 300
            success_rate = np.average(successes[success_cutoff:])
            self.success_history.append(success_rate)
            curr_error = self.error_history[-1] if len(self.error_history) > 0 else 0

            end = time.time()
            print(f"Finished training on board {n + 1} in {end - start} seconds\n"
                  f"Epsilon: {self.epsilon}\n"
                  f"Reward: {total_reward}\n"
                  f"Error: {curr_error}\n"
                  f"Success Rate: {success_rate}\n")
            print("-------------------------------------------------------")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if (n + 1) % 50 == 0:
                self.save_model(f"sudoku-dqn1.5-lr{self.learning_rate}-{n+1}ep.h5")
                self.save_history(f"dqn-history1.5-lr{self.learning_rate}-{n+1}ep.txt")

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
            errors_line = file.readline()
            successes_line = file.readline()
            rewards = map(int, rewards_line.split(','))
            steps = map(int, steps_line.split(','))
            errors = map(float, errors_line.split(','))
            successes = map(float, successes_line.split(','))
            self.reward_history = list(rewards)
            self.steps_history = list(steps)
            self.error_history = list(errors)
            self.success_history = list(successes)

    def save_history(self, filename):
        with open(filename, 'w') as file:
            file.write(','.join(map(str, self.reward_history)) + '\n')
            file.write(','.join(map(str, self.steps_history)) + '\n')
            file.write(','.join(map(str, self.error_history)) + '\n')
            file.write(','.join(map(str, self.success_history)) + '\n')

    def plot_history(self):
        self.plot_reward_history()
        self.plot_steps_history()
        self.plot_error_history()
        self.plot_success_history()

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

    def plot_error_history(self):
        if len(self.error_history) <= 0:
            print("Cannot plot error history with no history")
            return
        x_vals = range(len(self.error_history))
        plt.plot(x_vals, self.error_history)
        plt.title("Average Error per session History")
        plt.grid(True)
        plt.show()

    def plot_success_history(self):
        if len(self.reward_history) <= 0:
            print("Cannot plot success history with no history")
            return
        x_vals = range(len(self.success_history))
        plt.plot(x_vals, self.success_history)
        plt.title("Rate of Success History")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    model = SudokuDQN(epsilon=0.0, epsilon_decay=0.995, gamma=0, learning_rate=0.01)
    model.train(filename='sudoku-10k-1missing.csv', num_boards=10000)
    model.plot_history()
