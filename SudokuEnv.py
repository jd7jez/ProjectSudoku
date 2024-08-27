import numpy as np
import gym
from gym import spaces

class SudokuEnv(gym.Env):

    def __init__(self, sg):
        super(SudokuEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=9, shape=(81,), dtype=np.int)
        self.action_space = spaces.Discrete(81 * 9)
        self.sg = sg

    def step(self, action):
        # Move Codes
        # 0: Correct Guess
        # 1: Correct Guess that finishes board
        # 2: Incorrect Guess
        # 3: Invalid Guess that defies rules
        # 4: Correct Guess that finishes all boards

        cell = action // 9
        row = cell // 9
        col = cell % 9
        val = (action % 9) + 1

        reward, move_code = self.sg.makeMove(row, col, val)
        if move_code == 1:
            if self.sg.has_loaded_board():
                self.sg.setBoard()
            else:
                move_code = 4
        elif move_code == 2:
            self.sg.undo_move()

        return reward, move_code