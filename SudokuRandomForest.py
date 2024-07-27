import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import SudokuBoard as sb
import random

def get_boards(num_boards=100):
    # Initialize the unsolved and solved boards
    unsolved = []
    solved = []

    # Get the number of desired boards
    for i in range(num_boards):
        # Save the boards to the running collection of boards
        boards = sb.generate_board_pair(random.uniform(0.15, 0.7))
        unsolved.append(np.array(boards[0]).flatten())
        solved.append(np.array(boards[1]).flatten())

    # Return the collection of solved and unsolved
    return np.array(unsolved), np.array(solved)

def train_model(model, num_boards):
    # Train the model with the desired number of boards
    x_train, y_train = get_boards(num_boards)
    model.fit(x_train, y_train)

def eval_model(model, num_boards):
    # Evaluates the model against the desired number of boards
    x_test, y_test = get_boards(num_boards)
    y_pred = model.predict(x_test)
    print(y_pred)
    print(y_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_model(model, 3)
    print(eval_model(model, 3))