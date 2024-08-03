import numpy as np
from sklearn.ensemble import RandomForestClassifier
from SudokuBoard import SudokuBoard
import time
import joblib


class SudokuRandomForest:

    def __init__(self, width: int=3):
        self.width = width
        self.sb = SudokuBoard(self.width)


    def get_boards(self, num_boards=100, filename=None):
        unsolved, solved = self.sb.generate_board_pairs(num_boards=num_boards, filename=filename)
        unsolved, solved = np.array(unsolved).reshape((num_boards, 81)), np.array(solved).reshape((num_boards, 81))
        return unsolved, solved

    def get_model(self, n_estimators=100, random_state=42):
        # Return a RandomForestClassifier model with the specified properties or the default ones
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train_model(self, model, x_train, y_train):
        # Train the model with the provided data
        model.fit(x_train, y_train)

    def eval_model(self, model, x_test, y_test):
        # Evaluates the model with the provided data
        y_pred = model.predict(x_test)
        return self.calc_accuracy(y_test, y_pred)

    def calc_accuracy(self, truth, pred):
        # Create a numpy array where all correct guesses are 0s and incorrect guesses are 1s
        scores = pred - truth
        scores[scores != 0] = 1

        # Get the number of incorrect guesses, then return what percentage were correct
        score = np.sum(scores)
        guesses = 81.0 * truth.shape[0]
        return (guesses - score) / guesses

    def save_model(self, model, filename):
        # Save the model to the provided filename
        joblib.dump(model, filename)

    def load_model(self, filename):
        # Load the model with the provided filename and return it
        model = joblib.load(filename)
        return model

if __name__ == "__main__":
    srf = SudokuRandomForest()
    model = srf.get_model(n_estimators=100)
    x, y = srf.get_boards(50000, 'sudoku-3m.csv')
    x_train, y_train = x[:40000], y[:40000]
    x_test, y_test = x[40000:], y[40000:]
    start = time.time()
    srf.train_model(model, x_train, y_train)
    end = time.time()
    # srf.save_model(model, "sudokurandomforest_250trees_50000boards.joblib")
    print(f"Accuracy after attempting 10000 puzzles ({end - start} sec) with 250 trees: {srf.eval_model(model, x_test, y_test)}")


    # for ests in [150, 175, 200, 225, 250]:
    #     model = srf.get_model(n_estimators=ests)
    #     start = time.time()
    #     srf.train_model(model, 10000)
    #     end = time.time()
    #     print(f"Accuracy after attempting 10000 puzzles ({end - start} sec) with {ests} trees: {srf.eval_model(model, 1000)}")

    # unsolved, _ = srf.get_boards(num_boards=100000, filename='sudoku-3m.csv')
    # unsolved[unsolved != None] = 1
    # unsolved[unsolved == None] = 0
    # sum = np.sum(unsolved)
    # print(sum / (81 * 100000))
