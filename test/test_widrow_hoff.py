import os
import sys

from cases import test_cases
from sklearn.datasets import make_regression

from assignment2_neuralnetworks.generate_binary_data import generate_binary_data
from assignment2_neuralnetworks.widrow_hoff import WidrowHoff  # Assuming you've renamed it

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    for test_case in test_cases[:1]:
        # Using assignment 1 testing data
        X_train, y_train = generate_binary_data(
            test_case.case_condition, n_samples=100, n_features=test_case.n_features
        )

        # Create a Widrow-Hoff object and fit to data with no bias (This show the curve of accuracy)
        model = WidrowHoff(n_features=2, lr=0.0001, max_epochs=100, verbose=1, tol=1e-7)
        model.fit(X_train, y_train)

        # Create a Widrow-Hoff object and fit to data with bias
        model = WidrowHoff(n_features=2, lr=0.0001, max_epochs=1000, verbose=1, tol=1e-7, useBias=True)
        model.fit(X_train, y_train)


        # Generate synthetic regression data
        X_train, y_train = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

        # Initialize the WidrowHoff model no Bias
        model = WidrowHoff(n_features=2, lr=0.01, max_epochs=100, verbose=1)
        model.fit(X_train, y_train)

        # Initialize the WidrowHoff model with bias
        model = WidrowHoff(n_features=2, lr=0.01, max_epochs=100, verbose=1, useBias=True)
        model.fit(X_train, y_train)


if __name__ == "__main__":
    main()
