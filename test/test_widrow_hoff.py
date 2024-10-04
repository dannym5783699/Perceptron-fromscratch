import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cases import test_cases
from sklearn.datasets import make_regression

from assignment2_neuralnetworks.generate_binary_data import generate_binary_data
from assignment2_neuralnetworks.widrow_hoff import WidrowHoff  # Assuming you've renamed it

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    for test_case in test_cases[:1]:
        X_train, y_train = generate_binary_data(
            test_case.case_condition, n_samples=100, n_features=test_case.n_features
        )

        # X_test, y_test = generate_binary_data(
        #     test_case.case_condition, n_features=test_case.n_features, seed=90
        # )

        # Create a Widrow-Hoff object and fit to data
        model = WidrowHoff(n_features=2, lr=0.01, max_epochs=5, verbose=1, tol=1e-7)
        model.fit(X_train, y_train)

        # Generate synthetic regression data
        X_train, y_train = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

        # Initialize the WidrowHoff model
        model = WidrowHoff(n_features=2, lr=0.01, max_epochs=100, verbose=1)

        # Train the model using the generated data
        model.fit(X_train, y_train)

        # # Plot the decision boundary
        # x1 = np.linspace(-10, 10, 100)
        # x2 = (-model._W[0] * x1 - model._b) / model._W[1]
        # plt.plot(x1, x2, label="Decision Boundary")
        #
        # # Test the model
        # y_pred_continuous = model.forward(X_test)
        #
        # # Convert continuous predictions to binary class labels (thresholding at 0)
        # y_pred = np.where(y_pred_continuous >= 0, 1, -1)
        #
        # # Identify the misclassified samples
        # misclassified = y_test != y_pred
        # percent_misclassified = (misclassified.sum() / len(y_test)) * 100
        #
        # print(f"Percent Misclassified: {percent_misclassified:.2f}%")
        #
        # # Plot the test data points and the decision boundary
        # ax = sns.scatterplot(
        #     x=X_test[:, 0], y=X_test[:, 1], hue=y_test, style=misclassified
        # )
        # ax.set_xlabel("Feature 1")
        # ax.set_ylabel("Feature 2")
        #
        # plt.show()


if __name__ == "__main__":
    main()
