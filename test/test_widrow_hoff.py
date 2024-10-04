import os
import sys

from cases import test_cases
from sklearn.datasets import make_regression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from assignment2_neuralnetworks.generate_binary_data import generate_binary_data
from assignment2_neuralnetworks.widrow_hoff import WidrowHoff  # Assuming you've renamed it

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    for test_case in test_cases[:1]:
        # # Using assignment 1 testing data
        # X_train, y_train = generate_binary_data(
        #     test_case.case_condition, n_samples=100, n_features=test_case.n_features
        # )
        #
        # # Create a Widrow-Hoff object and fit to data with no bias (This show the curve of accuracy)
        # model = WidrowHoff(n_features=2, lr=0.0001, max_epochs=100, verbose=1, tol=1e-7)
        # model.fit(X_train, y_train)
        #
        # # Create a Widrow-Hoff object and fit to data with bias
        # model = WidrowHoff(n_features=2, lr=0.0001, max_epochs=1000, verbose=1, tol=1e-7, useBias=True)
        # model.fit(X_train, y_train)
        #
        #
        # # Generate synthetic regression data
        # X_train, y_train = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
        #
        # # Initialize the WidrowHoff model no Bias
        # model = WidrowHoff(n_features=2, lr=0.01, max_epochs=100, verbose=1)
        # model.fit(X_train, y_train)
        #
        # # Initialize the WidrowHoff model with bias
        # model = WidrowHoff(n_features=2, lr=0.01, max_epochs=100, verbose=1, useBias=True)
        # model.fit(X_train, y_train)

        # Load the breast cancer dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features for better convergence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the Widrow-Hoff model
        n_features = X_train.shape[1]
        widrow_hoff_model = WidrowHoff(n_features=n_features, lr=0.000001, tol=.00001, max_epochs=100000, verbose=0,
                                       useBias=True, plot=True)

        # Train the model
        widrow_hoff_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = widrow_hoff_model.forward(X_test_scaled)

        # Convert continuous output to binary labels
        y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

        # Evaluate the model's accuracy
        accuracy = np.mean(y_pred_binary == y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
