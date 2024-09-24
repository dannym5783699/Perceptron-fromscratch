import numpy as np


class WidrowHoff:
    def __init__(self, input_dim, learning_rate=0.01):
        # Initialize weights randomly with small values
        self.weights = np.random.randn(input_dim)
        self.learning_rate = learning_rate

    def predict(self, x):
        """
        Compute the linear prediction: y = w^T * x
        """
        return np.dot(self.weights, x)

    def update_weights(self, x, error):
        """
        Update weights based on the error and input x
        """
        self.weights += self.learning_rate * error * x

    def train(self, X, y, epochs=100):
        """
        Train the model using the training data X and target y
        for a specified number of epochs.
        """
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Get the prediction
                prediction = self.predict(X[i])
                # Calculate the error
                error = y[i] - prediction
                # Update weights based on error
                self.update_weights(X[i], error)
                # Keep track of the total error for this epoch
                total_error += error ** 2
            # Optionally, print the mean squared error for the epoch
            print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {total_error / len(X)}")

    def get_weights(self):
        """
        Return the current weights
        """
        return self.weights


# Example usage
if __name__ == "__main__":
    # Sample training data (X: input features, y: target values)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 4 samples, 2 features each
    y = np.array([5, 7, 9, 11])  # Target values

    # Initialize model with 2 input dimensions (matching the number of features in X)
    model = WidrowHoff(input_dim=2, learning_rate=0.01)

    # Train the model
    model.train(X, y, epochs=100)

    # Output the final weights
    print("Final Weights:", model.get_weights())