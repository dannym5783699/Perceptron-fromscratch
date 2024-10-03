import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initialize the Logistic Regression model with:
        learning_rate: The rate at which the model learns
        n_iters: Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Compute the sigmoid function of z.
        z: Input value, numpy array
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the Logistic Regression model on data X and labels y using gradient descent.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) with binary labels 0 or 1
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.n_iters):
            # Compute linear model and predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for samples in X.
        X: numpy array of shape (n_samples, n_features)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


# Example usage of Logistic Regression
if __name__ == "__main__":
    # Create a simple dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, 1, 0, 0])

    # Initialize and train the Logistic Regression model
    lr = LogisticRegression(learning_rate=0.01, n_iters=1000)
    lr.fit(X, y)

    # Make predictions
    predictions = lr.predict(X)
    print("Predictions:", predictions)
    print("Weights:", lr.weights)
    print("Bias:", lr.bias)
