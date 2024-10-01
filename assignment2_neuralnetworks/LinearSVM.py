import numpy as np
import matplotlib.pyplot as plt


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the Linear SVM with:
        learning_rate: The rate at which the model learns
        lambda_param: Regularization parameter to prevent overfitting
        n_iters: Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the SVM model on data X and labels y using gradient descent.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) with labels either -1 or 1
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to {-1, 1} for SVM if needed
        y_ = np.where(y <= 0, -1, 1)

        # Gradient descent optimization for SVM
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the prediction is violating the margin condition
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    # No penalty, only apply regularization on weights
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Apply both regularization and penalty terms
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predict class labels for samples in X.
        X: numpy array of shape (n_samples, n_features)
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


# Plotting the decision boundary
def plot_decision_boundary(X, y, model):
    """
    Plots the decision boundary of the SVM model and the data points.
    """

    def get_hyperplane_value(x, w, b, offset):
        # Calculate the boundary line values
        return (-w[0] * x + b + offset) / w[1]

    fig, ax = plt.subplots()

    # Plot the points
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    # Get the decision boundary (w · x - b = 0)
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, model.weights, model.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, model.weights, model.bias, 0)

    # Plot the decision boundary line
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    # Plot the margins
    x1_1_margin_pos = get_hyperplane_value(x0_1, model.weights, model.bias, 1)
    x1_2_margin_pos = get_hyperplane_value(x0_2, model.weights, model.bias, 1)
    ax.plot([x0_1, x0_2], [x1_1_margin_pos, x1_2_margin_pos], 'r--')

    x1_1_margin_neg = get_hyperplane_value(x0_1, model.weights, model.bias, -1)
    x1_2_margin_neg = get_hyperplane_value(x0_2, model.weights, model.bias, -1)
    ax.plot([x0_1, x0_2], [x1_1_margin_neg, x1_2_margin_neg], 'r--')

    # Set limits for the plot
    ax.set_xlim([x0_1 - 0.5, x0_2 + 0.5])
    ax.set_ylim([np.amin(X[:, 1]) - 0.5, np.amax(X[:, 1]) + 0.5])

    plt.show()


# Example usage of Linear SVM
if __name__ == "__main__":
    # Create a simple linearly separable dataset
    X = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, 1, -1, -1])

    # Initialize and train the SVM model
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)

    # Make predictions
    predictions = svm.predict(X)
    print("Predictions:", predictions)
    print("Weights:", svm.weights)
    print("Bias:", svm.bias)

    # Plot the decision boundary
    plot_decision_boundary(X, y, svm)
