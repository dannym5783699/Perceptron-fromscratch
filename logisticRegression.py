from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize the Logistic Regression model with:
        learning_rate: The rate at which the model learns
        epochs: Number of complete passes through the training data
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
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
        for epoch in range(self.epochs):
            # Compute linear model and predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optionally, print the cost for every 100 epochs for better insight
            if epoch % 100 == 0:
                cost = -(1 / n_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
                print(f"Epoch {epoch}/{self.epochs}, Cost: {cost}")

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
    # Generate a synthetic dataset for binary classification
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42, class_sep = 3)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    lr = LogisticRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = lr.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test) * 100
    print("\nAccuracy:", accuracy, "%")
    print("Weights:", lr.weights)
    print("Bias:", lr.bias)
