import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y == 0, -1, 1)

        loss_history = []

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

            loss = (1 / 2) * np.dot(self.weights, self.weights) + self.lambda_param * np.sum(
                np.maximum(0, 1 - y_ * (np.dot(X, self.weights) + self.bias)))
            loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return loss_history

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.where(linear_output >= 0, 1, 0)
        return predictions

if __name__ == "__main__":
    numClass = 2
    numFeat = 2  # Change this to 2 for visualization
    classSep = 6.0
    learningRate = 0.001

    testDataX, testDataY = make_classification(
        n_features=numFeat,
        n_samples=5000,
        n_clusters_per_class=1,
        n_informative=numFeat,
        n_redundant=0,
        scale=5,
        n_classes=numClass,
        class_sep=classSep
    )

    xTrain, xTest, yTrain, yTest = train_test_split(
        testDataX, testDataY,
        test_size=0.05, random_state=10
    )

    model = LinearSVM(learning_rate=learningRate, lambda_param=0.01, n_iters=1000)
    loss_history = model.fit(xTrain, yTrain)

    predictions = model.predict(xTest)
    accuracy = np.mean(predictions == yTest) * 100
    print(f"Accuracy = {accuracy:.2f} %")

    # Plot loss history
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()  # This should display the loss plot

    # Plot decision boundary
    if numFeat == 2:
        def plot_decision_boundary(X, y, model):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
            plt.title("Decision Boundary")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.show()  # Ensure this shows the decision boundary plot

        plot_decision_boundary(xTrain, yTrain, model)
