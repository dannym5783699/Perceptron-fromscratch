from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import numpy as np
class WestonWatkinsSVM:
    def __init__(self, n_features, n_classes, lr = 0.1):
        #create a (classes, inDem) weight array for all output neurons depending on inputs and classes.
        self._W = np.random.rand(n_classes, n_features)
        #create a random bias vector
        self._b = np.random.rand(n_classes)
        self._lr = lr
        self.n_classes = n_classes


    def predict(self, x):
        return np.dot(x, self._W.T) + self._b
    
    def forward(self, x):
        return np.argmax(self.predict(x), axis = 1)
    
    def fit(self, x : np.ndarray, y : np.ndarray, epoch = 1000):
        #Loop over all points for each epoch
        for m in range(epoch):
            if(m % 10 == 0):
                print("Epoch: ", m)
            #expects a column vector for x and y.
            for i in range(len(y)):
                x_i = x[i]
                
                y_hat = self._W @ x_i.T + self._b

                y_hat_i = y_hat[y[i]]                           # seperator of true class
                loss_terms = np.maximum(y_hat - y_hat_i + 1, 0) # Compute the loss for each class separator
                loss_terms[y[i]] = 0                            # Set the loss of the true class to zero
                delta = np.where(loss_terms > 0, 1, 0)          # Indicator function for each r

                # Set the delta for the true class to be the sum of all other deltas for the update 
                delta[y[i]] = -np.sum(delta)

                # Update weights
                self._W -= self._lr * np.outer(delta, x_i)
                self._b -= self._lr * delta


if __name__ == "__main__":

    LR = 1e-4

    X, y = load_iris(return_X_y=True)
    n_features = X.shape[1]
    n_classes = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=10, stratify=y)

    print("Training data shape: ", X_train.shape)
    print("Testing data shape: ", X_test.shape)
    print("Number of classes: ", n_classes)

    # Train model    
    model = WestonWatkinsSVM(n_features, n_classes, LR)
    model.fit(X_train, y_train, 500)

    # Check test accuracy
    y_pred = model.forward(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
