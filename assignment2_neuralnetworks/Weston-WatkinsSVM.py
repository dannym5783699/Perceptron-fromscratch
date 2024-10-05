from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles

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

                # Set the delta for the true class to be the sum of all other deltas 
                # We can remove the correct class with - 1 since y_hat - y_hat_i == 0 for the true class
                delta[y[i]] = -np.sum(delta)

                # Update weights
                self._W -= self._lr * np.outer(delta, x_i)
                self._b -= self._lr * delta


if __name__ == "__main__":
    #testing obviously seperable data.
    #x1 = np.array([[2], [20]])
    #y1 = np.array([0])
    #model.fit(x1, y1)
    #testing some obvious points, first should be 0, second should be 1.


    
    #Change number of classes and features here.
    numClass = 5
    numFeat = 10
    classSep = 2.0
    learningRate = 0.0005

    testDataX, testDataY = make_classification(n_features=numFeat, n_samples=50000, n_clusters_per_class=1,
                                                   n_informative=numFeat-5,
                                                   n_redundant=1, scale=5, n_classes=numClass, class_sep=classSep)

    xTrain, xTest, yTrain, yTest = train_test_split(testDataX, testDataY,
                                                    test_size=0.05, random_state=10)
    
    model3 = WestonWatkinsSVM(numFeat,numClass, learningRate)
    xTrain = xTrain.T
    xTest = xTest.T
    model3.fit(xTrain[:numFeat], yTrain.T, 100)
    totalCorrect = 0
    for i in range(len(yTest)):
        if(model3.forward(xTest[:, i]) == yTest[i]):
            totalCorrect += 1

    print("accuracy = ", (totalCorrect/len(yTest)*100), "%")        






        




