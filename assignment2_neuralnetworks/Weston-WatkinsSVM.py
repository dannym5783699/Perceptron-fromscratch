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


    def outVals(self, x):
        #Get vector of predictions.
        y_hat = np.dot(self._W, x)
        #Add bias and reshape for final class prediction and loss.
        return (y_hat.reshape(-1, 1) + self._b)
    
    
    def forward(self, x):
        out = self.outVals(x)
        #Pick the neuron with highest output value as the class prediction.
        return np.argmax(out)
    
    #determine individual loss contribution from each output neuron for a specific sample.
    def indLoss(self, y, y_hat):
        loss = np.zeros(self.n_classes)
        #get loss contribution of each neuron, correct neuron is set to 0.
        for i in range(self.n_classes):
            if(i != y):
                loss[i] = max(0, y_hat[i, 0] - y_hat[int(y), 0] + 1)
            else:
                #Loss contribution for the correct neuron is set to zero.
                loss[i] = 0;  
        return loss
    

    
    def fit(self, x, y, epoch = 1000):
        #Loop over all points for each epoch
        for m in range(epoch):
            if(m % 10 == 0):
                print("Epoch: ", m)
            #expects a column vector for x and y.
            for i in range(len(y)):
                #get column vector for a single sample.
                x_i = x[:,i]
                #get the outputs of the output neurons.
                y_hat = self.outVals(x_i)
                #get the array of loss contributions for weight updates.
                lossContr = self.indLoss(y[i], y_hat)
                #Using the indicator function on each element.
                indicator = np.where(lossContr > 0, 1, 0).reshape(-1,1)
                #get the sum of the indicator elements for correct neuron weight updates.
                sIndicator = np.sum(indicator)
                #update the weights and bias.
                for w in range(self.n_classes):
                    if(w == y[i]):
                        self._b[w, 0] = self._b[w, 0] + ((self._lr)*(sIndicator))
                        self._W[w, :] = self._W[w, :] + ((self._lr*x_i.T)*(sIndicator))
                    else:
                        self._b[w, 0] = self._b[w, 0] - ((self._lr)*(indicator[w, 0]))
                        self._W[w, :] = self._W[w, :] - ((self._lr*x_i.T)*(indicator[w, 0]))

                #TODO: uncomment bias and update them.



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






        




