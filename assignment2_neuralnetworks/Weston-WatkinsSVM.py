from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles

import numpy as np
class WestonWatkinsSVM:
    def __init__(self, inDem, classes, learnRate = 0.1):
        #create a (classes, inDem) weight array for all output neurons depending on inputs and classes.
        self.weights = np.random.rand(classes, inDem)
        #create a random bias vector
        self.bias = np.random.rand(classes, 1)
        self.learnRate = learnRate
        self.classes = classes


    def outVals(self, x):
        #Get vector of predictions.
        predicts = np.dot(self.weights, x)
        return predicts.reshape(-1, 1)
    
    def forward(self, x):
        out = self.outVals(x)
        #Pick the neuron with highest output value as the class prediction.
        return np.argmax(out)
    
    #determine individual loss contribution from each output neuron for a specific sample.
    def indLoss(self, y, predictions):
        loss = np.zeros(self.classes)
        #get loss contribution of each neuron, correct neuron is set to 0.
        for i in range(self.classes):
            if(i != y):
                loss[i] = max(0, predictions[i, 0] - predictions[int(y), 0] + 1)
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
                sample = x[:,i]
                #get the outputs of the output neurons.
                predictions = self.outVals(sample)
                #get the array of loss contributions for weight updates.
                lossContr = self.indLoss(y[i], predictions)
                #Using the indicator function on each element.
                indicator = np.where(lossContr > 0, 1, 0).reshape(-1,1)
                #get the sum of the indicator elements for correct neuron weight updates.
                sIndicator = np.sum(indicator)
                #update the weights
                for w in range(self.classes):
                    if(w == y[i]):
                        self.weights[w, :] = self.weights[w, :] + ((self.learnRate*sample.T)*(sIndicator))
                    else:
                        self.weights[w, :] = self.weights[w, :] - ((self.learnRate*sample.T)*(indicator[w, 0]))

                #TODO: uncomment bias and update them.



if __name__ == "__main__":
    #testing obviously seperable data.
    path = r'docs\wwSVMtestData.csv'
    data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)
    x = np.array([[2, 3, 4, 1, 10, 12, 13], [20, 21, 19, 18, 50, 51, 52]])
    y = np.array([0,0,0,0,1,1,1])
    model = WestonWatkinsSVM(2, 2)
    model.fit(x, y)
    #x1 = np.array([[2], [20]])
    #y1 = np.array([0])
    #model.fit(x1, y1)
    #testing some obvious points, first should be 1, second should be 0.
    print(model.forward([[2], [20]]))
    print(model.forward([[11],[53]]))
    print(model.forward([[3],[19]]))
    model2 = WestonWatkinsSVM(2,2)
    data = data.T
    model2.fit(data[:2], data[-1, :])
    print(model2.forward([[60],[5]]))

    
    #Change number of classes and features here.
    numClass = 3
    numFeat = 4
    classSep = 5.0

    testDataX, testDataY = make_classification(n_features=numFeat, n_samples=10000, n_clusters_per_class=1,
                                                   n_informative=numFeat,
                                                   n_redundant=0, scale=5, n_classes=numClass, class_sep=classSep)

    xTrain, xTest, yTrain, yTest = train_test_split(testDataX, testDataY,
                                                    test_size=0.05, random_state=10)
    
    model3 = WestonWatkinsSVM(numFeat,numClass)
    xTrain = xTrain.T
    xTest = xTest.T
    model3.fit(xTrain[:numFeat], yTrain.T, 100)
    totalCorrect = 0
    for i in range(len(yTest)):
        if(model3.forward(xTest[:, i]) == yTest[i]):
            totalCorrect += 1

    print("accuracy = ", (totalCorrect/len(yTest)*100), "%")        






        




