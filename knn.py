import numpy as np 

class Knn:
    """Takes vecotr Y of labels and vector X of predictors"""

    def __init__(self, y, x, K):
        self.y = y
        self.x = x
        self.K = K
    
    def distance(self, x, y):
        return np.linalg.norm(x-y)

    def Predict(self, X):
        trainrows = np.shape(self.x)[0]
        predrows = np.shape(X)[0]
        Ypred = np.zeros(predrows)

        for i in range(predrows):
            vector = X[i,]
            distances = np.array([self.distance(vector, self.x[j,:]) for j in range(trainrows)])
            index = np.argsort(distances)[0:self.K]
            Ypred[i] = np.bincount(self.y[index]).argmax()
        
        return Ypred 







        