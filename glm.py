import numpy as np

class LinReg:
    #Y must be nx1 vector of the response
    #x must be the nxp-1 vector of the predictors 

    def __init__(self):
      pass

    def designMatrix(self, X):
        shape = np.shape(X)
        design = np.ones((shape[0], shape[1]+1))
        design[:,1:] = X
        return design
    
    def Train(self, Y, x):
        self.Y = Y
        self.x = x
        self.dm = self.designMatrix(x)
        self.betaHat = np.linalg.inv(self.dm.transpose() @ self.dm) @ self.dm.transpose() @ Y
        self.Y_hat = self.dm @ self.betaHat

    def Predict(self, x):
        return self.betaHat @ self.designMatrix(x)










