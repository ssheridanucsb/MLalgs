import numpy as np

class LinReg:
    #Y must be nx1 vector
    #x must be the nxp designmatrix with all 1's in the first column 
    def __init__(self):
      pass
        
    
    def Train(self, Y, x):
        self.Y = Y
        self.x = x
        self.betaHat = np.linalg.inv(x.transpose() @ x) @ x.transpose() @ Y
        self.Y_hat = x @ self.betaHat

    def Predict(self, x):
        return self.betaHat @ x

    def mse(self):
        mse = np.mean(np.square(self.Y_hat - self.Y))
        return mse








