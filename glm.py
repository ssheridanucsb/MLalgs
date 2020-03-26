import numpy as np
import math

class LinReg:
    #Classic Linear Regression 
    #Y must be nx1 vector of the response
    #x must be the nxp-1 vector of the predictors 

    def __init__(self):
      pass

    def designMatrix(self, X):
        """creates a matrix of the predictors with the first column having all 1's"""
        shape = np.shape(X)
        design = np.ones((shape[0], shape[1]+1))
        design[:,1:] = X
        return design
    
    def Train(self, Y, x):
        """calculates the predictors through matrix multiplication"""
        self.Y = Y
        self.x = x
        self.dm = self.designMatrix(x)
        #from theory we know bethat = ((xTx)^-1)xTY where x is the design matrix
        self.betaHat = np.linalg.inv(self.dm.transpose() @ self.dm) @ self.dm.transpose() @ Y
        self.Y_hat = self.dm @ self.betaHat

    def Predict(self, x):
        """predictes Y values by using the inputed x values and coefficients"""
        return  self.designMatrix(x) @ self.betaHat 


class LogReg:
    #Logistic Regression

    def designMatrix(self, X):
        shape = np.shape(X)
        design = np.ones((shape[0], shape[1]+1))
        design[:,1:] = X
        return design
    
    def inverseLogit(self, eta):
        """inverse logit function that takes numpy array of the predictions"""
        exp_eta = np.array(list(map(math.exp, eta)))
        return exp_eta/(exp_eta + 1)

    
    def Train(self, Y, x):
        self.Y = Y
        self.x = x
        self.dm = self.designMatrix(x)
        self.betaHat = np.linalg.inv(self.dm.transpose() @ self.dm) @ self.dm.transpose() @ Y
        etaHat = self.dm @ self.betaHat
        self.logOdds = etaHat
        self.probs = self.inverseLogit(etaHat)
    
    def Predict(self, x, t=1):
        X = self.designMatrix(x)
        etaHat = X @ self.betaHat
        if(t==0):
            return etaHat
        else:
            return self.inverseLogit(etaHat)
        

    













