from glm import LinReg
from sklearn import datasets 
import numpy as np

X, y = datasets.load_diabetes(return_X_y=True)
design = np.ones((442,11))
design[:,1:] = X 

lin = LinReg()
lin.Train(y, X)

