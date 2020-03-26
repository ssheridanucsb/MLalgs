from glm import LinReg, LogReg
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

Xd, yd = datasets.load_diabetes(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.33)
lin = LinReg()
lin.Train(yd_train, Xd_train)
print(r2_score(lin.Y, lin.Y_hat))
print(r2_score(yd_test, lin.Predict(Xd_test)))
print(mean_squared_error(lin.Y, lin.Y_hat))
print(mean_squared_error(yd_test, lin.Predict(Xd_test)))


Xb, yb = datasets.load_breast_cancer(return_X_y=True)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.33)

log = LogReg()
log.Train(yb_train, Xb_train)
Yb_pred = log.Predict(Xb_test, t=1)
Yb_pred = np.where(Yb_pred > 0.5, 1, 0)
print(accuracy_score(Yb_pred, yb_test))