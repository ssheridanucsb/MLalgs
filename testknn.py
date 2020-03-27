from knn import Knn
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = Knn(y_train, X_train, 5)

knnreal = KNeighborsClassifier(n_neighbors=5)
knnreal.fit(X_train, y_train)

print(accuracy_score(y_test, knnreal.predict(X_test)))
print(accuracy_score(y_test, knn.Predict(X_test)))
