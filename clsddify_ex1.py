# -*- coding:utf-8 -*-  
from sklearn import datasets, neighbors, linear_model
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data 
y_digits = digits.target
n = int(X_digits.shape[0]*0.1) 

X_digits_train = X_digits[:n]
y_digits_train = y_digits[:n]
X_digits_test = X_digits[n:]
y_digits_test = y_digits[n:]

from sklearn.neighbors import KNeighborsClassifier
for n_neig in xrange(1,15):
	knn = KNeighborsClassifier(n_neighbors = n_neig)
	knn.fit(X_digits_train,y_digits_train)

	pred = knn.predict(X_digits_test)
	np.set_printoptions(threshold=np.nan)
	print np.mean(pred != y_digits_test)

########################################################
########################################################
print ' '
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_digits_train,y_digits_train)
pred2 = logistic.predict(X_digits_test)
print np.mean(pred2 != y_digits_test)