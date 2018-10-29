import sklearn
import numpy as np
from sklearn import tree
import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        predictions =[]
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, test_row):
        best_train = euc(test_row,self.X_train[0])
        best_index =0
        for i in range(1,len(self.X_train)):
            cur_train =euc(test_row,self.X_train[i])
            if cur_train<best_train:
                best_train = cur_train
                best_index = i
        return self.y_train[best_index]



from sklearn.datasets import load_iris
iris = load_iris()
y= iris.target
x = iris.data

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train,y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
