import sklearn
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree


iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

test_idx = [0,50,100]

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)


#Decision tree classifier
clf = tree.DecisionTreeClassifier()
clf =clf.fit(train_data,train_target)

print(clf.predict(test_data))
print(test_target)

