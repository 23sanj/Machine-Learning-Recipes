import sklearn
from sklearn import tree

#features are : weight and the texture
features =[[140,1],[130,1],[150,0],[170,0]]
labels =[0,0,1,1]

#the classifier object is a box of rules
clf = tree.DecisionTreeClassifier()
#The learning algorithm that finds patterns in the data to create rules
clf =clf.fit(features,labels)

print(clf.predict([[130,1]]))