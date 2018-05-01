

from numpy import *
import SVM
from sklearn.datasets.samples_generator import make_circles

################## test svm #####################
## step 1: load data
print ("step 1: load data...")
dataSet = []
labels = []
X,labels=make_circles(n_samples=1000,noise=0.2,factor=0.2,random_state=1)
for i in range(len(labels)):
    if labels[i] == 0:
        labels[i] = -1
dataSet = mat(X)
labels = mat(labels).T
train_x = dataSet[0:801, :]
train_y = labels[0:801, :]
test_x = dataSet[800:1001, :]
test_y = labels[800:1001, :]

## step 2: training...
print ("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 10000
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.3))

## step 3: testing
print ("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)
## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)

