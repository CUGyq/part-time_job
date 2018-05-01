from sklearn.datasets import load_iris
import random
import numpy as np
from sklearn import svm
#加载数据集
iris=load_iris()
iris.keys()
#数据的条数和维数
data=iris["data"]  #数据
label= iris["target"]
random.seed(100)
np.random.shuffle(data)
np.random.shuffle(label)
print(label.shape[0])
label = np.reshape(label,(label.shape[0],1))
print(data.shape)
print(label.shape)
x_train = data[0:121, :]
y_train = label[0:121, :]
x_test = data[120:151, :]
y_test = label[120:151, :]
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())


print (clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)

print (clf.score(x_test, y_test))
y_hat = clf.predict(x_test)


print ('decision_function:\n', clf.decision_function(x_train))
print ('\npredict:\n', clf.predict(x_train))
