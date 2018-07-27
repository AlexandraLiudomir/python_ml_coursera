import numpy as np
import pandas as pd
import sklearn.linear_model as lin
from sklearn.preprocessing import StandardScaler as scaler
import sklearn.metrics as metrics

trainData = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv',names=['0','1','2'],usecols=['1','2'])
trainAnsw = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv',names=['0','1','2'],usecols=['0'],squeeze=True)

testData = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv',names=['0','1','2'],usecols=['1','2'])
testAnsw = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv',names=['0','1','2'],usecols=['0'],squeeze=True)

print(trainData.shape,trainAnsw.shape,testData.shape,testAnsw.shape)
clf = lin.Perceptron(random_state=241)
clf.fit(trainData,trainAnsw)
#accuracy = clf.score(testData,testAnsw)
pred = clf.predict(testData)

acc=metrics.accuracy_score(testAnsw,pred)
sc = scaler()
train_sc = sc.fit_transform(trainData,trainAnsw)
test_sc = sc.transform(testData)
clf.fit(train_sc,trainAnsw)
#accuracy2 = clf.score(testData,testAnsw)
pred = clf.predict(test_sc)
acc2=metrics.accuracy_score(testAnsw,pred)
print(acc2 - acc)
