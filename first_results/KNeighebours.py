import numpy as np
import pandas as pd
import array
#from sklearn.model_selection import KFold
import sklearn.model_selection as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv('week2_wine.data', names=['0','1','2','3','4','5','6','7','8','9','10','11','12','13'], usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13])
types = pd.read_csv('week2_wine.data', names=['0','1','2','3','4','5','6','7','8','9','10','11','12','13'], usecols=[0],squeeze=True)
fold = sk.KFold(n_splits = 5, random_state=42, shuffle = True)
print (types.shape)
print(data.shape)
score = []
for i in range(1,50):
    clf = KNeighborsClassifier(n_neighbors=i)
    score.append(np.mean(sk.cross_val_score(estimator=clf,X=data,y=types,cv=fold)))

print(max(score))
n = np.argmax(score,axis=0)
print(n)

score = []
scaledData = preprocessing.scale(data)
for i in range(1,50):
    clf = KNeighborsClassifier(n_neighbors=i)
    score.append(np.mean(sk.cross_val_score(estimator=clf,X=scaledData,y=types,cv=fold)))

print(max(score))
print(score.argmax(axis=0))


