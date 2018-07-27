import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection as sk
from sklearn.neighbors import KNeighborsRegressor

boston = sklearn.datasets.load_boston()
data = sklearn.preprocessing.scale(boston.data)
p = np.linspace(start=1,stop=10,num=200)
fold = sk.KFold(n_splits = 5, random_state=42, shuffle = True)

score = []

for i in p:
    clf = KNeighborsRegressor(weights='distance')
    score.append(np.mean(sk.cross_val_score(estimator=clf,X=data,y=boston.target,cv=fold, scoring='neg_mean_squared_error')))

print(np.max(score), p[np.argmax(score)])

