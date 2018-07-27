import pandas as pd
import sklearn.svm as svm
import sklearn.model_selection as ms
from sklearn import datasets
import sklearn.model_selection as sk
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer

newsgroups = datasets.fetch_20newsgroups(subset='all',categories=['alt.atheism','sci.space'])
vect = vectorizer()
tfIdf = vect.fit_transform(newsgroups.data,newsgroups.target)
feature_mapping = vect.get_feature_names()
fold = sk.KFold(n_splits = 5, random_state=241, shuffle = True)
c_param = [10.e-5,10.e-4,10.e-3,10.e-2,10.e-1,10.e1,10.e2,10.e3,10.e4,10.e5]
c_target = 0
maxscore = 0
for i in c_param:
    clf = svm.SVC(kernel = 'linear',C=i, random_state=241)
    #clf = KNeighborsClassifier(n_neighbors=i)
    score=np.mean(sk.cross_val_score(estimator=clf,X=tfIdf,y=newsgroups.target,cv=fold))
    if (score>maxscore):
        c_target = i
        maxscore = score

clf = svm.SVC(kernel = 'linear',C=c_target, random_state=241)
clf.fit(X=tfIdf,y=newsgroups.target)

ind = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
for i in ind:
    print(feature_mapping[i])
