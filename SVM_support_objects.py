import pandas as pd
import sklearn.svm as svm
from sklearn import datasets


clf = svm.SVC(kernel = 'linear',C=100000, random_state=241)
print(clf.support_)