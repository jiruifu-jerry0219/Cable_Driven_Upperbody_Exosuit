# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:31:55 2022

@author: Users-NB
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

cancer = datasets.load_breast_cancer()
X = cancer.data
Y = cancer.target


att_tr,att_ts,class_tr,class_ts = train_test_split(X,Y,test_size = 0.20)



clf = SVC(kernel="linear")
clf.fit = (att_tr,class_tr)


prediction = clf.predict(att_ts)
print("Accuracy = ",accuracy_score(class_ts,prediction))