# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:50:37 2018

@author: 1628083
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("banking.csv")
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
x[:,1] = label_encoder_x.fit_transform(x[:,1])
x[:,2] = label_encoder_x.fit_transform(x[:,2])
x[:,3] = label_encoder_x.fit_transform(x[:,3])
x[:,4] = label_encoder_x.fit_transform(x[:,4])
x[:,5] = label_encoder_x.fit_transform(x[:,5])
x[:,6] = label_encoder_x.fit_transform(x[:,6])
x[:,7] = label_encoder_x.fit_transform(x[:,7])
x[:,8] = label_encoder_x.fit_transform(x[:,8])
x[:,9] = label_encoder_x.fit_transform(x[:,9])
x[:,14] = label_encoder_x.fit_transform(x[:,14])

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features=[1])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[12])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[15])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[22])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[24])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[26])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]
'''
onehot = OneHotEncoder(categorical_features=[28])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]
'''
onehot = OneHotEncoder(categorical_features=[29])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[38])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]

onehot = OneHotEncoder(categorical_features=[46])
x = onehot.fit_transform(x).toarray()
x = x[:,1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#1. KNN
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
classifier1.fit(x_train, y_train)
y_pred1 = classifier1.predict(x_test)
y_prob1 = classifier1.predict_proba(x_test)[:,1]

#2. SVM
from sklearn.svm import SVC
classifier2 = SVC(kernel='linear', probability=True, random_state=0)
classifier2.fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
y_prob2 = classifier2.predict_proba(x_test)[:,1]

#3. Decision-Tree
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier3.fit(x_train, y_train)
y_pred3 = classifier3.predict(x_test)
y_prob3 = classifier3.predict_proba(x_test)[:,1]

#4. KERNEL
from sklearn.svm import SVC
classifier4 = SVC(kernel='rbf', random_state=0, probability=True)
classifier4.fit(x_train, y_train)
y_pred4 = classifier4.predict(x_test)
y_prob4 = classifier4.predict_proba(x_test)[:,1]

#5. Random-Forest
from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 10, criterion='gini', random_state=0)
classifier5.fit(x_train, y_train)
y_pred5 = classifier5.predict(x_test)
y_prob5 = classifier5.predict_proba(x_test)[:,1]


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Confusion- Matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)
cm5 = confusion_matrix(y_test, y_pred5)

#Accuracy-Score
print(accuracy_score(y_test, y_pred1))
print(accuracy_score(y_test, y_pred2))
print(accuracy_score(y_test, y_pred3))
print(accuracy_score(y_test, y_pred4))
print(accuracy_score(y_test, y_pred5))

#Classification-Report
print(classification_report(y_test, y_pred1))
print(classification_report(y_test, y_pred2))
print(classification_report(y_test, y_pred3))
print(classification_report(y_test, y_pred4))
print(classification_report(y_test, y_pred5))

newdf = pd.read_csv("banking_batch.csv")
xnew = newdf.iloc[:,:].values
label_enc_newx = LabelEncoder()
xnew[:,1] = label_enc_newx.fit_transform(xnew[:,1])
xnew[:,2] = label_enc_newx.fit_transform(xnew[:,2])
xnew[:,3] = label_enc_newx.fit_transform(xnew[:,3])
xnew[:,4] = label_enc_newx.fit_transform(xnew[:,4])
xnew[:,5] = label_enc_newx.fit_transform(xnew[:,5])
xnew[:,6] = label_enc_newx.fit_transform(xnew[:,6])
xnew[:,7] = label_enc_newx.fit_transform(xnew[:,7])
xnew[:,8] = label_enc_newx.fit_transform(xnew[:,8])
xnew[:,9] = label_enc_newx.fit_transform(xnew[:,9])
xnew[:,14] = label_enc_newx.fit_transform(xnew[:,14])

onehotnew = OneHotEncoder(categorical_features = [1])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [12])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [15])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [22])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [24])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [26])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [28])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [29])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [38])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

onehotnew = OneHotEncoder(categorical_features = [46])
xnew = onehotnew.fit_transform(xnew).toarray()
xnew = xnew[:,1:]

xnew = sc.fit_transform(xnew)
xnew = sc.transform(xnew)

ynew_pred = classifier5.predict(xnew)
ynew_prob = classifier5.predict_proba(xnew)[:,1]

#converting ynew_pred to data frame
dfnew_prob = pd.DataFrame({'Probability':ynew_prob})

#conacatinating ynew_pred with newdf
final = pd.concat([dfnew_prob, newdf], axis=1)

result = final.sort_values(by = ['Probability'], ascending=False)
answer = result.head(50)
