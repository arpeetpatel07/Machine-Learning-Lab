# -*- coding: utf-8 -*-

LOGISTIC REGRESSION
"""

import numpy as np
 import pandas as pd

df=pd.read_csv('Social_Network_Ads.csv')
 df.head()

df.shape

df.info()

X=df[['Age','EstimatedSalary']]
 y=df['Purchased']#data is separated x-independent y-dependent

from sklearn.model_selection import train_test_split
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

X_train.shape

X_train.isna().sum()

from sklearn.impute import SimpleImputer
 imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

imputer.fit(X_train[['Age',"EstimatedSalary"]]) #.fit method- finds the mean

X_train[['Age','EstimatedSalary']]=imputer.transform(X_train[['Age','EstimatedSalary']])

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
scalar.fit(X_train)
X_train_scaled=scalar.transform(X_train)
X_test=scalar.transform(X_test)

X_train_scaled

from sklearn.linear_model import LogisticRegression
 model=LogisticRegression()
 model.fit(X_train_scaled,y_train)

model.score(X_train_scaled,y_train)

model.score(X_test,y_test)
