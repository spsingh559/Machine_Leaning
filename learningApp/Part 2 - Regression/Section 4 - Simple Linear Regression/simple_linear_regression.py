#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:59:19 2017

@author: root
"""

# Linear Regression
#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading CSV
dataset=pd.read_csv('Salary_Data.csv');

#preparing axis
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#Splitting Dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Regression algorithm
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred=regressor.predict(X_test)

#Visulising the Training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Expericance (Training Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

#Visulising the Test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Expericance (Test Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()