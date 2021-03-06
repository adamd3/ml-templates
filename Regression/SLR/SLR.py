#!/usr/bin/env python

##------------------------------------------------------------------------------
## Import libraries
##------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

##------------------------------------------------------------------------------
## Import the dataset
##------------------------------------------------------------------------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent variable (years experience)
y = dataset.iloc[:, 2].values #dependent variable (salary)

##------------------------------------------------------------------------------
## Pre-processing
##------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
# (for simplicity, put 10 in the Test set and 20 in the Training set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 1/3, random_state = 0
    )

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

##------------------------------------------------------------------------------
## Fit SLR to the Training set
##------------------------------------------------------------------------------
regressor = LinearRegression() # object of the LinearRegression class
regressor.fit(X_train, y_train) #fit method to fit object to the training set

# Predicting the results using the Training set
y_pred_train = regressor.predict(X_train)

# Predicting the results using the Test set
# NOTE: we don't plot these - just manually inspect them
y_pred_test = regressor.predict(X_test)

##------------------------------------------------------------------------------
## Plot the Training set data vs regression line
##------------------------------------------------------------------------------
plt.scatter(X_train, y_train, color = 'red') #data points
plt.plot(X_train, y_pred_train, color = 'blue') #regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

##------------------------------------------------------------------------------
## Plot the Test set data vs regression line
## (NB: regression line is still based on the Training dataset)
##------------------------------------------------------------------------------
plt.scatter(X_test, y_test, color = 'red') #data points
plt.plot(X_train, y_pred_train, color = 'blue') #regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
