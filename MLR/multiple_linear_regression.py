# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# To avoid introducing an order to the categorical
# variables, since we have 3 categories, we must 
# create an object of the OneHotEncoder class
# and apply the fit_transform() method to fit it to the 
# categorical data
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# this has replaced the single categorical variable 
# column with 3 separate columns - 1 per variable

# Avoiding the Dummy Variable Trap
# (recall, if we have X categorical variables, then 
# we only want X-1 dummy variables to avoid introducing
# multicollinearity)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#---------------------------------------------

# Feature Scaling

# NOT REQUIRED, since it will be done by the 
# sklearn.linear_model library automatically 

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#---------------------------------------------

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# compare y_pred and y_test to check accuracy

#----------------------------------------------

# Use Backward Elimination to improve accuracy

import statsmodels.formula.api as sm

# NB: for Backward Elimination, the data must have 
# the format y = mx + c (where c is a constant)

# However, the matrix of independent variables currently
# does not contain a column with a constant..
# therefore, we need to append a column of 1s which will 
# correspond to the constant (c) that does not vary
# -- this can be done using the ones() function in numpy
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # X has 50 rows

# create a new matrix ("X_opt") that will hold our 
# "optimal" set of independent variables when 
# backward elimination is complete
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# First, select a significance level
SL = 0.05 

# Second, fit the full model to X_opt by using the 
# OLS (ordinary least squares) class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# print a summary of the full model
# (p-values are given in the P>|t| column)
regressor_OLS.summary() 

# Remove the variable with the highest p-value (x2)
# and repeat the above steps
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()