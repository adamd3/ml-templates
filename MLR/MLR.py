#!/usr/bin/env python

##------------------------------------------------------------------------------
## Import libraries
##------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


##------------------------------------------------------------------------------
## Functions
##------------------------------------------------------------------------------
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


##------------------------------------------------------------------------------
## Import the dataset
##------------------------------------------------------------------------------
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

##------------------------------------------------------------------------------
## Pre-processing
##------------------------------------------------------------------------------
# Encode categorical data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
    )
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0
)


# Feature Scaling
# NOT REQUIRED, since it will be done by the
# sklearn.linear_model library automatically
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


##------------------------------------------------------------------------------
## Fit MLR to the Training set
##------------------------------------------------------------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# compare y_pred and y_test to check accuracy
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)
    ), 1)
    )


##------------------------------------------------------------------------------
## Backward Elimination for model improvement
##------------------------------------------------------------------------------
SL = 0.05
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype = float) # column indices for vars
X_Modeled = backwardElimination(X_opt, SL) #remove based on SL
