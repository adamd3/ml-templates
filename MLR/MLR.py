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
import statsmodels.formula.api as sm

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
