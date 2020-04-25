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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


##------------------------------------------------------------------------------
## Data pre-processing
##------------------------------------------------------------------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0
)
"""

# Feature Scaling: necessary because it's not automatically
# applied for the SVR class
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


##------------------------------------------------------------------------------
## Fitting SVR to the data set
##------------------------------------------------------------------------------
regressor = SVR(kernel = 'rbf') # rbf = gaussian
regressor.fit(X, y)


##------------------------------------------------------------------------------
## Visualising the results
##------------------------------------------------------------------------------
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# For higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

##------------------------------------------------------------------------------
## Predicting a new result
##------------------------------------------------------------------------------
y_pred = regressor.predict([[6.5]])
y_pred = sc_y.inverse_transform(y_pred)
