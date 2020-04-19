#!/usr/bin/env python

##------------------------------------------------------------------------------
## Import libraries
##------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



##------------------------------------------------------------------------------
## Import data
##------------------------------------------------------------------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



##------------------------------------------------------------------------------
## Fitting RFR to the Training set
##------------------------------------------------------------------------------
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)



##------------------------------------------------------------------------------
## Visualising the results
##------------------------------------------------------------------------------
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


##------------------------------------------------------------------------------
## Predicting a new result
##------------------------------------------------------------------------------
y_pred = regressor.predict([[6.5]])
