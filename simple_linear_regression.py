# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent variable (years experience)
y = dataset.iloc[:, 2].values #dependent variable (salary)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# NB: setting random state to 0 

# we have 30 observations in total, so
# we will put 10 observations in the Test set and 20 observations
# in the Trainin set (generally, 80/20 ratio is preferred, but
# for simplicity we will use 66.6/33.3 here)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #make an object of the LinearRegression class
regressor.fit(X_train, y_train) #use the fit method to fit the object to the training set

# NOTE on classes and methods:
# a given class, such as the LinearRegressor class, contains
# multiple methods; such as the "fit" method, used above.

# Predicting the results using the Training set
y_pred_train = regressor.predict(X_train)

# Predicting the results using the Test set
# NOTE: we don't plot these - it's just so manually inspect them
y_pred_test = regressor.predict(X_test)

# plot the Training set data vs our regression line
plt.scatter(X_train, y_train, color = 'red') #data points
plt.plot(X_train, y_pred_train, color = 'blue') #regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# plot the Test set data vs our regression line
# NB: the regression line is still based on the TRAINING dataset
plt.scatter(X_test, y_test, color = 'red') #data points
plt.plot(X_train, y_pred_train, color = 'blue') #regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
