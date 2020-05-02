#!/usr/bin/env python

##------------------------------------------------------------------------------
## Import libraries
##------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

##------------------------------------------------------------------------------
## Import data, split, transform
##------------------------------------------------------------------------------
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

## Encode categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [1])], remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

## Split into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0
)

## Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##------------------------------------------------------------------------------
## Initialise ANN
##------------------------------------------------------------------------------
classifier = Sequential()

## Add input layer and first hidden layer
classifier.add(
    Dense(
        units = 6, kernel_initializer = 'uniform',
        activation = 'relu', input_dim = 11
        )
)

## Add second hidden layer
classifier.add(
    Dense(
        units = 6, kernel_initializer = 'uniform', activation = 'relu'
        )
)

## Add output layer
classifier.add(
    Dense(
        units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'
        )
)

## Compile ANN
classifier.compile(
    optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']
)


##------------------------------------------------------------------------------
## Fit ANN to Training set
##------------------------------------------------------------------------------
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


##------------------------------------------------------------------------------
## Make predictions and evaluate the model
##------------------------------------------------------------------------------
## Test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

## Check accuracy with confusion Matrix
cm = confusion_matrix(y_test, y_pred)
