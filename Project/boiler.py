
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict([[3, 4, 7],
       [5, 8, 5],
       [6, 7, 7],
       [9, 5, 6],
       [4, 3, 7],
       [5, 4, 5],
       [2, 1, 3],
       [8, 5, 5],
       [3, 7, 7],
       [4, 3, 7],
       [2, 1, 3],
       [1, 0, 1]])