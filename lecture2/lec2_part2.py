import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import datasets

diabetes = datasets.load_diabetes(as_frame=True)


# Load the diabetes dataset
diabetes_X, diabetes_y = diabetes.data, diabetes.target

# Print part of the dataset
print(diabetes_X.head())



# Load the diabetes dataset
#diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)

# Use only the BMI feature
#diabetes_X = diabetes_X.loc[:, ['bmi']]

# The BMI is zero-centered and normalized; we recenter it for ease of presentation
#diabetes_X = diabetes_X * 30 + 25

# Collect 20 data points
#diabetes_X_train = diabetes_X.iloc[-20:]
#diabetes_y_train = diabetes_y.iloc[-20:]

# Create linear regression object
#regr = linear_model.LinearRegression()

# Train the model using the training sets
#regr.fit(diabetes_X_train, diabetes_y_train.values)

# Make predictions on the training set
#diabetes_y_train_pred = regr.predict(diabetes_X_train)

# The coefficients
#print('Slope (theta1): \t', regr.coef_[0])
#print('Intercept (theta0): \t', regr.intercept_)