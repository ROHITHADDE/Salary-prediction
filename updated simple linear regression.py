import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# load dataset
dataset = pd.read_csv(r"C:\Users\adder\Desktop\Data Science\salary prediction\Salary_Data.csv")
print(dataset)

# split the dataset to independent(x) and dependent variable (y)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

# split the dataset into training and testing sets (80-20)
X_train, X_test, y_train ,y_test  = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=0)

# Train the model 
regressor = LinearRegression() # regressor is used for continuous dependent variable (y)
regressor.fit(X_train,y_train) # .fit is used to train the model

# Test the model & create a predicted table 
y_pred = regressor.predict(X_test)

# Comarision for y_test and  y_pred
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# visualization for train the data points
plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualization for test the data points 
plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 12 years of experience: ${y_20[0]:,.2f}")

# Check model performance 
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test,y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error (y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")

# save the trained model to disk
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os 
print(os.getcwd())