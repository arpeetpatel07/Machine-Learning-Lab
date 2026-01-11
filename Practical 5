# -*- coding: utf-8 -*-


import numpy as np
 from sklearn.linear_model import LinearRegression
 import matplotlib.pyplot as plt
 from sklearn.metrics import mean_absolute_error, mean_squared_error

"""1.Prepare the dataset"""

import numpy as np
 #independent variable (features) - Years of experience
 x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
 #dependent variable (target) - Salary
 y = np.array([30000, 35000, 45000, 50000, 60000, 65000, 75000, 80000, 90000, 95000])

model = LinearRegression()
 model.fit(x,y)

y_pred = model.predict(x)

mae = mean_absolute_error(y,y_pred)
 mse = mean_squared_error(y,y_pred)
 rmse = np.sqrt(mse)

print("Mean Absoulte Error (MAE):",mae)
 print("Mean Squared Error (MSE):",mse)
 print("Root Mean Squared Error (RMSE):",rmse)

new_experience = np.array([[11]])
 predicted_salary = model.predict(new_experience)

print(f"Coefficients (slope): {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salary for 11 years of experience : ${predicted_salary[0]:.2f}")

#5. Visualize the results
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(new_experience, predicted_salary, color='green', marker = 'X', s=200)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression: Salary vs. Experience')
plt.legend()
plt.grid(True)
plt.show()

"""Multiple Linear Regression"""

import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_absolute_error, mean_squared_error

from google.colab import files
uploaded = files.upload()

import pandas as pd
 df = pd.read_csv('ICC_Test_Batting_Rankings.csv')
 df

model = LinearRegression()
 model.fit(df[['rating']], df['rank'])

y_pred = model.predict(df[['rating']])

mae = mean_absolute_error(df['rank'],y_pred)
 mse = mean_squared_error(df['rank'],y_pred)
 rmse = np.sqrt(mse)

print("Mean Absoulte Error (MAE):",mae)
print("Mean Squared Error (MSE):",mse)
print("Root Mean Squared Error (RMSE):",rmse)
