import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats

# Read data 
df1 = pd.read_csv("hemnet.csv")
plt.figure()
plt.scatter(df1["Living_area"],df1["Selling_price"])

#Create Linear Regresion
x = df1["Living_area"].values.reshape(-1, 1)
y = df1["Selling_price"].values
model = LinearRegression()
model.fit(x, y)
#plt.plot(x, model.predict(x), color='red', label="Raw Linear regression")

#Calculate the residual error of the raw data
predicted_values = model.predict(x)
residuals = y - predicted_values

# Boxplot to visualize the outlier points
"""plt.figure(figsize=(8, 6))
plt.boxplot(residuals, vert=False)
plt.title('Residuals Boxplot')
plt.xlabel('Residuals')
plt.show()"""

#Clean data
cleaned_data = df1[~df1['ID'].isin([41, 46])]

#New regression line
x_cleaned = cleaned_data['Living_area'].values.reshape(-1, 1)
y_cleaned = cleaned_data['Selling_price'].values
model_cleaned = LinearRegression()
model_cleaned.fit(x_cleaned, y_cleaned)
plt.plot(x_cleaned, model_cleaned.predict(x_cleaned), 'g', label="Cleaned Linear regression")

"""# Get the coefficients
slope = model_cleaned.coef_[0]
intercept = model_cleaned.intercept_

# QUESTION 1.1: Scatter plot to see the data 
plt.xlabel("Living area")
plt.ylabel("Selling price [$]")
plt.title("Living area VS Selling price")

# QUESTION 1.2 equation of the regression line
print(f"Regression Equation: Y = {slope:.2f} * X + {intercept:.2f}")

#QUESTION 1.3
x_predicted =[100,150,200]
x_predicted = np.array(x_predicted).reshape(-1, 1)
y_predicted = model_cleaned.predict(x_predicted)
print(y_predicted)
plt.scatter(x_predicted, y_predicted, color='yellow', marker='o', label='Points')

plt.legend()
plt.show()"""

#QUESTION 1.4
#Calculate the residual error of the cleaned data
predicted_values = model_cleaned.predict(x_cleaned)
residuals = y_cleaned - predicted_values
r_squared = r2_score(y_cleaned, predicted_values)

# Create a residual plot
"""plt.figure()
plt.scatter(predicted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Cleaned Data')
plt.show()"""

# Fit a second-degree polynomial regression model
x_cleaned = [item for sublist in x_cleaned for item in sublist]

coefficients = np.polyfit(x_cleaned, y_cleaned, 2)
polynomial = np.poly1d(coefficients)
x_curve = np.linspace(min(x_cleaned), max(x_cleaned), 100)
y_curve = polynomial(x_curve)

# Create a scatter plot of the original data
#plt.figure()
plt.scatter(x_cleaned, y_cleaned, label='Cleaned Data')

# Create a line plot for the second-degree regression curve
plt.plot(x_curve, y_curve, color='purple', label='Second-Degree Regression Curve')
plt.xlabel("Living area")
plt.ylabel("Selling price [$]")
plt.title("Living area VS Selling price")
plt.legend()
#plt.show()

#Repeat residual plot with the 2-degree regression
predicted_values = polynomial(x_cleaned)
residuals = y_cleaned - predicted_values
r_squared_second = r2_score(y_cleaned, predicted_values)
print(r_squared,r_squared_second)

# Create a residual plot
plt.figure()
plt.scatter(predicted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Cleaned Data & 2-degree regression')
plt.show()