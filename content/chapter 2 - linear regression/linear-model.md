### Building a Simple Linear Regression Model

In this section, we will walk through the process of building a simple linear regression model using the `advertising.csv` dataset. This dataset contains information about the amount of money spent on different types of advertising (TV, radio, and newspaper) and the corresponding sales of a product. We will use this data to predict sales based on TV advertising spend, demonstrating the steps involved in creating and evaluating a linear regression model.

#### 1. **Importing Libraries and Loading Data**
First, let’s import the necessary libraries and load the `advertising.csv` dataset into a pandas DataFrame.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('data/advertising.csv')

# Display the first few rows of the dataset
data.head()
```

#### 2. **Exploring the Dataset**
The `advertising.csv` dataset contains the following columns:
- **TV**: Amount spent on TV advertising (in thousands of dollars)
- **Radio**: Amount spent on radio advertising (in thousands of dollars)
- **Newspaper**: Amount spent on newspaper advertising (in thousands of dollars)
- **Sales**: Sales of the product (in thousands of units)

Let’s begin by exploring the dataset visually to understand the relationship between the variables.

```python
# Visualizing the relationship between TV advertising and sales
plt.figure(figsize=(8, 5))
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title('TV Advertising vs Sales')
plt.xlabel('TV Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.show()
```

This scatter plot will give us a sense of the relationship between TV advertising spend and sales.

#### 3. **Preparing the Data**
For this example, we will focus on predicting **Sales** based on **TV** advertising spend. This means that:
- The **independent variable (X)** is the **TV** column.
- The **dependent variable (Y)** is the **Sales** column.

We need to split the data into training and testing sets to evaluate the model’s performance.

```python
# Define the independent (X) and dependent (Y) variables
X = data[['TV']]  # Independent variable (TV spend)
Y = data['Sales']  # Dependent variable (Sales)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

#### 4. **Building the Linear Regression Model**
Now, let’s create a simple linear regression model using the training data and fit it to the data.

```python
# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)
```

#### 5. **Making Predictions**
Once the model is trained, we can use it to predict sales based on TV advertising spend in the test set.

```python
# Make predictions using the test set
Y_pred = model.predict(X_test)

# Plot the regression line along with the data points
plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, color='blue', label='Actual Sales')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: TV Advertising vs Sales')
plt.xlabel('TV Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.legend()
plt.show()
```

This plot will show the actual sales as blue points and the predicted sales (based on the regression model) as a red line.

#### 6. **Evaluating the Model**
To assess how well the model performs, we can calculate performance metrics such as the **Mean Squared Error (MSE)** and **R-squared (R²)**.

```python
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate R-squared (R²)
r2 = r2_score(Y_test, Y_pred)
print(f'R-squared (R²): {r2}')
```

- **Mean Squared Error (MSE)**: This measures the average of the squared differences between actual and predicted values. Lower values of MSE indicate a better fit.
- **R-squared (R²)**: This measures how well the independent variable explains the variability of the dependent variable. An R² value closer to 1 indicates a better fit.

#### 7. **Interpreting the Results**
- **Regression Coefficient**: The `model.coef_` attribute gives the slope of the regression line, which represents how much **sales** change for every unit increase in **TV advertising spend**.
  
```python
# Get the model coefficient (slope) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f'Regression Coefficient (Slope): {slope}')
print(f'Intercept: {intercept}')
```

- **Equation of the Line**: The equation of the regression line is:
  \[
  \text{Sales} = (\text{Slope}) \times (\text{TV Advertising Spend}) + \text{Intercept}
  \]
  For example, if the slope is 0.045 and the intercept is 5, the equation would be:
  \[
  \text{Sales} = 0.045 \times (\text{TV Advertising Spend}) + 5
  \]

#### 8. **Conclusion**
In this example, we built a simple linear regression model to predict **Sales** based on **TV** advertising spend. By using the model, we learned how to:
- Load and explore a dataset
- Split data into training and testing sets
- Train a linear regression model
- Make predictions and evaluate the model’s performance

While this model is simple, it demonstrates the foundational steps involved in building a linear regression model. You can extend this to more complex models by adding additional independent variables, such as **radio** and **newspaper** advertising, to improve prediction accuracy.
