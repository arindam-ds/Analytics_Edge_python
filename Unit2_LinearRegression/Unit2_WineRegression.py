import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

# VIDEO 4

# Read in data
wine = pd.read_csv("wine.csv")
wine.head()
wine.dtypes
wine.describe()
wine.shape

# Linear Regression (one variable)
feature_cols = ['AGST']
X = wine[feature_cols]
y = wine.Price
model1 = LinearRegression()
model1.fit(X, y)
print model1.intercept_
print model1.coef_
print(model1.score(X,y)) # Multiple R-squared error