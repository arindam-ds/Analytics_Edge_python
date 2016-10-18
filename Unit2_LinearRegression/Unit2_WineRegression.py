import numpy as np
import pandas as pd 
import scipy
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

# Sum of Squared Errors
print(model1.score(X,y))

# Linear Regression (two variables)
feature_cols = ['AGST', 'HarvestRain']
X = wine[feature_cols]
y = wine.Price
model2 = LinearRegression()
model2.fit(X, y)
print model2.intercept_
print model2.coef_

# Sum of Squared Errors
print(model2.score(X,y))

# Linear Regression (all variables)
feature_cols = ['AGST', 'HarvestRain', 'WinterRain', 'Age', 'FrancePop']
X = wine[feature_cols]
y = wine.Price
model3 = LinearRegression()
model3.fit(X, y)
print model3.intercept_
print model3.coef_

# Sum of Squared Errors
print(model3.score(X,y))


#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X,y)