# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Reading Boston housing input data
trainBoston = pd.read_csv('trainBoston.csv')
testBoston = pd.read_csv('testBoston.csv')
boston = [trainBoston, testBoston]
trainBoston.medv.describe()

# Performing EDA
print(trainBoston['medv'].value_counts(dropna='False'))

# Evaluating the skewness of quality
print('Skewness of Median Valuation of Owner-occupied Homes', trainBoston.medv.skew())
plt.hist(trainBoston.medv, color='navy')
plt.title('Skewness associated with Median Valuation of Owner-occupied Homes')
plt.show()

# Normalize the skewed distribution of Sale Price
target = np.log(trainBoston.medv)
print('Normalized Distribution of Median Valuation of Owner-occupied Homes', target.skew())
plt.hist(target, color='navy')
plt.title('Normalized Distribution of Median Valuation of Owner-occupied Homes')
plt.show()

# Correlation of numerical features associated with target
numeric_features = trainBoston.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[:5],'\n')
print('Features negatively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[-5:])

# Pivotable overall median valuation of homes
quality_pivot = trainBoston.pivot_table(index='rm', values='medv', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.title('Median Valuation of Owner-occupied Homes')
plt.show()

# Handling null values within the data set
nulls = pd.DataFrame(trainBoston.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print('Features of Boston Housing with associated Null values')
print(nulls)

# Identifying features and predictor and split the data set into training and test set
y = np.log(trainBoston.medv)
x = trainBoston.drop(['medv'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

# Building regression model
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
print('The model fit on the data set is', model)

# Evalution of the model fit using RMSE and R2 score
r2_error = model.score(x_test, y_test)
print('R^2 error of the model fit is ', r2_error)
prediction = model.predict(x_test)
rsme = mean_squared_error(y_test, prediction)
print('RMSE of the model fit is ', rsme)

# Graphs of evaluation results
actual_result = y_test
plt.scatter(prediction, actual_result, alpha=.75, color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted Median Valuation')
plt.ylabel('Actual Median Valuation')
plt.title('Multiple Regression of Mean Valuation of Owner-occupied Homes')
plt.show()
