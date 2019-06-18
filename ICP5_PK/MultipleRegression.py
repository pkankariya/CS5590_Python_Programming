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

# Reading wine input data
wineData = pd.read_csv('winequality-red.csv')
wineData.quality.describe()

# Evaluating the skewness of quality
print('Skewness associated with Quality ', wineData.quality.skew())
plt.hist(wineData.quality, color='navy')
plt.title('Skewness associated with Quality')
plt.show()

# Normalize the skewed distribution of Sale Price
target = np.log(wineData.quality)
print('Normalized Distribution of Quality', target.skew())
plt.hist(target, color='navy')
plt.title('Normalized Distribution of Quality')
plt.show()

# Correlation of numerical features associated with target
numeric_features = wineData.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['quality'].sort_values(ascending=False)[:5],'\n')
print('Features negatively correlated to the target:')
print(corr['quality'].sort_values(ascending=False)[-5:])

# Pivotable overall quality
quality_pivot = wineData.pivot_table(index='alcohol', values='quality', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.title('Quality Pivot of Alcohol')
plt.show()

# Handling null values within the data set
nulls = pd.DataFrame(wineData.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Identifying features and predictor and split the data set into training and test set
y = np.log(wineData.quality)
#x = wineData[:, 0:12]
x = wineData.drop(['quality'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

# Building regression model
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
print(model)

# Evalution of the model fit using RMSE and R2 score
r2_error = model.score(x_test, y_test)
print('R^2 error of the model fit is ', r2_error)
prediction = model.predict(x_test)
rsme = mean_squared_error(y_test, prediction)
print('RMSE of the model fit is ', rsme)

# Graphs of evaluation results
actual_result = y_test
plt.scatter(prediction, actual_result, alpha=.75, color='b')  #alpha helps to show overlapping data
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Multiple Regression of Wine Quality Evaluation')
plt.show()