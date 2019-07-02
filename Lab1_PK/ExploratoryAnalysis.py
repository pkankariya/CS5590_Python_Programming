# Importing libraries
import pandas as pd

# Reading Boston housing input data
trainBoston = pd.read_csv('trainBoston.csv')
testBoston = pd.read_csv('testBoston.csv')
boston = [trainBoston, testBoston]
print(trainBoston['medv'].value_counts(dropna='False'))

# Analyzing the data
print('Training Data Set:')
print(trainBoston.info())
print('Test Data Set:')
print(testBoston.info())

# Fixing null values
print(trainBoston.isnull().sum())
trainBoston["lstat"] = trainBoston["lstat"].fillna("S")

# Analyze pivoting features
result = trainBoston[['tax', 'medv']].groupby(['tax'], as_index=False).mean().sort_values(by='medv', ascending=False)
print('The correlation of median valuation of owner-occupied homes with tax is :')
print(result)
