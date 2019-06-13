# Importing libraries
import pandas as pd

# Loading and reading the data set made avaialable
titanic_train = pd.read_csv('train_preprocessed.csv')
titanic_test = pd.read_csv('test_preprocessed.csv')
titanic = [titanic_train, titanic_test]

print(titanic_train['Survived'].value_counts(dropna='False'))

# Analyzing the data
print('Training Data Set:')
print(titanic_train.info())
print('Test Data Set:')
print(titanic_test.info())

# Fixing null values
print(titanic_train.isnull().sum())
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

# Analyze pivoting features
result = titanic_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print('The correlation of survival with gender is :')
print(result)

