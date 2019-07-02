# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from warnings import simplefilter

# Helps remove any future warnings when executing the code
simplefilter(action='ignore', category=FutureWarning)

# Reading input data related to bank into array form
bankData = pd.read_csv('bank.csv')
bankInfo = [bankData]

df = pd.DataFrame(bankData)
print('Data Frame Of Bank Dataset')
print(df)

# Identifying and handling null values within the data set
nulls = pd.DataFrame(bankData.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Converting categorical features into numerical features
for bank in bankInfo:
    bank['job'] = bank['job'].map({'unknown': 0, 'unemployed': 1, 'services': 2, 'management': 3, 'blue-collar': 4, 'self-employed': 5, 'technician': 6, 'entrepreneur': 7, 'admin': 8, 'student': 9, 'housemaid': 10, 'retired': 11}).astype(int)
    bank['marital'] = bank['marital'].map({'single': 0, 'married': 1, 'divorced': 2}).astype(int)
    bank['education'] = bank['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}).astype(int)
    bank['default'] = bank['default'].map({'no': 0, 'yes': 1}).astype(int)
    bank['housing'] = bank['housing'].map({'no': 0, 'yes': 1}).astype(int)
    bank['loan'] = bank['loan'].map({'no': 0, 'yes': 1}).astype(int)
    bank['contact'] = bank['contact'].map({'unknown': 0, 'cellular': 1, 'telephone': 2}).astype(int)
    bank['month'] = bank['month'].map({'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}).astype(int)
    bank['poutcome'] = bank['poutcome'].map({'unknown': 0, 'success': 1, 'failure': 2, 'other': 3}).astype(int)
    bank['result'] = bank['result'].map({'yes': 1, 'no': 0}).astype(int)

# Displaying the data set updated to provide numerical values to non-numerical features
print('Updated Bank Dataset displaying values corresponding to Non-numerical features')
print(bank)

# Correlation of numerical features associated with target
numeric_features = bank.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['result'].sort_values(ascending=False)[:5], '\n')
print('Features negatively correlated to the target:')
print(corr['result'].sort_values(ascending=False)[-5:])

# Identifying predictors and target variables along with training and test sets
bankArray = bank.values
x = bankArray[:, :16]
y = bankArray[:, 16]

# Dropping features not correlated to the target
x = bank.drop(['contact', 'poutcome'], axis=1)
print('Predictor features are', x)
print('Target variable is', y)

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('Training predictor features')
print(x_train)
print('Test predictor features')
print(x_test)

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
modelNB = GaussianNB()
modelNB_train = modelNB.fit(x_train, y_train)
print('The model fit on the training data set using Naive Bayes approach')
print(modelNB_train, '\n')

# Predicting the results of the model on the test data
y_predictNB = modelNB.predict(x_test)

# Computing the error rate of the model fit
print('Accuracy of the Naive Bayes model fit is', metrics.accuracy_score(y_test, y_predictNB), '\n')

# Fitting Linear SVM model
svm = SVC()
modelSVM_train = svm.fit(x_train, y_train)
train_score = svm.score(x_train, y_train)
print('The model fit on the training data set using Support Vector Machines approach')
print(modelSVM_train, '\n')

# Predicting the results of the model on the test data
y_predictSVM = svm.predict(x_test)

# Computing the error rate of the model fit
accuracy_svm = round(svm.score(x_train, y_train) * 100, 2)
print('Accuracy of the linear SVM model fit is', accuracy_svm, '\n')

# Building K-Nearest Neighbours Model on training data set
knnModel = KNeighborsClassifier()
modelKNN_train = knnModel.fit(x_train, y_train)
print('The model fit on the training data set using K-Nearest Neighbours approach')
print(modelKNN_train, '\n')

# Predicting the model fit on test data set
knnPredict = knnModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
knnScore = metrics.accuracy_score(y_test, knnPredict)

print('The score obtained for K-Nearest Neighbours model is', knnScore)
