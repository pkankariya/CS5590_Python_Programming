# Importing libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Reading input data pertaining to Boston Housing
trainBoston = pd.read_csv('trainBoston.csv')
testBoston = pd.read_csv('testBoston.csv')
bostonData = [trainBoston, testBoston]

print(trainBoston['medv'].value_counts(dropna='False'))

# Analyzing the data
print('Training Data Set:')
print(trainBoston.info())
print('Test Data Set:')
print(testBoston.info())

# Identifying and handling null values within the data set
nulls = pd.DataFrame(trainBoston.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Correlation of numerical features associated with target
numeric_features = trainBoston.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[:5],'\n')
print('Features negatively correlated to the target:')
print(corr['medv'].sort_values(ascending=False)[-5:])

# Median valuation of owner-occupied homes evaluation identifying the pivotal feature with > 0.5 significance
medvPivot = trainBoston.pivot_table(index='rm', values='medv', aggfunc=np.median)
medvPivot.plot(kind='bar', color='blue')
plt.title('Median Valuation of Owner-occupied Homes Pivotal Feature')
plt.xlabel('rm')
plt.ylabel('medv')
plt.show()

# Defining the training and test data sets along with ground truth and predicted labels

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
modelNB = GaussianNB()
modelNB_train = modelNB.fit(x_train, y_train)
print('The model fit on the training data set:', modelNB_train)

# Predicting the results of the model on the test data
# y_predictNB = modelNB.predict(x_test)

# Computing the error rate of the model fit
# print('Accuracy of the Naive Bayes model fit:', metrics.accuracy_score(y_test, y_predictNB))

# Fitting Linear SVM model
# svm = SVC()
# modelSVM_train = svm.fit(x_train, y_train)
# train_score = svm.score(x_train, y_train)
# print('The linear SVM model fit on the training data set: ', modelSVM_train)

# Predicting the results of the model on the test data
# y_predictSVM = svm.predict(x_test)

# Computing the error rate of the model fit
# accuracy_svm = round(svm.score(x_train, y_train) * 100, 2)
# print('Accuracy of the linear SVM model fit is ', accuracy_svm)

# Building K-Nearest Neighbours Model on training data set
# knnModel = KNeighborsClassifier()
# knnModel.fit(x_train, y_train)

# Predicting the model fit on test data set
# knnPredict = knnModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
# knnScore = metrics.accuracy_score(y_test, knnPredict)
# print('The score obtained for K-Nearest Neighbours model is', knnScore)