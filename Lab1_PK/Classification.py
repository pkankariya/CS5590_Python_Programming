# Importing libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Loading and reading the data set made available
diabetesData = pd.read_csv('diabetes.csv')
diabetesArray = diabetesData.values
x = diabetesArray[:, :8]
y = diabetesArray[:, 8]
print(x)
print(y)

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
modelNB = GaussianNB()
modelNB_train = modelNB.fit(x_train, y_train)
print('The model fit on the training data set:', modelNB_train)

# Predicting the results of the model on the test data
y_predictNB = modelNB.predict(x_test)

# Computing the error rate of the model fit
print('Accuracy of the Naive Bayes model fit:', metrics.accuracy_score(y_test, y_predictNB))

# Fitting Linear SVM model
svm = SVC()
modelSVM_train = svm.fit(x_train, y_train)
train_score = svm.score(x_train, y_train)
print('The linear SVM model fit on the training data set: ', modelSVM_train)

# Predicting the results of the model on the test data
y_predictSVM = svm.predict(x_test)

# Computing the error rate of the model fit
accuracy_svm = round(svm.score(x_train, y_train) * 100, 2)
print('Accuracy of the linear SVM model fit is ', accuracy_svm)

# Building K-Nearest Neighbours Model on training data set
knnModel = KNeighborsClassifier()
knnModel.fit(x_train, y_train)

# Predicting the model fit on test data set
knnPredict = knnModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
knnScore = metrics.accuracy_score(y_test, knnPredict)
print('The score obtained for K-Nearest Neighbours model is', knnScore)
