# Importing libraries as needed
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics

# Loading and reading the data set made avaialable
glass_data = pd.read_csv('glass.csv')

# Identifying the predictor and target variables from the features set of data
array = glass_data.values
x = array[:, :9]
y = array[:, 9]

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
model = GaussianNB()
model_train = model.fit(x_train, y_train)
print('The model fit on the training data set: ', model_train)

# Predicting the results of the model on the test data
y_predict = model.predict(x_test)

# Computing the error rate of the model fit
print('Accuracy of the model fit: ', metrics.accuracy_score(y_test, y_predict))