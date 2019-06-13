# Importing libraries to load data and define training and test data sets
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Loading and reading the data set made avaialable
glass_data = pd.read_csv('glass.csv')

# Identifying the predictor and target variables from the features set of data
array = glass_data.values
x = array[:, :9]
y = array[:, 9]
# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1)
model_train = svm.fit(x_train, y_train)
train_score = svm.score(x_train, y_train)
print('The model fit on the training data set: ', model_train)

# Predicting the results of the model on the test data
y_predict = svm.predict(x_test)

# Computing the error rate of the model fit
accuracy_svm = round(svm.score(x_train, y_train) * 100, 2)
print('Accuracy of the model fit is ', accuracy_svm)