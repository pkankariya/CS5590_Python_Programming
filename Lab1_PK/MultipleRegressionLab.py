# Importing libraries
from sklearn.datasets import load_diabetes
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Reading bank input data
bankData = pd.read_csv('bank.csv')
#x = bankData[:, 0:16]
#y = bankData[:, 16]
print(bankData.y.describe())

# Converting non-numerical data into numerical along with fine tuning the features needed as input


# Evaluating the skewness of y (customer's subscription towards a term deposit)
print('Skewness of target y is ', bankData.y.skew())
plt.hist(bankData.y, color='royal')
plt.title('Skewness of y (Term Deposit Subscription)')
plt.show()

# Normalize the skewed distribution of y (customer's subscription towards a term deposit)
target = np.log(bankData.y)
print('Normalized Distribution of y (Term Deposit Subscription)', target.skew())
plt.hist(target, color='royal')
plt.title('Normalization of y (Term Deposit Subscription)')
plt.show()

