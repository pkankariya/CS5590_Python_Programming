# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Reading input data pertaining to home sales
homeData = pd.read_csv('train.csv')

# Evaluating the skewness associated with the target variable: Sale Price
print('Skewness associated to Sales Price is', homeData.SalePrice.skew())
plt.hist(homeData.SalePrice, color='navy')
plt.title('Skewness associated with SalePrice')
plt.show()

# Normalize the skewed distribution of Sale Price
target = np.log(homeData.SalePrice)
print('Normalized Distribution of SalePrice', target.skew())
plt.hist(target, color='navy')
plt.title('Normalized Distribution of SalePrice')
plt.show()

# Scatter plot showing relationship between SalePrice and GarageArea fields
data = pd.concat([homeData['SalePrice'], homeData['GarageArea']], axis=1)
homeData.plot.scatter(x='GarageArea', y='SalePrice', ylim=(0, 500000))
plt.title('Scatter Plot')
plt.show()

# Deleting the anomaly data (outliers) associated with GarageArea
GarageData = pd.concat([homeData['GarageArea']], axis=1, keys=['GaragData'])
homeData = homeData.drop((GarageData[GarageData['GaragData'] < 50]).index)
homeData = homeData.drop((GarageData[GarageData['GaragData'] > 1200]).index)
print('Revised Data Set \n', homeData)

var = 'GarageArea'
newData = pd.concat([homeData['GarageArea'], homeData['SalePrice']], axis=1)
newData.plot.scatter(x=var, y='SalePrice', ylim=(0, 500000))
plt.title('Revised Scatter Plot without Outliers')
plt.show()

