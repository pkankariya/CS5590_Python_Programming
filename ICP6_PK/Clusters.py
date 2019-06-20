# Importing libraries
import pandas as pd
from pandas.tests.groupby.test_value_counts import df
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

# Reading the data & identifying the feature set and target variable to form the clusters
customer = pd.read_csv('CC.csv')
x = customer.iloc[:, 1:17]
y = customer.iloc[:, -1]

# Computing mean of data containing null values to replace them with its mean
meanMinPyt = customer.loc[:, "MINIMUM_PAYMENTS"].mean()
print('Mean of Minimum Payments is ', meanMinPyt)
x = x.fillna(meanMinPyt)

print(x.shape, y.shape)
# Number of customers associated with specific credit period
print(customer["TENURE"].value_counts())

# Cluster identification
#pairs = sns.pairplot(customer, palette='hus1')
sns.FacetGrid(customer, hue="TENURE", size=4).map(plt.scatter, "CREDIT_LIMIT", "BALANCE_FREQUENCY").add_legend()
sns.FacetGrid(customer, hue="TENURE", size=4).map(plt.scatter, "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY").add_legend()
sns.FacetGrid(customer, hue="TENURE", size=4).map(plt.scatter, "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES").add_legend()
plt.show()

# Preprocessing the data for standardizing its distribution
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)

# Performing K-Means clustering on the data available
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_KMeans, metric='euclidean', sample_size=15)
print('Silhoutee Score of the Clusters is ', score)

# Elbow point computation to determine optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()