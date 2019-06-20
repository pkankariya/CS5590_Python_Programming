# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


# Reading data
customer = pd.read_csv('CC.csv')
x = customer.iloc[:, 1:17]
y = customer.iloc[:, -1]
x = x.fillna(0)

# Standardization of the data
scaler = StandardScaler()
scaler.fit(x)

# Projecting data on reduced dimension
x_scaler = scaler.transform(x)

# Performing Principle Component Analysis (PCA)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, customer["TENURE"]], axis=1)
print(finaldf)

# Bonus: KMeans on PCA
# Performing K-Means clustering on the PCA data
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x_pca)

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_KMeans, metric='euclidean', sample_size=50)
print('Silhoutee Score of the Clusters using PCA is ', score)

# Elbow point computation to determine optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)

#Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()