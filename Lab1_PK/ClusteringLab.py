# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

# Loading and reading the data set made available
diabetesData = pd.read_csv('diabetes.csv')
x = diabetesData.iloc[:, :8]
y = diabetesData.iloc[:, -1]
print(x.shape, y.shape)

# Number of patients associated with or without diabetes
print(diabetesData["Outcome"].value_counts())

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Elbow point computation to determine optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Cluster identification
sns.FacetGrid(diabetesData, hue="Outcome", size=4).map(plt.scatter, "Glucose", "BMI").add_legend()
sns.FacetGrid(diabetesData, hue="Outcome", size=4).map(plt.scatter, "Glucose", "BloodPressure").add_legend()
sns.FacetGrid(diabetesData, hue="Outcome", size=4).map(plt.scatter, "Glucose", "Insulin").add_legend()
plt.show()

scoreNormal = metrics.silhouette_score(x, y, sample_size=40)
print('Silhoutee Score without the Clusters is ', scoreNormal)

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
score = metrics.silhouette_score(x, y_cluster_KMeans, sample_size=40)
print('Silhoutee Score of the Clusters is ', score)
