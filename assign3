import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler

# Read data 
df1 = pd.read_csv("protein-angle-dataset.csv")

"""plt.figure()
plt.scatter(df1["phi"],df1["psi"], s=10, c='darkblue', marker='*',alpha=0.1)
#plt.hist2d(df1["phi"],df1["psi"], bins=(80, 80), cmap=plt.cm.jet)
plt.xlim(-180,180.001)
plt.ylim(-180,180.001)
plt.xticks(np.arange(-180,180.01,40), fontsize=8)
plt.yticks(np.arange(-180,180.01,40), fontsize=8)
plt.xlabel('Phi Distribution')
plt.ylabel('Psi Distribution')
plt.title("Distribution of phi VS psi")
plt.legend()
#plt.colorbar()
plt.show()"""

#Generate features and labels:
features = df1[["phi","psi"]].values

kmeans_kwarg = {"init":"random","n_init":10,"max_iter":200}

#Calculate optimal k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,**kmeans_kwarg)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)

#Plot elbow-graph
plt.figure()
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")
#elbow = kl.elbow

elbow = 3  #Manual input from graph
# Fit KMeans with the optimal number of clusters (k 3)
kmeans = KMeans(n_clusters=elbow, **kmeans_kwarg)
kmeans.fit(features)

# Predict cluster labels for each data point
cluster_labels = kmeans.predict(features)

# Plot the data points colored by cluster labels
plt.figure()
plt.scatter(df1["phi"], df1["psi"], s=10, c=cluster_labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.xlim(-180, 180.001)
plt.ylim(-180, 180.001)
plt.xticks(np.arange(-180, 180.01, 40), fontsize=8)
plt.yticks(np.arange(-180, 180.01, 40), fontsize=8)
plt.xlabel('Phi Distribution')
plt.ylabel('Psi Distribution')
plt.title("K-Means Clustering (k = {})".format(elbow))
plt.colorbar(label='Cluster')
plt.show()
