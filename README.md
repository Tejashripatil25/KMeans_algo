# KMeans_algo

K-means is an unsupervised learning method for clustering data points. The algorithm iteratively divides data points into K clusters by minimizing the variance in each cluster.

each data point is randomly assigned to one of the K clusters. Then, we compute the centroid (functionally the center) of each cluster, and reassign each data point to the cluster with the closest centroid. 

We repeat this process until the cluster assignments for each data point are no longer changing.

K-means clustering requires us to select K, the number of clusters we want to group the data into. 

The elbow method lets us graph the inertia (a distance-based metric) and visualize the point at which it starts decreasing linearly. This point is referred to as the "eblow" and is a good estimate for the best value for K based on our data.

Example:

Start by visualizing some data points:

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]

y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)

plt.show()

![image](https://github.com/Tejashripatil25/KMeans_algo/assets/124791646/0cec280a-8e35-468c-919b-3d998d4ce87c)

Now we utilize the elbow method to visualize the intertia for different values of K:

from sklearn.cluster import KMeans

data = list(zip(x, y))

inertias = []


for i in range(1,11):

    kmeans = KMeans(n_clusters=i)
    
    kmeans.fit(data)
    
    inertias.append(kmeans.inertia_)
    

plt.plot(range(1,11), inertias, marker='o')

plt.title('Elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()

![image](https://github.com/Tejashripatil25/KMeans_algo/assets/124791646/b8726018-a99d-47fa-9cce-d1f81114ece6)

The elbow method shows that 2 is a good value for K, so we retrain and visualize the result:


kmeans = KMeans(n_clusters=2)

kmeans.fit(data)


plt.scatter(x, y, c=kmeans.labels_)

plt.show()

![image](https://github.com/Tejashripatil25/KMeans_algo/assets/124791646/ad2369bd-3923-471e-ab58-30163d85b4be)



