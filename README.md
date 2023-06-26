### KMeans_algo

K-means is an unsupervised learning method for clustering data points. The algorithm iteratively divides data points into K clusters by minimizing the variance in each cluster.

each data point is randomly assigned to one of the K clusters. Then, we compute the centroid (functionally the center) of each cluster, and reassign each data point to the cluster with the closest centroid. 

We repeat this process until the cluster assignments for each data point are no longer changing.

K-means clustering requires us to select K, the number of clusters we want to group the data into. 

The elbow method lets us graph the inertia (a distance-based metric) and visualize the point at which it starts decreasing linearly. This point is referred to as the "eblow" and is a good estimate for the best value for K based on our data.

### Inertia

Inertia is not a normalized metric.

The lower values of inertia are better and zero is optimal.

But in very high-dimensional spaces, euclidean distances tend to become inflated (this is an instance of curse of dimensionality).

Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.

### Elbow Method
The Elbow method is one of the most popular ways to find the optimal number of clusters. This method uses the concept of WCSS value. 

WCSS stands for Within Cluster Sum of Squares, which defines the total variations within a cluster. 

The formula to calculate the value of WCSS (for 3 clusters) is given below:

WCSS= ∑Pi in Cluster1 distance(Pi C1)2 +∑Pi in Cluster2distance(Pi C2)2+∑Pi in CLuster3 distance(Pi C3)2

In the above formula of WCSS,

∑Pi in Cluster1 distance(Pi C1)2: It is the sum of the square of the distances between each data point and its centroid within a cluster1 and the same for the other two terms.

To measure the distance between data points and centroid, we can use any method such as Euclidean distance or Manhattan distance.

To find the optimal value of clusters, the elbow method follows the below steps:

o	It executes the K-means clustering on a given dataset for different K values (ranges from 1-10).

o	For each value of K, calculates the WCSS value.

o	Plots a curve between calculated WCSS values and the number of clusters K.

o	The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K.

Since the graph shows the sharp bend, which looks like an elbow, hence it is known as the elbow method. The graph for the elbow method looks like the below image:

![image](https://github.com/Tejashripatil25/KMeans_algo/assets/124791646/bdc5ae35-67b5-4c32-ad76-b35c2381c10d)


###  Applications of clustering

 K-Means clustering is the most common unsupervised machine learning algorithm. It is widely used for many applications which include-

#### Image segmentation

#### Customer segmentation

#### Species clustering

#### Anomaly detection

#### Clustering languages

### Example:

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




