# Clustering is a method of grouping together data points with similar features together
# In each clustering algorith it might be necessary to scale the data in order to range those points
# within a same range

# k - mean Clustering - Here k stands for number of clusters which has to be specified during initializing
# classifier. k-random points are plotted at feature space. Than distance between the data points is
# calculated from all the random points and the data point that have minimum distance from which point is
# alotted to that random point and form a cluster than random point is shifted a bit towards to centre of that
# cluster than again distance is calculated of all the data points. After a number of changing of random point
# that point becomes constant and that becomes the centre of the cluster.

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']].values
y_fruits = fruits[['fruit_label']] - 1

X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)

kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X_fruits_normalized)

plot_labelled_scatter(X_fruits_normalized, kmeans.labels_,
                      ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

# Agglomerative Clustering - It is a clustering method in which each data point is considered as a cluster.
# Than based on any of the three below methods these clusters are grouped together one by one and than step
# by step form the larger clusters
# Ward - Points are grouped such that they minimize variance
# Average - Average linkage merges the two clusters that have the smallest average distance between their
# points.
# Complete - Complete linkage, which is also known as maximum linkage, merges the two clusters that have
# the smallest maximum distance between their points.

# Dendogram - Since clusters are formed using joining multiple clusters than each cluster for a heirarchy
# for which a graph or tree can be plotted

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from adspy_shared_utilities import plot_labelled_scatter
import matplotlib.pyplot as plt

X, y = make_blobs(random_state = 10)

cls = AgglomerativeClustering(n_clusters = 3)
cls_assignment = cls.fit_predict(X)

plot_labelled_scatter(X, cls_assignment,
                      ['Cluster 1', 'Cluster 2', 'Cluster 3'])

X, y = make_blobs(random_state = 10, n_samples = 10)
plot_labelled_scatter(X, y,
                      ['Cluster 1', 'Cluster 2', 'Cluster 3'])
print(X)

from scipy.cluster.hierarchy import ward, dendrogram
plt.figure()
dendrogram(ward(X))
plt.show()

# DBSCAN - Stands for density-based spatial clustering of applications with noise. One advantage of
# DBSCAN is that you don't need to specify the number of clusters in advance.

# The two main parameters for DBSCAN are min samples and eps. eps is the value within which each point
# must lie to be in a same cluster. There are 3 probabilities of a position of a point that is inside a
# cluster, at the boundary line or outliers (Not within any cluster)
# All points that lie in a more dense region are called core samples.
# Whereas min_samples are the minimum number of points that must lie in a core sample to call it a cluster
# In addition to points being categorized as core samples, points that don't end up belonging to any cluster
# are considered as noise

# And just like with a agglomerative clustering, DBSCAN doesn't make cluster assignments from new data. So we
# use the fit predict method to cluster and get the cluster assignments back in one step.

# With DBSCAN, if you've scaled your data using a standard scalar or min-max scalar to make sure the feature
# values have comparable ranges, finding an appropriate value for eps is a bit easier to do.

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state = 9, n_samples = 25)

dbscan = DBSCAN(eps = 2, min_samples = 2)

cls = dbscan.fit_predict(X)
print("Cluster membership values:\n{}".format(cls))

plot_labelled_scatter(X, cls + 1,
                      ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2'])