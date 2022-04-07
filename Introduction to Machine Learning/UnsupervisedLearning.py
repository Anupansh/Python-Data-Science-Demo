# Unsupervised learning is when we have to raw data but there is no raw or target value to predict.
# Like it is used when we want to find some kind of patterns, groups or to find some strcture betwwen the
# data . Like for a user visiting a website it might be important to know that which page users visit for
# frequently by capturing the browsing data

# By clustering users into groups, you might gain some insight into who your typical customers are and
# what site features different types of users find important.

# There is one method we use to transform the data into useful ways. Density estimation curve which plot
# the most dense (having more data points all together) and less data points together. Which estimates
# that where there is more data and where is less.

#  density estimation calculates a continuous probability density over the feature space, given a set
#  of discrete samples in that feature space.
# With this density estimate, we can estimate how likely any given combination of features is to occur.

# In Scikit-Learn, you can use the kernel density class in the sklearn.neighbors module to perform one
# widely used form of density estimation called kernel density estimation.

# Dimensionality Reduction - As the name suggests it reduces the dimensions of the dataset
# (i.e features of dataset). Consider a cloud of data points in 2D dataspace

# One very important form of dimensionality reduction is called principal component analysis, or PCA.
# What PCA does is take the original cloud of data points and rotate them in the direction of the highest
# variance i.e. the longest line between the end data points also called as first principal component.
# Than it plots the right angles and try to cover the remaining data points known as second principal
# components hence the two feature space gets converted into a single dimension location space whose
# position is given wrt to two dimensionality lines

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Our sample fruits dataset
fruits = pd.read_table('readonly/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1

# Using PCA to find the first two principal components of the breast cancer dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)

# n_components indicates how how much dimensionality reduction we want i.e in below example
# n_components = 2 indicates that we want to reduce it to 2 - Dimension
pca = PCA(n_components = 2).fit(X_normalized)

X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

from adspy_shared_utilities import plot_labelled_scatter
plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Breast Cancer Dataset PCA (n_components = 2)');

# Manifold learning - It is very good at finding low dimensional structure in high dimensional data
# and are very useful for visualizations

# One widely used manifold learning method is called multi-dimensional scaling, or MDS. There are many
# flavors of MDS, but they all have the same general goal; to visualize a high dimensional dataset and
# project it onto a lower dimensional space - in most cases, a two-dimensional page - in a way that
# preserves information about how the points in the original data space are close to each other.


from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

# each feature should be centered (zero mean) and with unit variance
# X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)
#
# mds = MDS(n_components = 2)
#
# X_fruits_mds = mds.fit_transform(X_fruits_normalized)
#
# plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
# plt.xlabel('First MDS feature')
# plt.ylabel('Second MDS feature')
# plt.title('Fruit sample dataset MDS');

# An especially powerful manifold learning algorithm for visualizing your data is called t-SNE. t-SNE
# finds a two-dimensional representation of your data, such that the distances between points in the 2D
# scatterplot match as closely as possible the distances between the same points in the original high
# dimensional dataset. In particular, t-SNE gives much more weight to preserving information about
# distances between points that are neighbors.

# from sklearn.manifold import TSNE
#
# tsne = TSNE(random_state = 0)
#
# X_tsne = tsne.fit_transform(X_fruits_normalized)
#
# plot_labelled_scatter(X_tsne, y_fruits,
#                       ['apple', 'mandarin', 'orange', 'lemon'])
# plt.xlabel('First t-SNE feature')
# plt.ylabel('Second t-SNE feature')
# plt.title('Fruits dataset t-SNE');
