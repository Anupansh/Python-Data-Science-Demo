from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import numpy as np
from matplotlib.colors import ListedColormap

# Like any other method linear models also use the line equation w1x1+w2x2+b and send this value to a sign
# function which in return returns a value of +1 if the value is greater than 0 and -1 if the value is lesser
# than 0 . For ex. suppose there is a line x1 - x2 = 0 where x1 and x2 are features where we can approximate
# the value of w1 = 1 , w2 = -1 and b= 0 to fit in this line equation . Now for the points x1=2 and x2=3 the
# value of y will be sign(1*2 -2*3) which result is -1 thus it belongs to -1 class and for points x1=3 and
# x2=2 the value of y will be sign(1*3 -1*2) which result is +1 thus it belongs to +1 class.

# So among all possible classifiers that separate these two classes then, we can define the best classifier
# as the classifier that has the maximum amount of margin. Margin is the distance the width that we can go
# from the decision boundary perpendicular to the nearest data point. Maximum margin is the one in which
# any nearest point from the line has the maximum difference

# Regularisation is being calculated from C parameter . Smaller value of C larger regularization - Chances of
# overfitting are there . Therefore value of C is to be used to minimize underfitting and overfitting

X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y = 0.1,
                                 class_sep = 0.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
title = 'Linear SVC, C = {:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

# Linear Support Vector Machine: Evaluating the C parameter
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                              None, None, title, subplot)
    print("Accuracy",clf.score(X_test,y_test))
plt.tight_layout()
# plt.show()

# Application to real dataset

from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LinearSVC(C=0.000001).fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Multi class classification - Just like binary classification, multi class classification is based on the
# converting the problem into a multi level binary problem. Like for every class there might be either belong
# to this class or not . i.e. If value is above 0 that belongs to this class otherwise not. We get multiple
# coefficients for different classes and intercept values which on putting into the equation predicts the class

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
            c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)

plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)

# Kernelized support vector machines - Sometimes it is impossible to predict the class of a data simply
# by just a linear line due to some complex data. So, we increase the dimension by 1 like if there is a
# feature x, than we can increase its dimension by (x,x^2) and plot a decision boundary between the classes
# After reversing it to same 1D figure we can plot the same decision boundary which comes out to be a
# parabola. Similarly in case of 2D figure we include a third dimension as (x0,x1,1-(x0^2 + x1^2)) and than
# plotting a decision boundary we get a paraboloid decision boundary

# Here two classes are represented by circle and square.
# Radial basis Function Kernel - Using the radial basis function kernel in effect, transforms all the
# points inside a certain distance of the circle class to one area of the transformed +1 dimenson feature space.
# And all the points in the square class outside a certain radius get moved to a different area of the
# feature space.  The linear decision boundary in the higher dimensional feature space corresponds to a
# non-linear decision boundary in the original input space.

# the kernelized SVM can compute these more complex decision boundaries just in terms of similarity
# calculations using a formula between pairs of points in the high dimensional space where the transformed feature
# representation is implicit. This similarity function which mathematically is a kind of dot product
# is the kernel in kernelized SVM. It still uses the maximum margin to find the boundary

# Small gamma means a larger similarity radius which means less tight bound boundaries and more generralization. So that
# points farther apart are considered similar.
# Which results in more points being group together and smoother decision boundaries.
# Gamma is mostly used when we need tight boundaries but that can overfit the method

# On the other hand for larger values of gamma, the kernel value to K is more quickly and points have
# to be very close to be considered similar since the boundaries are so concise. This results in more
# complex, tightly constrained decision boundaries.

# To control model complexity 3 parameters are important 1 - kernel - Type of kernel default 'rbf'
# gamma - RBF kernel width
# C - Regularization paramter
# The similarity function which checks the similarity between the points in feature space is basically a
# kernel

# Classification

from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.datasets import make_blobs


X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,
                        cluster_std = 1.3, random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

# The default SVC kernel is radial basis function (RBF)
plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
                                  X_train, y_train, None, None,
                                  'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
                                  .fit(X_train, y_train), X_train,
                                  y_train, None, None,
                                  'Support Vector Classifier: Polynomial kernel, degree = 3')

# Support Vector Machine with RBF kernel: gamma parameter - Shows if gamma value is high boundaries will be more tight
fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
    clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                              None, None, title, subplot)
    plt.tight_layout()


# Support Vector Machine with RBF kernel: using both C and gamma parameter - If the value of gamma is very more C value
# becomes irrelavnt as it does not shows much effect

from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):

    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                  C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                  X_test, y_test, title,
                                                  subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# Application of SVM on real dataset

from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                    random_state = 0)

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset (unnormalized features)')
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Scaling - If the range of the parameters is too high than a scaling is necessary to bring  the features values
# within a same range using MinMax Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
      .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
      .format(clf.score(X_test_scaled, y_test)))