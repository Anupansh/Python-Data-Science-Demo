# Random Forests - Random forests are an ensemble version of decision trees in which
# prediction is based on multiple decision trees. Node in each tree in forest is partitioned on the basis
# of subset of feature rather than using all the features. It is specificed by max_features parameter

# Ensemble means combining multiple models to create a more significant and valid one

# Every tree in forest is build with different sample called bootstrap. To support this every tree has n number of
# rows which is same as the total number of rows. Than rows are choosed randomly and than added to
# every decision tree. Therefore each tree can have same row for more than one time and not neccesary
# that all rows are rows of dataset are there in a tree which shows randomness

# max_features paramter is set to low or one than each tree might contain different feature since it
# partitions on the basis of one feature only that was selected randomly instead of providing the best split
# over several variables
# Meanwhile if value is kept too high than all the trees might contain the same features and would look same

# In case of regression we take the predicted values for each tree and predict the mean of all three.

# In case of classification we have probabilities of being in a class than the average across all the
# trees is taken place for a given class and than the one with more probability is predicted

# Key parameters for random forest
# n_estimators - Set the number of trees to be used. For a larger dimension problem n_estimators should kept high
# max_features - Used to make trees highly different by using the different subset of features available. For
# classification its value is square root of total no. of features . For regression it is log 2 base of total
# number of features
# max_depth - Controls the maximum depth of each tree . The default setting for this is none, in other words, the
# nodes in a tree will continue to be split until all leaves contain the same class or have fewer samples than
# the minimum sample split parameter value, which is two by default
# n_jobs -  Most systems now have a multi-core processor and so you can use the end jobs parameter to tell the
# random forest algorithm how many cores to use in parallel to train the model. If you have four cores, the
# training will be four times as fast as if you just used one. If you set n_jobs to negative one it will use
# all the cores on your system and setting n_jobs to a number that's more than the number of cores on your system
# random - To generate the same random numbers if we want reproducible results

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                        centers = 8, cluster_std = 1.3,
                        random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
                                                    random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = RandomForestClassifier().fit(X_train, y_train)
title = 'Random Forest Classifier, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
                                          y_test, title, subaxes)

# plt.show()

# Random Forest on fruit dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import pandas as pd

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']


X_train, X_test, y_train, y_test = train_test_split(X_fruits.values,
                                                    y_fruits.values,
                                                    random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

title = 'Random Forest, fruits dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train

    clf = RandomForestClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                              None, title, axis,
                                              target_names_fruits)

    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])

plt.tight_layout()
plt.show()

clf = RandomForestClassifier(n_estimators = 10,
                             random_state=0).fit(X_train, y_train)

print('Random Forest, Fruit dataset, default settings')
print('Accuracy of RF classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Application on a real world dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))