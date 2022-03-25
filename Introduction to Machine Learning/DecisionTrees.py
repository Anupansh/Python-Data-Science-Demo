# Decision tree perform a set of if else rules on a data and produce a target value. Can be used to get a better
# idea of more influential features. We continue to narrow down the possible results by asking more and more
# questions like if yes than than next question with yes or no

#  We can form these questions into a tree with a node representing one question and the yes or no possible
#  answers as the left and right branches from that node that connect the node to the next level of the tree.
#  One question being answered at each level. At the bottom of the tree are nodes called leaf nodes that
#  represent actual objects as the possible answers.

#  Rather than try to figure out these rules manually for every task, there are supervised algorithms that can learn
#  them for us in a way that gets to an accurate decision quickly.

# The goal when building a decision tree is to find the sequence of questions that has the best accuracy at classifying
# the data in the fewest steps.

#  an informative split of the data is one that does an excellent job at separating one class from the others
# A less informative split. Like a rule like sepal width less than or equal to three centimeters, would not produce
# as clear a separation of one class from the others. So for the best split, the results should produce as homogeneous
# a set of classes as possible

#  Trees whose leaf nodes each have all the same target value are called pure, as opposed to mixed where the leaf
#  nodes are allowed to contain at least some mixture of the classes.

# Decision trees can also be used for regression using the same process of testing the future values at each node and
# predicting the target value based on the contents of the leafnode. For regression, the leafnode prediction would be
# the mean value of the target values for the training points in that leaf.

# Since the Decision trees seprates the data until it get all the leaf nodes that creates a complex model
# and creates a chance of over fitting
# One strategy to prevent overfitting is to prevent the tree from becoming really detailed and complex
# by stopping its growth early. This is called pre-pruning. Another strategy is to build a complete tree
# with pure leaves but then to prune back the tree into a simpler form. This is called post-pruning or
# sometimes just pruning.

# Feature importance is typically a number between 0 and 1 that's assigned to an individual feature.
# It indicates how important that feature is to the overall prediction accuracy

#  A feature importance of zero means that the feature is not used at all in the prediction. A feature importance
#  of one, means the feature perfectly predicts the target.

# The pedal length feature easily has the largest feature importance weight in iris dataset. We can confirm
# this by looking at the decision tree that this is indeed corresponds to that features position at the top
# of the decision tree, showing that this first level just using the petal length feature

# Three important factors to control the complexity of decision tree
# Max Depth - Controls the maximum depth of the tree and reduces the points to split
# Max leaf - Limits the total number of nodes that are leaves of the tree.
# The min samples leaf parameter - It is the threshold that controls what the minimum number of data instances has to
# be in a leaf to avoid splitting it further for ex - In a leaf there must not be less than 30 samples


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=3)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test))) # Leading to overfit

# Setting max decision tree depth to help avoid overfitting

clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf2.score(X_test, y_test)))

# Visualizing decision trees (Pre pruned version - Means no extra parameters)

plot_decision_tree(clf, iris.feature_names, iris.target_names)

# Feature importance

from adspy_shared_utilities import plot_feature_importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf, iris.feature_names)

print('Feature importances: {}'.format(clf.feature_importances_))

from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train

    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                              None, title, axis,
                                              iris.target_names)

    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])

plt.tight_layout()
plt.show()