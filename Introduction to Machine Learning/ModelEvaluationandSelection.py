# Model selection is the phase after fitting the model to chck the performance of our classifier and if it
# does not match as par than we can perform entire cycle i.e. choosing the classifier again .

# Imbalanced class scenario - When there are more negative class items or positive items than the other one

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target
for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name,class_count) # np.bincount - Count the number of individual values

# Here the above example has balanced classes . Now let's create an imbalanced class by if the class value
# is not 1 in above example than it is given class as 0
# Creating a dataset with imbalanced binary classes:
# Negative class (0) is 'not digit 1'
# Positive class (1) is 'digit 1'
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])

print(np.bincount(y_binary_imbalanced)) # Class 1 has only 182 occurences while class 2 has 1615 showing
# imbalance

# Training an SVC classifier using imbalanced classes gives an accuracy of nearly 90 percent
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
print(svm.score(X_test, y_test))

# DummyClassifier is a classifier that makes predictions using simple rules, which can be
# useful as a baseline for comparison against actual classifiers, especially with imbalanced classes.

from sklearn.dummy import DummyClassifier

# Negative class (0) is most frequent
# There are other parameters for strategy
# most_frequent - Predicts the most frequent class
# stratified - Random predictions made on training set classifiers
# uniform - generates predictions uniformly at random
# constant - Always predict a constant label provided by user
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0 (Most frequent class)
y_dummy_predictions = dummy_majority.predict(X_test)
print(y_dummy_predictions)
print(dummy_majority.score(X_test, y_test))

# As we can see the dummy classifier score is approx equal to normal classifier score which signifies
# that many times classifier takes the dominant values in label set which leads

# The dummy classifier provides what is called a null accuracy baseline. That is the accuracy that can
# be achieved by always picking the most frequent class.

# If we change the support vector classifier's kernel parameter to linear from rbf. And recompute the
# accuracy on this retrain classifier, we can see that this leads to much better performance of almost
# 98% compared to the most frequently class based line of 90%.

#  If you have accuracy that is close to that of a dummy classifier, it could be because there is indeed
#  a large class imbalance. And the accuracy gains produced by the classifier on the test set simply applied
#  too few examples to produce a significant gain.

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
print(svm.score(X_test, y_test))

# Confusion matrices - Suppose we have n feautures for our model than confusion matrix is a matrix of
# dimension n * n of form [[TN,FP],[FN,TP]] where TN indicates True Positive that how much correct values
# for negative class has been predicted with FP as False Positives which means that they have to predicted
# as negative but falsely has been predicted as Positive. Same for the second row. Hence FP and FN denotes
# the error values for a particular model FP(Is Negative but predicted positive), FN(Is Positive but predicted
# negative). Following examples shows this confusion matrix for the classifiers

from sklearn.metrics import confusion_matrix

# Negative class (0) is most frequent. Here it shows all the FP as 0 because dummy classifier always predict
# 0 class since it takes it from majority
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('Most frequent class (dummy classifier)\n', confusion)


# produces random predictions w/ same class proportion as training set
dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_classprop_predicted = dummy_classprop.predict(X_test)
confusion = confusion_matrix(y_test, y_classprop_predicted)

print('Random class-proportional prediction (dummy classifier)\n', confusion)

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print('Support vector machine classifier (linear kernel, C=1)\n', confusion)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print('Decision tree classifier (max_depth = 2)\n', confusion)

# We can often calculate certain parameters using the confusion matrix
# TN - True Negative - How much negative class that is predicted is true
# FN - False Negative - How much negative class that is predicted is false
# TP - True Positive - How much positive class that is predicted is true
# TN - True Negative - How much positive class that is predicted is false
# Accuracy - TP + TN / TP + FN + TN + FP
# Precision - TP / TP + FP - Website example
# Recall - TP / TP + FN - Tumor example


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))

# Combined report with all above metrics
from sklearn.metrics import classification_report

print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))

print('Random class-proportional (dummy)\n',
      classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
print('SVM\n',
      classification_report(y_test, svm_predicted, target_names = ['not 1', '1']))
print('Logistic regression\n',
      classification_report(y_test, lr_predicted, target_names = ['not 1', '1']))
print('Decision tree\n',
      classification_report(y_test, tree_predicted, target_names = ['not 1', '1']))

# Decision Functions or Predict proba predicts the probability of a particular class instance for predict
# proba and values for decision function which is how far the distance is from the decision boundary. We
# can change the decision boundary to adjust Recall, Precision , Accuracy by changing the decision threshold

# show the decision_function scores for first 20 instances - The value returned is the distance from the
# decision hyperplane
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))


# show the probability of positive class for first 20 instances
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))

# We can plot the curve between precision and recall by changing the decision boundary in decision functions
# and see the results how precision and recall vary w.r.t. each other
# the x axis shows precision and the y axis shows recall

# An ideal classifier would be able to achieve perfect precision of 1.0 and perfect recall of 1.0. So the
# optimal point would be up here in the top right. And in general, with precision recall curves, the
# closer in some sense, the curve is to the top right corner, the more preferable it is, the more
# beneficial the tradeoff it gives between precision and recall

# ROC curves or receiver operating characteristic curves are a very widely used visualization method that
# illustrate the performance of a binary classifier.

# ROC curves on the X-axis show a classifier's False Positive Rate so that would go from 0 to 1.0, and
# on the Y-axis they show a classifier's True Positive Rate so that will also go from 0 to 1.0. The ideal
# point in ROC space is one where the classifier achieves zero, a false positive rate of zero, and a true
# positive rate of one. So that would be the upper left corner.

# So curves in ROC space represent different tradeoffs as the decision boundary, the decision threshold
# is varied for the classifier. So just as in the precision recall case, as we vary decision threshold,
# we'll get different numbers of false positives and true positives that we can plot on a chart.

# we can qualify the goodness of a classifier in some sense by looking at how much area there is
# underneath the curve.

# Plotting the precision recall curve

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]
print("Threshold", thresholds) # Decision Threshold
print("Closest Zero",closest_zero) # Value closest to 0

plt.close()
plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
# plt.axes().set_aspect('equal')
plt.show()