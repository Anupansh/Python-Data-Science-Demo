# Logistic regression is same as linear regression with difference that b + x1w1 ....... that is the equation
# to calculate the y posseses an extra step is by passing that value through a function f  which is known as
# Logistic function that calculates the final value . The f function is an S shaped function of graph which
# gets closer to 1 once and value increases above 0 and produces 0 as value go far beyond 0 which is used to
# produce values between 0 and 1. It is normally used to predict a probability between 0 and 1 like with a
# sample of observation for the students whether they pass or not like the student with no. of hours studied
# more than 3 has value of 1 i.e. passed and those having value less than 3 are failed i.e. probability of 0.
# With these values we can plot a curve on a graph which can calucate the value of weight and constant. Using
# these values we can estimate the probability that a student will pass in an exam or fail .Students having
# pass probability greater than 50 percent estimated to be in a single class otherwise predicted to be in negative
# class . Different classes are associated with different range of prabability which in terms come from training data

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd

# Predicting the class of fruit using logical regression
fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
y_fruits_apple = y_fruits_2d == 1   # make into a binary problem: apples vs everything else
X_train, X_test, y_train, y_test = (
    train_test_split(X_fruits_2d.values,
                     y_fruits_apple.values,
                     random_state = 0))

clf = LogisticRegression(C=100).fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
                                          None, 'Logistic regression \
for binary classification\nFruit dataset: Apple vs others',
                                          subaxes)

h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'
      .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

h = 10
w = 7
print('A fruit with height {} and width {} is predicted to be: {}'
      .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Example on a simple dataset with number of features set to 2 with C regulization set to 1 (default value)

X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y = 0.1,
                                 class_sep = 0.5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,
                                                    random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
clf = LogisticRegression().fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                          None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))


# Logistic regression regularization: C parameter

X_train, X_test, y_train, y_test = (
    train_test_split(X_fruits_2d.values,
                     y_fruits_apple.values,
                     random_state=0))

fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

for this_C, subplot in zip([0.1, 1, 100], subaxes):
    clf = LogisticRegression(C=this_C).fit(X_train, y_train)
    title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)

    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                              X_test, y_test, title,
                                              subplot)
plt.tight_layout()


# Calculating accuracy using a real time database
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
print(len(X_cancer))
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
print("X_Train",X_train)
print("Y_Train",y_train)
plt.figure()
plt.plot(X_train,y_train)
plt.legend()
clf = LogisticRegression().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))
plt.show()