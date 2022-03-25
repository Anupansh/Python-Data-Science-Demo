# Using the train test split entire data is divided into some training data and some test data
# normally in ration of 1:3. Using cross validation we may differ the ratio in which the data is being divided
# For default value of cross validation paramter is set to 5. So, the data is divided into three parts where
# each part is called a fold. Than a single fold is considered as test data and the rest two folds will be
# considered as train data. In the next round second fold is considered as test data and first and third
# fold as training data. This in return gives three scores for training rather than 1

# Class ratio is divided in the folds similar to that as in the original dataset which is known as
# Stratifies k -fold validation

# This method of dividing perform multiple train test split to accurate data more effectively

# If we want to change the number of folds we can change the cv parameter

# One benefit of computing the accuracy of a model on multiple splits instead of a single split, is that
# it gives us potentially useful information about how sensitive the model is to the nature of the specific
# training set. We can look at the distribution of these multiple scores across all the cross-validation
# folds to see how likely it is that by chance, the model will perform very badly or very well on any new
# data set, so we can do a sort of worst case or best case performance estimate from these multiple scores.

#  For regression, scikit-learn uses regular k-fold cross-validation since the concept of preserving class
#  proportions isn't something that's really relevant for everyday regression problems.

# At one extreme we can do something called "Leave-one-out cross-validation", which is just k-fold
# cross-validation, with K sets to the number of data samples in the data set. In other words, each
# fold consists of a single sample as the test set and the rest of the data as the training set

from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.values
y = y_fruits_2d.values
cv_scores = cross_val_score(clf, X, y,cv=3)

print('Cross-validation scores (3-fold):', len(cv_scores))
print('Mean cross-validation score (3-fold): {:.3f}'
      .format(np.mean(cv_scores)))

# Plotting the relation with diffrent valaues of gamma for CVC with cv=3 and gamma to 4 will result two
# 4 * 3 arrays for 3 cross validation paramter and 4 values of gamma

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import numpy as np

param_range = np.logspace(-3, 3, 4)
print(param_range)
train_scores, test_scores = validation_curve(SVC(), X, y,
                                             param_name='gamma',
                                             param_range=param_range, cv=3)

print(train_scores,test_scores)
print(train_scores.shape,test_scores.shape)