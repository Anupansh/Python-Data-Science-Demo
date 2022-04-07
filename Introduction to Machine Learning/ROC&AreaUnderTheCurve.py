from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression

dataset = load_digits()
X, y = dataset.data, dataset.target
for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name, class_count)  # np.bincount - Count the number of individual values

y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# fpr - Increasing false positive rates such that element i is the false positive rate of predictions with
# score >= thresholds[i].
# tpr - Increasing true positive rates such that element i is the true positive rate of predictions with score
# >= thresholds[i].
# AUC - Compute Area Under the Curve (AUC) using the trapezoidal rule.
# This is a general function, given points on a curve. For computing the area under the ROC-curve
lr = LogisticRegression().fit(X_train, y_train)
y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

print(roc_auc_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
# plt.axes().set_aspect('equal')
# plt.show()

# Multi class evaluation - Just as binary class evaluation. Multi class evaluation is used on
# the datasets having more than 2 classes . We can perform every operation on multi class which is applicable
# for binary classification also like confusion matrices , classification report etc

dataset = load_digits()
X, y = dataset.data, dataset.target
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

svm = SVC(kernel='linear').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc,
                     index=[i for i in range(0, 10)], columns=[i for i in range(0, 10)])
print(df_cm)

a = 1e-01
print("AAAAAAAAA",a)

plt.figure(figsize=(5.5, 4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc,
                                                                       svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

svm = SVC(kernel='rbf').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, index=[i for i in range(0, 10)],
                     columns=[i for i in range(0, 10)])

plt.figure(figsize=(5.5, 4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM RBF Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc,
                                                                    svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label');

# plt.show()

# Multi class classification report

from sklearn.metrics import classification_report

print("Classification report for multi classification problems is",
      print(classification_report(y_test_mc, svm_predicted_mc)))

# Macro Averaged Matrices - In this kind of average matrix each class has been given equal importance, than for each class
# true predictions / the total predictions is calculated than for each predictions for every class they are added and than
# divided by the total number of classes . For ex there are three classes with prediction score of 0.4,0.3 and 0.3 than for
# macro metric their sum / total classes i.e. 0.4 + 0.3 + 0.3 / 3 = 0.33 will be average macro score

# Micro Averaged MAtrices - It is the same as macro with difference that weigtage is given to every instance rather than
# the class. For ex if there are 10 instances of data with three class the micro will be calulated on the basis of
# correct predictions for each instance divided by total number of instance. For ex - If there are 20 instances in a
# dataset out of which 11 are predicted to be true than the micro score will be 11/20

from sklearn.metrics import precision_score, f1_score

print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average='micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average='macro')))

print('Micro-averaged f1 = {:.2f} (treat instances equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average='micro')))
print('Macro-averaged f1 = {:.2f} (treat classes equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average='macro')))


#  the default r squared score that's available for regression and psychiclearn and that summarizes how
#  well future instances will be predicted. the r2_score for perfect predictor is 1.0. And for a
#  predictor that always output the same constant value, the r2_score is 0.0. The r2_score despite
#  the squared in the name that suggests it's always positive does have the potential to go negative
#  for bad model fits, such as when fitting non-linear functions to data.

#  the dummy regressor achieves an r squared score of 0. Since it always makes a constant prediction without
#  looking at the output as it returns a constant value which is specified as mean, median or anything


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

diabetes = datasets.load_diabetes()

X = diabetes.data[:, None, 6]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy='mean').fit(X_train, y_train)

y_predict = lm.predict(X_test)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

print('Linear model, coefficients: ', lm.coef_)
print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test,
                                                                     y_predict_dummy_mean)))
print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))

# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_predict, color='green', linewidth=2)
plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle='dashed',
         linewidth=2, label='dummy')

plt.show()
