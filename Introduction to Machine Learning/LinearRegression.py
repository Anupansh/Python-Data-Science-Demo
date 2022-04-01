# Linear regression
# An equation of type y = a + xb + zc where a is a constant x and z are weights and b and c are the features.
# It basically expresses the relation between feature and target on basis of training set values calculating
# the value of x and y which are basically coefficients
# For a single feature y = ax + b a is Slope and b is represented as y - intercept - Formula of line in terms
# of slope

# Least square method to find the value of a and b is that for a value of x (feature) in training set it is
# the square of the error that is the square of difference between predicted value - actual value. Taking error
# of each training set and dividing by number of observations will be known as mean square error value. Slope
# and y intercept are calculated to minimize this square error. Result will always be a straight line so we
# cannot deal with accuracy .
# Model calucates the value of a and b in a manner that is reduces the square value for all training sets.
# Least squares is what it is called becuase to have a and b in a manner that mean square is reduced to least
# Means we have to minimize Sum of all target values ((y - (ax + b))**2) where y is actual value and ax+b is
# predicted value from line
# Least square method assumes heavily that there is a linear relationship between the x and y

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adspy_shared_utilities import load_crime_dataset
import numpy as np

# Expression for simple linear regression
X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, bias=150, noise=30)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)
print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

# Expression to plot the X values on the graph
plt.figure(figsize=(5, 4))
plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
# plt.show()

# Ridge Regression - Ridge regression is used when there is a correlation between the scales of the features
# Regularisation term is added so that MSE value is increased for which to came to accuracy results weight
# value has to be reduced


from sklearn.linear_model import Ridge

X_Crime,y_crime = load_crime_dataset()

X_train, X_test, y_train, y_test = train_test_split(X_Crime, y_crime,
                                                    random_state = 0)

linridge = Ridge(alpha=20.0).fit(X_train, y_train)

print('ridge regression linear model intercept: {}'
      .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
      .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
      .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
      .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
      .format(np.sum(linridge.coef_ != 0)))

# Feature Normalization - Reducing multiple features to same scales in order to reduce value of w

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_Crime, y_crime,
                                                    random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
      .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
      .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
      .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
      .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
      .format(np.sum(linridge.coef_ != 0)))

# Lasso Regression

from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_Crime, y_crime,
                                                    random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
      .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
      .format(linlasso.coef_))
print('Non-zero features: {}'
      .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
      .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
      .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')

# Polynomial Regression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_friedman1

X_F1, y_F1 = make_friedman1(n_samples = 100,
                            n_features = 7, random_state=0)


X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1,
                                                    random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
      .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
      .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
      .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
      .format(linreg.score(X_test, y_test)))

print('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n')
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                    random_state = 0)
print(X_train.shape)
print(y_train.shape)
linreg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) linear model coeff (w):\n{}'
      .format(linreg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'
      .format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
      .format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
      .format(linreg.score(X_test, y_test)))

print('\nAddition of many polynomial features often leads to\n\
overfitting, so we often use polynomial features in combination\n\
with regression that has a regularization penalty, like ridge\n\
regression.\n')

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                    random_state = 0)
linreg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
      .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
      .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
      .format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
      .format(linreg.score(X_test, y_test)))

