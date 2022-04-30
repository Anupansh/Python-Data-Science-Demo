# Neural networks works as an enhanced form of linear and logistic regression . Different features
# produces different weights and pass them through a non linear logistic function. Neural networks
# add a hidden layer which are basically known as activation function.
# Each activation function than produces a differnet value h0.h1.h2.... and than uses other weights v0,v1..
# to form the final function y
# Above phenomenon is called a multi-layer perceptron. Which sometimes abbreviate by MLP. These are
# also known as feed-forward neural networks.

# MLPs take this idea of computing weighted sums of the input features, like we saw in logistic regression.
# But it takes it a step beyond logistic regression, by adding an additional processing step called a
# hidden layer. Represented by this additional set of boxes, h0, h1, and h2 in the diagram.
# These boxes, within the hidden layer, are called hidden units. And each hidden unit in the hidden
# layer computes a nonlinear function of the weighted sums of the input features. Resulting in
# intermediate output values, v0, v1, v2. Then the MLP computes a weighted sum of these hidden unit
# outputs, to form the final output value, Y hat.

#  The three main activation functions we'll compare later in this lecture are the hyperbolic tangent.
#  That's the S-shaped function in green. The rectified linear unit function, which I'll abbreviate to
#  relu, shown as the piecewise linear function in blue and the logisticn function. The relu activation
#  function is the default activation function for neural networks in scikit-learn. It maps any negative
#  input values to zero. The hyperbolic tangent function, or tanh function. Maps large positive input
#  values to outputs very close to one. And large negative input values, to outputs very close to
#  negative one.

# Synthetic Dataset : 1 hidden layer
# Increasing the units lead to more complexity

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


from sklearn.neural_network import MLPClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                        centers = 8, cluster_std = 1.3,
                        random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for units, axis in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs',
                          random_state = 0).fit(X_train, y_train)

    title = 'Dataset 1: Neural net classifier, 1 layer, {} units'.format(units)

    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                              X_test, y_test, title, axis)
    plt.tight_layout()

# There can be multiple hidden layers for more complex functions and complex datasets
# By default, if you don't specify the hidden_layer_sizes parameter, scikit-learn will create a
# single hidden layer with 100 hidden units

from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs',
                      random_state = 0).fit(X_train, y_train)

plot_class_regions_for_classifier(nnclf, X_train, y_train, X_test, y_test,
                                  'Dataset 1: Neural net classifier, 2 layers, 10/10 units')

# Alpha regularization parameter - It uses L2 regularization that is decerasing the coefficients
# by of the values by squared sum of weights
#  when alpha is small, the decision boundaries are much more complex and variable.

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))

for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh',
                          alpha = this_alpha,
                          hidden_layer_sizes = [100, 100],
                          random_state = 0).fit(X_train, y_train)

    title = 'Dataset 2: NN classifier, alpha = {:.3f} '.format(this_alpha)

    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                              X_test, y_test, title, axis)
    plt.tight_layout()

# Effect of diffeent types of activation functions

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for this_activation, axis in zip(['logistic', 'tanh', 'relu'], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = this_activation,
                          alpha = 0.1, hidden_layer_sizes = [10, 10],
                          random_state = 0).fit(X_train, y_train)

    title = 'Dataset 2: NN classifier, 2 layers 10/10, {} \
activation function'.format(this_activation)

    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                              X_test, y_test, title, axis)
    plt.tight_layout()

# As with other supervised learning models, like regularized regression and support vector machines.
# It can be critical, when using neural networks, to properly normalize the input features.

# Neural network with Regression

# There is also a solver algorithm that actually does the numerical work of finding the optimal weights.

from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.datasets import make_regression

fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)

X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                             n_informative=1, bias = 150.0,
                             noise = 30, random_state=0)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpreg = MLPRegressor(hidden_layer_sizes = [100,100],
                              activation = thisactivation,
                              alpha = thisalpha,
                              solver = 'lbfgs').fit(X_train, y_train)
        y_predict_output = mlpreg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output,
                      '^', markersize = 10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha={}, activation={})'
                           .format(thisalpha, thisactivation))
        plt.tight_layout()

