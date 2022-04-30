from sklearn import naive_bayes
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

clfrNB = naive_bayes.MultinomialNB()
clfrNB.fit(X_train, y_train)
predicted_labels = clfrNB.predict(X_test)
print(metrics.f1_score(y_test, predicted_labels, average='micro'))

# Using Sk learn SVM classifier

from sklearn import svm
clfrSVM = svm.SVC(kernel='linear', C=0.1)
clfrSVM.fit(X_train, y_train)
predicted_labels = clfrSVM.predict(X_test)

# Model selection using SciKit learn

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train,
                                 test_size = 0.333, random_state = 0)
predicted_labels = model_selection.cross_val_predict(clfrSVM,
                                                     X_train, y_train, cv=5)

# Using NLTK Naive Bayes Classifier
import nltk
from nltk.classify import NaiveBayesClassifier
import pandas as pd
import numpy as np

spam_data = pd.read_csv('resources/spam.csv')
spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)
print(X_train.head())
classifier = NaiveBayesClassifier.train(X_train)
classifier.classify(X_test)
classifier.classify_many(X_test)
print(nltk.classify.util.accuracy(classifier, y_test))
print(classifier.labels())
print(classifier.show_most_informative_features())

# USing NLTK SKlearn classifier

from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
clfrNB = SklearnClassifier(MultinomialNB()).train(X_train)
clfrSVM = SklearnClassifier(SVC(),kernel='linear').train(X_train)