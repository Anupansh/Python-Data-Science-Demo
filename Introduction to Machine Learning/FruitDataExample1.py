import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fruits = pd.read_table("assets/fruit_data_with_colors.txt")
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(),
                             fruits.fruit_name.unique()))  # Will fetch distinct fruit label along with their name

# Will divide the training data entries in ratio of 3:1 which we use as train and test data
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Creating a scatter matrix to represent properties
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9),cmap=cmap)

# Creating a 3D scatter plot to visualize more accurately

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')

# k - Nearest Nearest Algorithm - For a given point the classifier (function) will search for its k-th neighbour meaning
# that if k = 1 it will look for a single point near it in the feature dimension and alot its label to that particular
# new row/ entry. Objects are identified on the basis of boundaries in which they fall . The line that seprates a region
# from another region is called decision boundary because points on one side match to one class and other to other class
# Areas are classified on the basis of euclidean distance that for a given point which distance is less and which region
# will it fall. In case of k more than 1 we take the vote from majority. Usually value of k is odd to have a decision
# otherwise if there is likewise voting than we can pick any of the neighbour. Four steps
# 1- Distance metric like how to specift differnece between the k - Neibours - Most commonly used Eucledean distance
# 2 - Value of k (Number of neighbours)
# 3 - Any additional weightage if to be given to any neighbour
# 4 - How to aggregate after getting all the final details / label by majority voting


# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Import the kNeigbour module
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # Setting necessary parameter i.e the value of k

knn.fit(X_train, y_train) # Train the classifier with  training data once it is done it changes the state and store the
                            # training data somewhere in memory with classifier becoming trained

knn.score(X_test, y_test) # Determine the accuracy of the test that how much label matches the original values

# Use the trained k-NN classifier model to classify new, previously unseen objects

fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print(lookup_fruit_name[fruit_prediction[0]])

fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print(lookup_fruit_name[fruit_prediction[0]])

# Check the accuracy for k from 1-20 and plot a scatter plot to check where the accuracy is more
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

# How sensitive is k-NN classification accuracy to the train/test split proportion?

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');


# Example function to plot a bar graph with value onto the bar and frame removed 

# def accuracy_plot():
#     import matplotlib.pyplot as plt
#
#     %matplotlib notebook
#
#     X_train, X_test, y_train, y_test = answer_four()
#
#     # Find the training and testing accuracies by target value (i.e. malignant, benign)
#     mal_train_X = X_train[y_train==0]
#     mal_train_y = y_train[y_train==0]
#     ben_train_X = X_train[y_train==1]
#     ben_train_y = y_train[y_train==1]
#
#     mal_test_X = X_test[y_test==0]
#     mal_test_y = y_test[y_test==0]
#     ben_test_X = X_test[y_test==1]
#     ben_test_y = y_test[y_test==1]
#
#     knn = answer_five()
#
#     scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
#               knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]
#
#
#     plt.figure()
#
#     # Plot the scores as a bar chart
#     bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])
#
#     # directly label the score onto the bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
#                        ha='center', color='w', fontsize=11)
#
#     # remove all the ticks (both axes), and tick labels on the Y axis
#     plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
#
#     # remove the frame of the chart
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)
#
#     plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
#     plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
