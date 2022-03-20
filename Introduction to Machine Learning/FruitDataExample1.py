import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fruits = pd.read_table("assets/fruit_data_with_colors.txt")
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique())) # Will fetch distinct fruit label along with their name

# Will divide the training data in ratio of 3:1 which we use as train and test data
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)