import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

np.random.seed(1234)

v1 = pd.Series(np.random.normal(0, 10, 1000),
               name='v1')  # Generate a series of 1000 elements with mean of 0 and SD of 10
v2 = pd.Series(2 * v1 + np.random.normal(60, 15, 1000), name='v2')

v1 = random.sample(range(30, 60), 20)
v2 = random.sample(range(50, 80), 20)
print("v1",v1)
print("v2",v2)

# Plotting a histogram with bins starting with -50 and ending at 150 with gap of 2 excluding 150
plt.figure()
plt.hist(v1, alpha=0.7, bins=np.arange(-50, 150, 5), label='v1');
plt.hist(v2, alpha=0.7, bins=np.arange(-50, 150, 5), label='v2');
plt.legend()

plt.figure()
plt.hist([v1, v2], histtype='barstacked',bins=5);
v3 = np.concatenate((v1, v2))
sns.kdeplot(v3)
plt.figure()
print(v3)
# we can pass keyword arguments for each individual component of the plot
sns.histplot(v3,color="deepskyblue",stat="density",bins=5)
sns.kdeplot(v3,color="orange")

# Will create a scatter plot with histograms of both variables on either side
sns.jointplot(x=v1, y=v2, alpha=0.4);

# Sns.jointplot will return the axes of grid in this case Grid. We can than use grid to plot properties
grid = sns.jointplot(x=v1, y=v2, alpha=0.4);
grid.ax_joint.set_aspect('equal')

# Kind hex can be used to see higher data
sns.jointplot(x=v1,y= v2, kind='hex',);

# Setting sns style as white
sns.set_style('dark')

# Will plot a kde on top
sns.jointplot(x=v1, y=v2, kind='kde', space=0);

# Reading from iris csv files
plt.figure()
iris = pd.read_csv('iris.csv')

# Plot categorical data on basis of name
sns.pairplot(iris, hue='Name', diag_kind='kde', height=2);

# Draw a swarmplot and next to it violen plot- for iris data set
plt.figure(figsize=(8,6))
plt.subplot(121)
sns.swarmplot(x='Name', y='PetalLength', data=iris); # Scatter plot for categorical
plt.subplot(122)
sns.violinplot(x='Name', y='PetalLength', data=iris);


plt.show()
