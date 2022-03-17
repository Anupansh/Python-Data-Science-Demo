import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plotting using pandas dataframe
plt.style.use("_mpl-gallery")

np.random.seed(123)

df = pd.DataFrame({'A': np.random.randn(365).cumsum(0),
                   'B': np.random.randn(365).cumsum(0) + 20,
                   'C': np.random.randn(365).cumsum(0) - 20},
                  index=pd.date_range('1/1/2017', periods=365))
df.plot('A', 'B', kind='scatter')

# create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')

ax = df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')
ax.set_aspect('equal')
df.plot.box();
df.plot.hist(alpha=0.7);
df.plot.kde();

plt.clf()
iris = pd.read_csv("iris.csv")
pd.plotting.scatter_matrix(iris);
plt.figure()
pd.plotting.parallel_coordinates(iris, 'Name'); # Will plot line based on diffent names - Each variable in the data set corresponds to an equally spaced parallel vertical line. The values of each variable are then connected by lines between for each individual observation.
plt.show()
