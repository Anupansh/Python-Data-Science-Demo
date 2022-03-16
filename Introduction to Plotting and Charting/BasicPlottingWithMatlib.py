import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import random

# Using the scripting layer
plt.plot(5, 2, '.')
plt.close()
# plt.show()

# Plot using artist layer

fig = Figure()
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)
ax.plot(3, 2, '.')

# Plotting with axes

plt.plot(3, 50, '+')
plt.plot(2, 80, '+')
plt.plot(7, 12, '+')
plt.plot(5, 26, '+')  # Plotting multiple points
plt.plot(9, 97, '+')
axis = plt.gca()
print(
    axis.get_children())  # Prints the children of axis like plots or spine - Rendering border of frame including tick marker and background rectanle
axis.axis([0, 10, 0, 100])  # Setting axes endpoints
# plt.show()
plt.close()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x
colors = ["blue"] * (len(x) - 1)
colors.append("yellow")
plt.figure()
plt.scatter(x, y, s=200, c=colors)
plt.close()
# plt.show()

zip_generator = zip([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])  # Will create a tuple of corresponding items

zip_generator = zip([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
x, y = zip(*zip_generator)  # Opposite of zipping - Will restore back to two lists from tuples list
plt.figure()
plt.scatter(x[:2], y[:2], s=100, c="red", label="Tall Students")
plt.scatter(x[2:], y[2:], s=100, c="green", label="Small Students")
plt.xlabel("Number of times child kicked a ball")
plt.ylabel("Grade of the student")
plt.title("Halla Bol ")
plt.legend()
plt.legend(loc=4, title="Legend", frameon=False)
plt.close()
# plt.show()

# Line Plots

linear = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
quadratic = linear ** 2
plt.figure()
plt.plot([23, 56, 89, 34], '--r')  # Will show the dashed line
plt.plot(linear, '-o', quadratic, '-o')  # Identifies -o as . in graphs
plt.xlabel("X axis Value")
plt.ylabel("Y axis value")
plt.title("A title")
plt.legend(["Dashed", "Linear", "Quadratic"])
plt.close()
plt.figure()
observable_dates = np.arange("2017-01-01", "2017-11-01", dtype="datetime64[M]")
observable_dates = list(map(pd.to_datetime, observable_dates))
print(observable_dates)
# observable_dates = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
plt.plot(observable_dates, linear, '-o', observable_dates, quadratic, '-o')
x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(45)
plt.subplots_adjust(bottom=0.25)
plt.gca().fill_between(observable_dates, linear, quadratic, facecolor="green", alpha=0.4)
ax = plt.gca()
ax.set_xlabel("Dates")
ax.set_ylabel("Units")
ax.set_title("Quadratic ($x^2$) vs Linear ($x$) comparison")
# plt.show()
plt.close()

# Bar Plotting

plt.figure()
xvals = range((len(linear)))
plt.bar(xvals, linear, width=0.3)
newvals = []
for item in xvals:
    newvals.append(item + 0.3)
plt.bar(newvals, quadratic, width=0.3, color="red")
# plt.show()
plt.close()

linear_err = [random.randint(0, 40) for x in range(len(linear))]  # For showing some error percent or error margin
plt.bar(xvals, linear, width=0.3, yerr=linear_err)
# plt.show()
plt.close()

plt.figure()
xvals = range(len(linear))
plt.bar(xvals, linear, color="red")
plt.bar(xvals, quadratic, color="green", alpha=0.3, bottom=linear)  # Plotting bar on another bar vaertically
# plt.show()
plt.close()

plt.figure()
plt.barh(xvals, linear, color="red")  # Plotting horizontal bar
plt.barh(xvals, quadratic, left=linear, color="green")
# plt.show()
plt.close()
