# Use the following data for this assignment:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import TextBox

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000, 200000, 3650),
                   np.random.normal(43000, 100000, 3650),
                   np.random.normal(43500, 140000, 3650),
                   np.random.normal(48000, 70000, 3650)],
                  index=[1992, 1993, 1994, 1995])


def getstats(row):
    row["mean"] = row.mean()
    row["error"] = row.std() / math.sqrt(len(row) - 1)
    row["min"] = row.mean() - row.std() / math.sqrt(len(row) - 1)
    row["max"] = row.mean() + row.std() / math.sqrt(len(row) - 1)
    return row


def applyColor(row):
    if y < row["min"]:
        row["color"] = "blue"
    elif y > row["max"]:
        row["color"] = "red"
    else:
        row["color"] = "white"
    return row

def visualizeGraph(expr):
    print(expr)

plt.figure()
ax = plt.gca()
fig, axes = plt.subplots()
fig.subplots_adjust(bottom=0.2)
graphBox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
txtBox = TextBox(graphBox, "Plot: ")
txtBox.on_submit(visualizeGraph)
txtBox.set_val(41000)
ax.set_ylim([0, 55000])
y = 41000
plt.axhline(y=y, color='purple', linestyle='-')
df = df.apply(getstats, axis=1)
df = df[["mean", "error", "min", "max"]]
df = df.apply(applyColor, axis=1)
plt.bar(df.index.astype(str), df["mean"], width=0.5, color=df["color"], edgecolor=["black"] * 4)
plt.errorbar(df.index.astype(str), df["mean"], color='lightgreen', fmt='o', ecolor='black', yerr=df["error"],
             elinewidth=5)
plt.legend(["{}".format(y), "Below y", "Mean standard deviation"], loc='upper left')
plt.xlabel("Years")
plt.ylabel("Mean of datasets")
plt.title("Random distribution of some dataset over the years")
plt.show()
