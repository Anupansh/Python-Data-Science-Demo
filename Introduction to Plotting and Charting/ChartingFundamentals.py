import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

# Subplots
plt.figure()
linear = np.array([1, 2, 3, 4, 5, 6, 7, 8])
plt.subplot(2, 1, 1)  # Denotes two rows, 1 column and will be the first figure
plt.plot(linear, '-o')
quadratic = np.array([1, 4, 9, 16, 25, 36, 49, 64])
plt.subplot(2, 1, 2)  # Denotes two rows, 1 column and will be the second figure
plt.plot(quadratic, '-o')
# plt.show()
plt.close()

plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.plot(linear, '-o')
# pass sharey=ax1 to ensure the two subplots share the same y axis
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(quadratic, '-x')
# plt.show()
plt.close()

# create a 3x3 grid of subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
# plot the linear_data on the 5th subplot axes
ax5.plot(linear, '-')
# set inside tick labels to visible
for ax in plt.gcf().get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)
plt.gcf().canvas.draw()
# plt.show()
plt.close()

# Histograms

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]

# draw n = 10, 100, 1000, and 10000 samples from the normal distribution and plot corresponding histograms
for n in range(0, len(axs)):
    sample_size = 10 ** (n + 1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample, bins=250)
    axs[n].set_title('n={}'.format(sample_size))

# repeat with number of bins set to 100

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]

for n in range(0, len(axs)):
    sample_size = 10 ** (n + 1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample, bins=100)
    axs[n].set_title('n={}'.format(sample_size))

plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)  # loc is mean, scale is S.D. and size is no. of elements
X = np.random.random(size=10000)  # List of size 10000 between 0 and 1
plt.scatter(X, Y)
# plt.show()
plt.close()

# Plotting through grid
import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
lower_right.scatter(X, Y)
top_histogram.hist(X, bins=100)
s = side_histogram.hist(Y, bins=100, orientation='horizontal')

# clear the histograms and plot normed histograms
top_histogram.clear()
top_histogram.hist(X, bins=100)
side_histogram.clear()
side_histogram.hist(Y, bins=100, orientation='horizontal')
# flip the side histogram's x axis
side_histogram.invert_xaxis()

# change axes limits
for ax in [top_histogram, lower_right]:
    ax.set_xlim(0, 1)
for ax in [side_histogram, lower_right]:
    ax.set_ylim(-5, 5)

# plt.show()
plt.close()

# Box Plots - Used to display minimum , maximum, median , 1st quartile and 3 quartile value over the graph
import pandas as pd

normal_sample = np.random.normal(loc=0.0, scale=1.0, size=10000)
random_sample = np.random.random(size=10000)
gamma_sample = np.random.gamma(2, size=10000)

df = pd.DataFrame({'normal': normal_sample,
                   'random': random_sample,
                   'gamma': gamma_sample})

df.describe()  # Print the data insights of df
# clear the current figure
plt.clf()
# plot boxplots for all three of df's columns
_ = plt.boxplot([df['normal'], df['random'], df['gamma']])

plt.figure()
_ = plt.hist(df['gamma'], bins=100)

import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([df['normal'], df['random'], df['gamma']])
# overlay axis on top of another
ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2)
ax2.hist(df['gamma'], bins=100)
ax2.margins(x=0.5)  # Sets the margin from parent graph

# switch the y axis ticks for ax2 to the right side
ax2.yaxis.tick_right()

# plt.show()
plt.close()
# Heatmaps

plt.figure()
arr = np.array([2, 3, 5, 6, 7, 10, 34, 12, 7, 11, 18, 18])
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 6])
plt.hist2d(x, arr)
plt.colorbar()
# plt.show()

# Animations
import matplotlib.animation as animation

n = 10
x = np.random.randn(n)  # Returns a normal random sample for n
print(x)


def update(currentFrame):
    if currentFrame == n:  # If last value stop updating the frame
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)  # Arrange the x axis with lower set to -4 and higher to 4 with gap of 0.5
    plt.hist(x[:currentFrame], bins=bins)
    print("Current frame", currentFrame)
    print("x", x[0:currentFrame])
    plt.axis([-4, 4, 0, 30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(currentFrame), [3, 27])


fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)
# plt.show()
plt.close()

# Interactivity

plt.figure()
data = np.random.rand(10)
print(data)
plt.plot(data)


def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))


# tell mpl_connect we want to pass a 'button_press_event' into onclick when the event is detected
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

from random import shuffle
origins = ['China', 'Brazil', 'India', 'USA', 'Canada', 'UK', 'Germany', 'Iraq', 'Chile', 'Mexico']

shuffle(origins)

df = pd.DataFrame({'height': np.random.rand(10),
                   'weight': np.random.rand(10),
                   'origin': origins})

plt.figure()
# picker=5 means the mouse doesn't have to click directly on an event, but can be up to 5 pixels away
plt.scatter(df['height'], df['weight'], picker=5)
plt.gca().set_ylabel('Weight')
plt.gca().set_xlabel('Height')

def onpick(event):
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))

# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)

plt.show()
