import matplotlib.pyplot as plt
import numpy as np

years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])
lowtemp = np.array([3.4, 3.8, 4.2, 3.7, 3.4, 4.6, 4.0, 3.2, 4.6, 3.0, 2.0])
hightemp = np.array([38.2, 39.4, 39.0, 40.1, 40.6, 41.0, 39.7, 38.8, 40.6, 41.3, 42.0])

# plt.figure()
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(years[:-1], lowtemp[:-1], '-o', years[:-1], hightemp[:-1], '-o')
plt.gca().fill_between(years[:-1], lowtemp[:-1], hightemp[:-1], facecolor="purple", alpha=0.4)
plt.xticks(years)
plt.scatter([years[len(years) - 1]] * 2, [lowtemp[len(lowtemp) - 1], hightemp[len(hightemp) - 1]], color=["green"] * 2)
plt.text(x=years[len(years) - 1], y=lowtemp[len(lowtemp) - 1], s="27 June")
plt.text(x=years[len(years) - 1], y=hightemp[len(hightemp) - 1], s="14 Dec")
plt.title("Temperature over the years in Delhi, India")
plt.xlabel("Years")
plt.ylabel("Temperature in Celsius")
plt.legend(["Recorded Lows", "Recorded Highs", "Difference", "Record breaking high and low"])
plt.show()
