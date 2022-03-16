import pandas as pd
import numpy as np
import scipy.stats as stats

# Reading from Energy Indicators
df = pd.read_excel("assets/Energy Indicators.xls", index_col=None, header=None)
Energy = df[18:245].reset_index()
del Energy["index"]
del Energy[0]
del Energy[1]
Energy.rename(columns={2: "Country Name", 3: "Energy Supply", 4: "Energy Supply per Capita", 5: "% Renewable"},
              inplace=True)
Energy["Energy Supply"] = pd.to_numeric(Energy["Energy Supply"], errors="coerce")
Energy["Energy Supply"] = Energy["Energy Supply"] * 1000000
Energy["Country Name"].replace(to_replace=["[\d].*$", "\s\([\w].*\)$"], value="", regex=True, inplace=True)
Energy["Country Name"] = Energy["Country Name"].replace(["Republic of Korea",
                                                         "United States of America",
                                                         "United Kingdom of Great Britain and Northern Ireland",
                                                         "China, Hong Kong Special Administrative Region"],
                                                        ["South Korea",
                                                         "United States",
                                                         "United Kingdom",
                                                         "Hong Kong"])
# Reading from world bank

GDP = pd.read_csv("../assets/world_bank.csv", header=4)
GDP["Country Name"] = GDP["Country Name"].replace(["Korea, Rep.",
                                                   "Iran, Islamic Rep.",
                                                   "Hong Kong SAR, China"],
                                                  ["South Korea",
                                                   "Iran",
                                                   "Hong Kong"])
print(len(GDP))

# Reading from ScimEn

ScimEn = pd.read_excel("assets/scimagojr-3.xlsx", index_col=4)
ScimEn.reset_index(inplace=True)
ScimEn.rename(columns={"Country": "Country Name"}, inplace=True)
print(len(ScimEn))

# Merging tables

# firstMerge = pd.merge(Energy, GDP[
#     ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "Country Name"]],
#                       how="inner", on="Country Name")
firstMerge = pd.merge(ScimEn[["Rank", "Documents", "Citable documents", "Citations", "Self-citations",
                              "Citations per document", "H index", "Country Name"]], Energy, how="inner",
                      on="Country Name")
df = pd.merge(firstMerge, GDP[
    ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "Country Name"]], how="inner",
              on="Country Name")
df.set_index("Country Name", inplace=True)
df.sort_values("Rank", inplace=True)

# After this
Dataframe = df[0:15]

col = Dataframe.loc[:, "2006":"2015"]
Dataframe["mean"] = col.mean(axis=1, skipna=True)
gdpAvg = Dataframe["mean"].sort_values(ascending=False)[0:15]
country = gdpAvg.index[5]
firstValue = Dataframe["2015"][country]
secondValue = Dataframe["2006"][country]
difference = secondValue - firstValue

# Question 6
# maxIndex = Dataframe["% Renewable"].idxmax()
# print(Dataframe["% Renewable"].idxmax())
k = pd.to_numeric(Dataframe["% Renewable"], errors="coerce")
print(k.idxmax())
t = (k.idxmax(), df["% Renewable"][k.idxmax()])
print(t)

# Question 7
print(Dataframe.columns)
Dataframe["ratio"] = Dataframe.apply(lambda x: x["Self-citations"] / (x["Citations"]), axis=1)
country = Dataframe["ratio"].idxmax()
ratio = Dataframe["ratio"][Dataframe["ratio"].idxmax()]

# Question
# print(Dataframe[["Energy Supply per Capita", "Energy Supply"]].head())
newSeries = Dataframe.apply(lambda x: x["Energy Supply"] / (x["Energy Supply per Capita"]), axis=1)
country = newSeries.idxmax()

# Question 9
Dataframe["capita"] = Dataframe.apply(lambda x: x["Energy Supply"] / (x["Energy Supply per Capita"]), axis=1)
Dataframe["Citable docs per capita"] = Dataframe.apply(lambda x: x["Citable documents"] / (x["capita"]), axis=1)
newDf = Dataframe[["Citable docs per capita", "Energy Supply per Capita"]]
pvalue, corr = stats.pearsonr(newDf["Citable docs per capita"], (newDf["Energy Supply per Capita"]))

# Question 10
median = Dataframe["% Renewable"].median()


def getValue(row):
    if row["% Renewable"] < median:
        return 0
    else:
        return 1


Dataframe["HighRenew"] = Dataframe.apply(getValue, axis=1)

# Question 11

Dataframe["capita"] = Dataframe["capita"].apply(lambda x: "{:,}".format(x))
print(Dataframe["capita"])

#
# def getAverageGDP(group):
#     mean = np.nanmean([group["2006"] + group["2007"] + group["2008"] + group["2009"] + group["2010"] + group["2011"] +
#                        group["2012"] + group["2013"] + group["2014"] + group["2015"]])
#     return mean
#
#
# newSeries = df.apply(getAverageGDP, axis=1)
# newSeries.sort_values(ascending=False, inplace=True)
# # print(newSeries[0:20])
#
# # Mean energy supply per capita
# Dataframe["Energy Supply per Capita"] = pd.to_numeric(Dataframe["Energy Supply per Capita"], errors="coerce")
# print(Dataframe["Energy Supply per Capita"].mean())
