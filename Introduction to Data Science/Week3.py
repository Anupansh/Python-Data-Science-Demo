import pandas as pd
import numpy as np

# Merging Vertically

staff_df = pd.DataFrame(({"Name": "Kelly", "Role": "PM", "Location": "U.P."},
                         {"Name": "Sally", "Role": "Developer", "Location": "Delhi"},
                         {"Name": "Mike", "Role": "Director", "Location": "Noida"}))
student_df = pd.DataFrame(({"Name": "Ron", "School": "Xaviers", "Location": "Aligarh"},
                           {"Name": "Sally", "School": "Magneto", "Location": "Punjab"},
                           {"Name": "Mike", "School": "Ironman", "Location": "CG"}))
mergedDf = pd.merge(staff_df, student_df, how="outer", on="Name")

# Merging horizontally

df2010 = pd.read_csv("datasets/college_scorecard/MERGED2010_11_PP.csv")
df2011 = pd.read_csv("datasets/college_scorecard/MERGED2011_12_PP.csv")
df2012 = pd.read_csv("datasets/college_scorecard/MERGED2012_13_PP.csv")
newPd = pd.concat([df2010, df2011, df2012],
                  keys=["2010", "2011", "2012"])  # To add an extra parameter to indicate which data is what

# Idioms

df = pd.read_csv("../datasets/census.csv")
df = (df.where(df["SUMLEV"] == 50)  # Setting multiple conditions
      .dropna()
      .set_index(["STNAME", "CTYNAME"]))
print(df.head())


# Apply function use a function as a parameter and a list to iterate over it and return a new df with mapped values

def min_max(row):
    data = row[["POPESTIMATE2010",
                "POPESTIMATE2011",
                "POPESTIMATE2012",
                "POPESTIMATE2013",
                "POPESTIMATE2014",
                "POPESTIMATE2015"]]
    row["Max"] = np.max(data)
    row["Min"] = np.min(data)
    return row


rf = pd.read_csv("../datasets/census.csv")
rf = rf.apply(min_max, axis="columns")

# Using Lambda
rows = ["POPESTIMATE2010",
        "POPESTIMATE2011",
        "POPESTIMATE2012",
        "POPESTIMATE2013",
        "POPESTIMATE2014",
        "POPESTIMATE2015"]
df = pd.read_csv("../datasets/census.csv")

# To simultaneously set new column also

df["MAXI"] = df.apply(lambda x: np.max(x[rows]), axis=1)
print(df["MAXI"].head())

# Group by

census = pd.read_csv("../datasets/census.csv")

for value, frame in census.groupby("STNAME"):
    average = np.average(frame["CENSUS2010POP"])

listings = pd.read_csv("../datasets/listings.csv")
print(listings["review_scores_value"].head())
listings.set_index(["cancellation_policy", "review_scores_value"], inplace=True)

for group, frame in listings.groupby(level=(0, 1)):
    print(group)


def grouping_item(row):
    if row[1] == 10:
        return (row[0], 10.0)
    else:
        return (row[0], "Not 10")


for group, frame in listings.groupby(grouping_item):
    print(group)

# Aggregation - For this example aggregaation is first grouping the values according to cancellation_policy that
# finding the mean value of review_scores_value i.e. avergae for all review scores for flexible, moderate etc. Also can be used for multiple columns and np. any function which can return a single value for that particular group

listings = listings.reset_index()
print(listings.columns)
newList = listings.groupby("cancellation_policy").agg(
    {"review_scores_value": (np.nanmean, np.nanstd), "id": np.nanmean})
print(newList.head())

# Transform - When we want those values but preserve the df frame i.e. its style so it can be merged

cols = ["cancellation_policy", "review_scores_value"]
transform_df = listings[cols].groupby("cancellation_policy").transform(np.nanmean)
print("DADASDas", transform_df.head())
transform_df.rename({"review_scores_value": "mean_score_values"}, axis="columns", inplace=True)
print("DADASDas", transform_df.head())
listings = listings.merge(transform_df, left_index=True, right_index=True)
print(listings.head())

# Filtering - To discard any group which doesn't satify the condition

listings.groupby("cancellation_policy").filter(lambda x: np.nanmean(x["review_scores_value"]) > 9.5)
print(listings.head())

# Applying - Can be used to shorthand the above task like we can directly calculate the mean of a group, subtract it
# from original value and insert it in a new column . Apply can be slow from agg for large values

df = pd.read_csv("../datasets/listings.csv")
df = df[["cancellation_policy", "review_scores_value"]]


def calulate_mean_score_review(group):
    avg = np.nanmean(group["review_scores_value"])
    group["review_scores_mean"] = np.abs(avg - group["review_scores_value"])
    return group


newList = df.groupby("cancellation_policy").apply(calulate_mean_score_review)

# Pivot Tables - Adjust dataframe according to categories for any numeric values

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
new_df = pd.pivot_table(values=["D", "E"], index=["A", "B"], columns=["C"], data=df, aggfunc={np.sum, np.nanmean})
new_df.rename(columns={"nanmean": "Mean", "sum": "Addition"}, level=1, inplace=True)
# print(new_df)
# print(new_df.unstack())
# print(new_df.columns)
# print(new_df.index)
# print(new_df.loc["bar"])

# Date Time functionality - Pandas has four classes related to Timestamp, DataTimeIndex, Period and PeriodIndex

# Timestamp
print(pd.Timestamp("21/02/2022 11:50AM"))
print(pd.Timestamp(2022, 12, 15, 0, 0))
print(pd.Timestamp(2022, 12, 15, 0, 0).isoweekday())  # Prints day of week 1 for Monday 7 for Sunday
print(pd.Timestamp(2022, 12, 15, 0, 0, 23).second)

# Period - Represents a single timespan for a day and month. Arthimetic operations can be applied
print(pd.Period("01/2016") + 20)  # Adding month
print(pd.Period("01/17/2016") + 20)  # Adding day (Format - MM/dd/yyy)
t1 = pd.Series(list("abc"), [pd.Period("01/17/2026"), pd.Period("02/17/2027"), pd.Period("03/17/2030")])
print(t1)

# Converting to datetime

dateArray = ["2 June 2013", "Aug 29, 2014", "2015-06-26", "7/12/16"]

df = pd.DataFrame(np.random.randint(2, 200, (4, 2)), index=dateArray, columns=list("rt"))
df.index = pd.to_datetime(df.index)
print(df)
print(pd.to_datetime("4-7-12", dayfirst=True))  # To specify that first paramter is day

# TimeDelta
t1 = pd.Timestamp(2022, 12, 18, 9, 30, 00)
t2 = pd.Timestamp(2022, 12, 14, 23, 15, 30)
print(t1 - t2)
print(pd.Timestamp("12/06/2022 08:30AM") + pd.Timedelta("12D 6H"))

# Offset
print(pd.Timestamp("4/9/21").weekday())
print(pd.Timestamp("4/9/2021") + pd.offsets.Week())
print(pd.Timestamp("4/9/2021") + pd.offsets.MonthEnd())

# Dates in dataframe

dates = pd.date_range("10/01/16", periods=9,
                      freq="2W-SUN")  # Will print the 12 dates from 10/01/16 with frequency biweekly and on sunday
dates2 = pd.date_range("10/01/16", periods=9,
                       freq="B")  # Will print the 12 dates from 10/01/16 with frequency of business days
dates3 = pd.date_range("10/01/16", periods=9,
                       freq="QS-JUN")  # Will print the 12 dates from 10/01/16 with frequency quarterly starting from June

df = pd.DataFrame({"Column 1": 100 + np.random.randint(-20, -5, 9),
                   "Column 2": 100 + np.random.randint(-20, -5, 9)}, index=dates)
print(df)
print(df.index.weekday)  # 0 for Monday , 6 for Sunday
df.diff()  # To find difference between the date columns

print(df.resample('M').mean())  # Will reshape the dataframe to get the means grouping by month

# DateTime Indexing and Slicing
print(df.loc["2017"])  # Will print dates of 2017
print(df.loc["2016-11"])  # Will print dates of 2016 November
print(df.loc["2016-12":])  # Will print dates of december 2016 and after

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor',
                         'poor'], columns=['Grades'])
my_categories = pd.CategoricalDtype(categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                                    ordered=True)
grades = df['Grades'].astype(my_categories)
result = grades[(grades > 'B') & (grades < 'A')]
