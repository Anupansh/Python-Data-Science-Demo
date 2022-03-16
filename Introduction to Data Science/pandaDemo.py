import pandas
import pandas as pd
import numpy as np

intArray = [1, 2, 4, 6]
print(pd.Series(intArray))
dict = {'Ani': 'Aggarwal',
        'Taani': 'Sabharwal',
        'Krishan': 'Bishpal'
        }
print(pd.Series(dict))
print(pd.Series(dict, index=['Ani', 'Taani', 'Raman']))
print(pd.Series([('Ani', 'Agarwal'), ('Himanshi', 'Khurana')]))

# Querying the pandas data

print(pd.Series(dict)[0])
print(pd.Series(dict)['Taani'])
print(pd.Series(dict).iloc[0])  # Find a value by that particular index
print(pd.Series(dict).loc['Krishan'])  # Find a value by that particular value

intList = {
    20: "Tarzan",
    30: "Avengers",
    40: "Revengers"
}
print(pd.Series(intList)[30])

grades = pd.Series([20, 40, 60, 125, 115])
total = np.sum(grades)
print(total / len(grades))

randomNumbers = pandas.Series(
    np.random.randint(0, 1000, 10000))  # Generate 10000 Random integer numbers between 0 and 1000
print(len(randomNumbers))
print(np.sum(randomNumbers) / len(randomNumbers))
print(randomNumbers.head())  # Print first five numbers of a series
print(randomNumbers.head() + 2)

for label, value in randomNumbers.iteritems():
    randomNumbers._set_value(label, value + 5)

print(randomNumbers.head())
newDict = ({
    "Alice": "Morgan"
})
newSeries = pd.Series(newDict)
print(newSeries)
newSeries["John"] = "Luther"
print(newSeries)
index = [1, 2]
newSeries.index = index
print(newSeries)

# Dataframes

data = [{"name": "Alice",
         "grade": "B+",
         "subject": "math"},
        {"name": "R0man",
         "grade": "C+",
         "subject": "english"},
        {"name": "David",
         "grade": "A+",
         "subject": "physics"}
        ]
tf = pd.DataFrame(data, index=["Student 1", "Student 2", "Student 1"])
print(tf)
print(tf["name"])  # Get a column
print(tf.loc["Student 1"])  # Get a row
print(tf.loc["Student 1"]["grade"])  # Get both
print(tf.T)  # Reverse column and row
print(tf.T.loc["grade"])  # Return all the grades
print(tf.T.loc["grade"]["Student 2"])  # Return the grade of student 2
print(tf.loc[:, ["name", "grade"]])
print(tf.drop("Student 2"))
tempDf = tf.copy()
print(tempDf.drop("name", inplace=True,
                  axis=1))  # Will delete column in original table as well with inplace set to true and axis specifying columns
tempDf["Class Ratings"] = None
print(tempDf)
tempDf._set_value("Student 2", "Class Ratings", 108.0)
print(tempDf)

# Seeing a file on Jupyter or Kernel
# !cat resources/Admission_Predict.csv

# Reading from a csv file in dataframe
tf = pandas.read_csv("../resources/Admission_Predict.csv",
                     index_col=0)  # Index column 0 will set first column as index column
print(tf.head())
print(tf.columns)
newColumn = list(tf.columns)
print(newColumn)
newColumn = [column.strip() for column in newColumn]  # Cleaning the Column names to remove extra space
print("Columns after removing space", newColumn)
tf.columns = newColumn
# print("cfygjhkjn",newColumn)
print(tf.columns)
tf.rename(columns={"LOR": "Letter of Reccomendation", "SOP": "Statement of Purpose"}, inplace=True)
print(tf.head())
print(tf.columns)
print("Dataframe Columns", tf.columns)
admit_mask = tf['Chance of Admit'] > 0.9  # First method of masking
print(tf.where(admit_mask).dropna())  # Will drop NA from array
print(tf['Chance of Admit'] > 0.9)
print(tf[(tf['Chance of Admit'].gt(0.9)) & (
    tf['TOEFL Score'].lt(118))])  # Applying multiple boolean mask to a dataframe () are mandatory
tf["Serial No."] = tf.index  # Setting serial no. as another column before removing from primary column
tf = tf.set_index('Chance of Admit')  # Setting Chance of Admit as Primary index
print(tf)
tf.reset_index(inplace=True)  # To reset original indexes with inplace = 1 and set index from 0,1,2
print(tf['GRE Score'].unique())  # Prints the unique value of particular column
tf.set_index(['GRE Score', 'CGPA'], inplace=True)
print(tf.head())
tf.sort_index(inplace=True)
print(tf.loc[324, 8.87])  # Get values for those keys set as index
print(tf.loc[[(324, 8.87), (316, 8.00)]])  # Get the values multiple keys using tuples

classGrades = pandas.read_csv("../datasets/class_grades.csv",
                              index_col=0)
print(classGrades.head(10).dropna())
classGrades.fillna(0.0, inplace=True)
print(classGrades)

logCsv = pandas.read_csv("../datasets/log.csv", index_col=0)
print(logCsv.head(10))
logCsv.sort_index(inplace=True)
print(logCsv.head(10))
logCsv.reset_index(inplace=True)
logCsv.set_index(["time", "user"], inplace=True)
print(logCsv)
print(logCsv.fillna(
    method="ffill").head())  # Used to fill the null values from the previous row. Good when df is already sorted

newDf = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [1, 1, 23, 1, 4],
    "C": ['a', 't', 'g', '4', None]
})
print(newDf.fillna(method="ffill"))
print(newDf.replace([1, 3], [10, 300]))  # Replace 1 with 10 and 3 with 300
print(logCsv.replace(to_replace=".html$", value=".webpage", regex=True))  # Replace a regex with values

presidents = pd.read_csv("../datasets/presidents.csv")
presidents["first"] = presidents["President"].replace("\s.*", "",
                                                      regex=True)  # Make a new column first and discard last name
print(presidents.head())
del (presidents["first"])
print(presidents)


# Spit the president first name and last name and apply that to columns
def split(row):
    row["firstname"] = row["President"].split(" ")[0]
    row["lastname"] = row["President"].split(" ")[-1]
    return row


presidents = presidents.apply(split, axis=1)
print(presidents.head())

del (presidents["firstname"])
del (presidents["lastname"])

# Extract firstname and lastname from Born using Regex
pattern = "(?P<firstname>[\w].*)(?:\s)(?P<lastname>[\w].*$)"
names = presidents["President"].str.extract(pattern)
presidents["Firstname"] = names["firstname"]
presidents["Lastname"] = names["lastname"]

print(presidents["Born"])

# Removing [] from Born column
presidents["Born"] = presidents["Born"].str.extract("([\w]{3} [\w]{1,2}, [\w]{4})")
print(presidents["Born"])
presidents["Born"] = pd.to_datetime(presidents["Born"])  # Setting in date time format
print(presidents["Born"])

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj1 = pd.Series(sdata)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj2 = pd.Series(sdata, index=states)
obj3 = pd.isnull(obj2)
x = obj2['California']
print(obj2['California'] == None)

s1 = pd.Series({1: 'Alice', 2: 'Jack', 3: 'Molly'})
s2 = pd.Series({'Alice': 1, 'Jack': 2, 'Molly': 3})
# print(s2.loc[1])
