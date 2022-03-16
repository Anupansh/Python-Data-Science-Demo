import csv

file = open("../resources/demo.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
print(len(rows))
file.close()
