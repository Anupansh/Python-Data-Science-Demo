import re

text = 'Amy is a good girl. Amy is wonderful. Amy wakes up at 9.'
if re.search("Amy", text):
    print('Amy is here')
else:
    print('Amy is not here')

print(re.split("Amy", text))
print(re.findall("Amy", text))
print(re.search(".$", text))  # To check the end
print(re.search("^Amy", text))  # To check the beginning

gradesList = "ABAAAABCACCABCAAC"
print(re.findall('A', gradesList))
print(re.findall('[AB]', gradesList))  # Set Pattern
print(re.findall('[A][B-C]', gradesList))  # A followed by B or C
print(re.findall('AB|AC', gradesList))  # A followed by B or C
print(re.findall('[^A]', gradesList))  # Does not contain A
print(re.findall('^[^A]', gradesList))  # Does not start with A
print(re.findall('A{2,10}', gradesList))  # Check for min 2 to max 10 occurrences of A
print(re.findall('A{1,1}A{1,1}', gradesList))  # Check for consecutive occurrences of A
print(re.findall('A{2}', gradesList))  # Check for consecutive occurrences of A

with open('../resources/ferpa.txt', 'r') as file:
    wiki = file.read()
print("Wiki", wiki)
print(re.findall("[a-zA-Z]{1,100}\[edit\]",
                 wiki))  # Will find all the characters starting with btw A & Z or a&z ending with edit
print(re.findall("[a-zA-Z ]{1,100}\[edit\]", wiki))
print(re.findall("[\w]{1,100}\[edit\]", wiki))  # \w for any character
print(re.findall("[\w\s]{1,100}\[edit\]", wiki))  # \s for space
print(re.findall("[\w\s]*\[edit\]", wiki))  # \s for space
for value in re.findall("[\w\s]*\[edit\]", wiki):
    print(re.split("[\[]", value)[0])  # split to split the string based on [ on the starting

for value in re.finditer("([\w\s]*)(\[edit\])", wiki):
    print("Whole item", value.groups())
    print("Only first index", value.group(1))

for value in re.finditer("(?P<title>[\w\s]*)(?P<edit_key>\[edit\])", wiki):
    print("Whole Dict", value.groupdict())
    print("Whole Dict title", value.groupdict()["title"])

for value in re.finditer("(?P<title>[\w\s]*)(?=\[edit\])", wiki):
    print("Whole Dict with no edit", value.groupdict())

with open("../resources/nytimeshealth.txt", "r") as file:
    nbt = file.read()
print("New York Times", nbt)
