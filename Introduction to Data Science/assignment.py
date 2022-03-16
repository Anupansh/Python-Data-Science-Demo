import re


# Practice assignment number 2

def grades():
    with open("../resources/grades.txt", "r") as file:
        grades = file.read()
        gradesList = []
        for value in re.finditer("(?P<name>[\w\s]*:)(?P<grade>\s[ABC])", grades):
            if value.groupdict()["grade"].__contains__("B"):
                name = value.groupdict()["name"].replace(":", "").strip()
                print(name)
                gradesList.append(name)
        return gradesList


gradeName = grades()
# for name in gradeName:
#     print("Name" ,name)


# print("asdasd")
# with open("logdata.txt", "r") as file:
#     logsdata = file.read()
#     list = []
#     regex = """
#     (?P<host>[\d\.]*)
#     (\s-\s)
#     (?P<user_name>[\w\d-]*)
#     (\s\[)
#     (?P<time>.*)
#     (\]\s\")
#     (?P<request>.*)
#     \"
#     """
#     for item in re.finditer(regex, logsdata, re.VERBOSE):
#         print(item.groupdict())
#         list.append(item.groupdict())
