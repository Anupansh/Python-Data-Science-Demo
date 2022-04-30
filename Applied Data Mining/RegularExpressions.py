import re

text7 = '@UN @UN_Women "Ethics are built right into the ideals and objectives of the United Nations" \
#UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text8 = text7.split(' ')

# @ will be the starting points. + below shows one or more occurences of character in brackets
print([w for w in text8 if re.search('@[A-Za-z0-9_]+', w)])

# . - Dot matches a single character
# ^ - Carat indicates the start of a string
# $ - Dollar End of the string
# [] - Match one of the character within the bracket like [a-z]
# [^abc] - Matches a character that is not a,b or c with carat at the beginning
#  a|b - Matches either a or b where a and b can be strings
# () - Scoping of operators
# \ - Escape for special characters like \t,\n
# \b - Match a word boundary
# \d - Any digit [0-9]
# \D - Any digit except [0-9]
# \s - Whitespace character
# \s - Any non- whitespace character
# \w - Alphanumeric character
# \W - Non alphanumeric
# * - Written before must match zero or more occurences
# + - One or one more time
# ? - Zero or once occurence
#  {n} - Exactly n- repetetions
#  {n,} - Atleast 3 times
# {,n} - Atmost 3 times
# {m,n} - Atleast m and max n characters

text7 = 'ouagadougou'
print(re.findall('[aeiou]',text7))
print(re.findall('[^aeiou]',text7))

import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.",
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
print(df['text'].str.len())

# find the number of tokens for each string in df['text']
print(df['text'].str.split().str.len())

# find which entries contain the word 'appointment'
print(df['text'].str.contains('appointment'))

# find how many times a digit occurs in each string
print(df['text'].str.count(r'\d'))

# find all occurences of the digits
print(df['text'].str.findall(r'\d'))

# group and find the hours and minutes . \d? indicates that there might be a digit otherise it will
# accept only one digit
print(df['text'].str.findall(r'(\d?\d):(\d\d)'))

# replace weekdays with '???'
print(df['text'].str.replace(r'\w+day\b', '???',regex=True))

# replace weekdays with 3 letter abbrevations
print(df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3],regex=True))

# create new columns from first match of extracted groups
print(df['text'].str.extract(r'(\d?\d):(\d\d)'))

# extract the entire time, the hours, the minutes, and the period
print(df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))'))

# extract the entire time, the hours, the minutes, and the period with group names
# Every captured group will return a seprate value
print(df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))'))