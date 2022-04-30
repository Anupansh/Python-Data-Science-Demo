text1 = "Ethics are built right into the ideals and objectives of the United Nations "
print(len(text1))  # The length of text1
text2 = text1.split(' ')  # Return a list of the words in text2, separating by ' '.
print(len(text2))
print([w for w in text2 if len(w) > 3])  # Words that are greater than 3 letters long in text2
print([w for w in text2 if w.istitle()])  # Capitalized words in text2
print([w for w in text2 if w.endswith('s')])  # Words in text2 that end in 's'
text3 = 'To be or not to be'
text4 = text3.split(' ')
print(len(text4))
print(set(text4))  # Return the different - different words
print(set([w.lower() for w in text4]))  # .lower converts the string to lowercase and than gives single
text5 = '"Ethics are built right into the ideals and objectives of the United Nations" \
#UNSG @ NY Society for Ethical Culture bit.ly/2guVelr   '
text6 = text5.split(' ')
print(text6)
print([w for w in text6 if w.startswith('#')])  # Starting with #
text5.strip();
text5.rstrip()  # Strip removes the character from the starting and end and rstrip only from end
text5.find('a');
text5.rfind('h')  # find Return the index character from character from start and rfind from end
text7 = 'ouagadougou'
text8 = text7.split('ou')  # Strip the text on basis of ou
print(text8)
print('ou'.join(text8))  # Joins ou on the array
print(list(text7))

# Reading files line by line

f = open('resources/dates.txt', 'r')
f.readline()  # Read line by line - Will read the first line
f.seek(0)  # Set cursor to beginning of first line
text9 = f.read()  # Reading the full file
print(len(text9))
text10 = text9.splitlines()
print(len(text10))
print(text10[0])  # First line
text5.rstrip()  # Remove all last spaces
