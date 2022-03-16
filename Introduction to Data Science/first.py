# print('Hello from Windows PC')
# x = input("Name the value you want to input ?")
# print("Value of x", x)
# print("Type of x", type(x))
# x = input("Enter the value")
# iVal = 12.03
# print("Float Value" , float(iVal))
# print("Int Value" , int(iVal))
# print("String Value" , str("ASDASDASDAS"))


# print("ASDASDASD")
# score = input("Enter Score: ")
# iScore = float(score)
# if iScore >= 0.0 and iScore <= 1.0:
#     if iScore > 0.9:
#         print("A")
#     elif iScore > 0.8:
#         print("B")
#     elif iScore > 0.7:
#         print("C")
#     elif iScore > 0.6:
#         print("D")
#     else:
#         print("F")
# else:
#     print("Invalid Score")
# from typing import Any
#
#
def computepay(h, r):
    if h <= 40:
        return h * r
    else:
        return 40 * r + (h - 40) * (1.5 * r)


print('{} adasd {}'.format(12, 26))
hrs = input("Enter Hours:")
h = float(hrs)
rate = input("Enter rate")
r = float(rate)
p = computepay(h, r)

print("Pay", p)

# try:
#     iVal = int(x)
# except:
#     try:
#         iVal = str(x)
#     except:
#         iVal = -1
# print("Value of iVal is" ,iVal)

isDone = False
largest = None
smallest = None

while True:
    x = input("Enter the value")
    if x == "done":
        break
    floatX = None
    try:
        floatX = int(x)
    except:
        print("Invalid Input")
        continue
    if largest == None:
        largest = floatX
        smallest = floatX
    else:
        if floatX > largest:
            largest = floatX
        elif floatX < smallest:
            smallest = floatX
print("Maximum is", largest)
print("Minimum is", smallest)
