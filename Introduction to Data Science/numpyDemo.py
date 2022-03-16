import numpy as np
import pandas as pd

newList = np.array([[1.4, 2, 3, 4], [5, 6, 7, 8]])
print(newList)
print(newList.ndim)  # Prints number of dimensions
print(newList.shape)  # Prints rows and columns
print(newList.dtype.name)  # Prints the type

newList = np.zeros((2, 3))
print(newList)
newList = np.ones((3, 4))
print(newList)
newList = np.random.rand(2, 3)
print(newList)
newList = np.arange(10, 50,
                    2)  # First Argument - Lower bound, Second argument - Upper Bound exclusive, Third difference for int only
print(newList)
newList = np.linspace(0, 2,
                      20)  # First Argument - Lower bound, Second argument - Upper Bound inclusive, Third number of characters wanted between
print(newList)
A = np.array([2, 3, 4, 5, 6])
B = np.array([7, 8, 5, 1, 2])
print(A + B)  # Will print sum
print(A * B)  # Will print product
print((A * B) % 5 == 0)  # Will print bool array if A*B is divisible by 5
mAArray = np.array([[1, 2, ],
                    [5, 6]])
mBArray = np.array([[9, 10],
                    [13, 14]])
print(mAArray @ mBArray)  # Produces sum of matrics
print(mAArray + mBArray)  # Produces product of matrices

rangeMatrix = np.arange(0, 56, 2).reshape(4, 7)  # Crete an array with range
print(rangeMatrix)

newMatrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(newMatrix)
print(np.array([newMatrix[0, 0], newMatrix[1, 1], newMatrix[2, 1]]))  # Forming a new array from matrix
print(np.array(newMatrix[[0, 1, 2], [0, 1, 1]]))  # Forming a new array from matrix zips
print(newMatrix > 5)
print(newMatrix[newMatrix > 5])

newMatrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(newMatrix)
print(newMatrix[:3, 1:3])

newMatrix = np.genfromtxt("../resources/winequality-red.csv", skip_header=1, delimiter=";")
print(newMatrix)
wineMatrix = newMatrix[2:5, 1:3]
print("Wine matrix", wineMatrix)
print("Wine Matrix mean", wineMatrix[:, 1].mean())
admissionMatrix = np.genfromtxt("../resources/Admission_Predict.csv", delimiter=",", skip_header=1, dtype=None,
                                names=["Serial No", "GRE Score", "TOEFL Score", "University", "SOP", "LOR", "CGPA",
                                       "Research", "Chance of Admit"])
print(admissionMatrix)
print(admissionMatrix["CGPA"][0:5])
admissionMatrix["CGPA"] = admissionMatrix["CGPA"] * 0.4
print(admissionMatrix["CGPA"][0:20])
print(len(admissionMatrix[admissionMatrix["Research"] == 1]))
print(admissionMatrix[admissionMatrix["CGPA"] > 3.9])

# print(admissionMatrix[admissionMatrix["CGPA"] > 3.9]["GRE Score"])
