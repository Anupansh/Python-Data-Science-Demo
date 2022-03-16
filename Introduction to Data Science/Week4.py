import numpy as np
import pandas as pd
import re

a = np.arange(8)
b = a[4:6]
b[:] = 40
c = a[4] + a[6]

print(a)
print(b)
print(c)

s1 = pd.Series({"Mango": 20,
                "Strawberry": 15,
                "Blueberry": 18,
                "Vanilla": 30})
s2 = pd.Series({"Strawberry": 20,
                "Vanilla": 30,
                "Banana": 15,
                "Mango": 20,
                "Plain": 20})
s3 = s1 + s2
print(s3)

S = pd.Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
print(S[2])

df = pd.DataFrame([[5, 6, 20], [5, 82, 28], [71, 31, 92], [67, 37, 49]], index=["R1", "R2", "R3", "R4"],
                  columns=["a", "b", "c"])
f = lambda x: x.max() + x.min()
df_new = df.apply(f)
print(df_new)

# S['b':'e']

# s3['Mango'] >=  s1.add(s2, fill_value = 0)['Mango']

# f.index[0]

# "[B-Z](?=AAA)"

s = 'ABCAC'
# print(bool(re.match('A', s)) == True)
