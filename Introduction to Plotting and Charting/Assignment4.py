import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

teams = np.array(["MI", "CSK", "KKR", "RR", "DC", "SRH", "RCB", "PBKS"])
years = np.array(["2018", "2019", "2020", "2021"])
df = pd.DataFrame([[0.75, 1.8, 1.33, 1.0, 0.56, 1.8, 0.75, 0.75],
                   [1.8, 1.8, 0.75, 0.625, 1.8, 0.75, 0.625, 0.75],
                   [1.8, 0.75, 1.0, 0.75, 1.33, 1.0, 1.0, 0.75],
                   [1.0, 1.8, 1.0, 0.56, 2.5, 0.27, 1.8, 0.75]], index=years, columns=teams)
df.plot(kind="bar")
plt.xticks(rotation=0)
plt.legend(loc="upper left", ncol=len(df.columns))
plt.xlabel("Years")
plt.ylabel("Win/Loss%")
plt.title("IPL teams W/L ratio for last auction cycle")
plt.show()
