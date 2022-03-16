import pandas as pd
import scipy.stats as stats

# First Answer
data = pd.read_csv("../assets/NISPUF17.csv", index_col=0)
motherEducation = data["EDUC1"].value_counts()
dict = {
    "less than high school": motherEducation[1] / motherEducation.sum(),
    "high school": motherEducation[2] / motherEducation.sum(),
    "more than high school but not college": motherEducation[3] / motherEducation.sum(),
    "college": motherEducation[4] / motherEducation.sum()
}
cbf_01 = data[["P_NUMVRC", "HAD_CPOX", "SEX"]]
cbf_01 = cbf_01.dropna()

# Third Answer
dict = {}
maleData = cbf_01[cbf_01["SEX"] == 1]
maleGotChickenPox = maleData[(maleData["HAD_CPOX"] == 1) & (maleData["P_NUMVRC"] >= 1)]
maleNotGotChickenPox = maleData[(maleData["HAD_CPOX"] == 2) & (maleData["P_NUMVRC"] >= 1)]
dict["male"] = len(maleGotChickenPox) / len(maleNotGotChickenPox)
femaleData = cbf_01[cbf_01["SEX"] == 2]
femaleGotChickenPox = femaleData[(femaleData["HAD_CPOX"] == 1) & (femaleData["P_NUMVRC"] >= 1)]
femaleNotGotChickenPox = femaleData[(femaleData["HAD_CPOX"] == 2) & (femaleData["P_NUMVRC"] >= 1)]
dict["female"] = len(femaleGotChickenPox) / len(femaleNotGotChickenPox)

# Second Answer

vdf = data[["CBF_01", "P_NUMFLU"]]
vdf = vdf.dropna()
forPositiveFeed = vdf[vdf["CBF_01"] == 1]
forNegativeFeed = vdf[vdf["CBF_01"] == 2]
firstParameter = forPositiveFeed["P_NUMFLU"].sum() / len(forPositiveFeed)
secondParameter = forNegativeFeed["P_NUMFLU"].sum() / len(forNegativeFeed)

# Fourth answer
relation = data[["HAD_CPOX", "P_NUMVRC"]]
relation = relation.dropna()
print(relation["HAD_CPOX"].value_counts())
relation = relation[(relation["HAD_CPOX"] == 1) | (relation["HAD_CPOX"] == 2)]
pval, corr = stats.pearsonr(relation["HAD_CPOX"], (relation["P_NUMVRC"]))
print(corr)
print(pval)

# 0.0034753171515295557
# 0.6669755758983295
