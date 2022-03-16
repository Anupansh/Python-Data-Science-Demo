import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nhl_df = pd.read_csv("../assets/nhl.csv")
cities = pd.read_html("../assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]


def nhl_correlation():
    team = cities[Big4].str.extract(
        '([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team['Metropolitan area'] = cities['Metropolitan area']
    team = pd.melt(team, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("", np.nan).replace("—",
                                                                                                               np.nan).dropna().reset_index().rename(
        columns={"value": "team"})
    team = pd.merge(team, cities, how='left', on='Metropolitan area').iloc[:, 1:4]
    team = team.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team['team'] = team['team'].str.replace('[\w.]*\ ', '')

    _df = pd.read_csv("assets/" + str.lower(Big4) + ".csv")
    _df = _df[_df['year'] == 2018]
    _df['team'] = _df['team'].str.replace(r'\*', "")
    _df = _df[['team', 'W', 'L']]

    dropList = []
    for i in range(_df.shape[0]):
        row = _df.iloc[i]
        if row['team'] == row['W'] and row['L'] == row['W']:
            dropList.append(i)
    _df = _df.drop(dropList)

    _df['team'] = _df['team'].str.replace('[\w.]* ', '')
    _df = _df.astype({'team': str, 'W': int, 'L': int})
    _df['W/L%'] = _df['W'] / (_df['W'] + _df['L'])

    merge = pd.merge(team, _df, 'outer', on='team')
    merge = merge.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    population_by_region = merge['Population']
    win_loss_by_region = merge['W/L%']

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


print(nhl_correlation())
