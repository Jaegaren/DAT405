import pandas as pd
from matplotlib import pyplot as plt


def createCorrectGDPTable(year):
    df = pd.read_csv('/Users/gille/Documents/GitHub/DAT405/files/gdp-per-capita-in-us-dollar-world-bank.csv')
    return df.loc[df["Year"] == year]


def createCorrectLifeExpectancy(year):
    df = pd.read_csv("/Users/gille/Documents/GitHub/DAT405/files/life-expectancy.csv")
    df2 = df.dropna()
    return df2.loc[df["Year"] == year]


if __name__ == '__main__':
    correct_GDP_table = createCorrectGDPTable(2020)
    correct_Life_Table = createCorrectLifeExpectancy(2020)

    correct_GDP_table.merge(correct_Life_Table, on="Code")
    correct_Life_Table.merge(correct_GDP_table, on="Code")

    print(correct_GDP_table.merge(correct_Life_Table, on="Code"),
          correct_Life_Table.merge(correct_GDP_table, on="Code"))
