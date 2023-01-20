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
    correct_GDP_table.merge(correct_Life_Table, on="Entity")
    correct_Life_Table.merge(correct_GDP_table, on="Entity")

    print(correct_GDP_table.merge(correct_Life_Table, on="Entity"),
          correct_Life_Table.merge(correct_GDP_table, on="Entity"))

    for col in correct_GDP_table.merge(correct_Life_Table, on="Entity").columns:
        plt.scatter(correct_Life_Table.merge(correct_GDP_table, on="Entity")["Life expectancy at birth (historical)"],
                    correct_GDP_table.merge(correct_Life_Table, on="Entity")["GDP per capita (constant 2015 US$)"])
    plt.xlabel('GDP')
    plt.ylabel('Life Expectancy')
    plt.show()
