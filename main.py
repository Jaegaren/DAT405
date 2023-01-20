import pandas as pd
from matplotlib import pyplot as plt


def createCorrectGDPTable(year):
    df = pd.read_csv('/Users/gille/Documents/GitHub/DAT405/files/gdp-per-capita-in-us-dollar-world-bank.csv')
    return df.loc[df["Year"] == year]


def createCorrectLifeExpectancy(year):
    df = pd.read_csv("/Users/gille/Documents/GitHub/DAT405/files/life-expectancy.csv")
    df2 = df.dropna()
    return df2.loc[df["Year"] == year]


def scatterPlot():
    global row
    for row in enumerate(correct_GDP_table.merge(correct_Life_Table, on="Entity").columns):
        plt.scatter(x, y)
        print(type(x))
    giveScatterPlotName()


def giveScatterPlotName():
    global i, row
    for i, row in enumerate(correct_GDP_table.merge(correct_Life_Table, on="Entity").iterrows()):
        plt.annotate(row[1][0], (x_values[i], y_values[i]))


def createScatterIndex():
    global x_values, i, y_values
    x_values = []
    for i in x:
        x_values.append(i)
    y_values = []
    for i in y:
        y_values.append(i)


if __name__ == '__main__':
    correct_GDP_table = createCorrectGDPTable(2020)
    correct_Life_Table = createCorrectLifeExpectancy(2020)

    x = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Life expectancy at birth (historical)"]
    y = correct_GDP_table.merge(correct_Life_Table, on="Entity")["GDP per capita (constant 2015 US$)"]

    createScatterIndex()

    scatterPlot()
#    print(y_values)
    plt.yscale("log")
    plt.xlabel('Life Expectancy')
    plt.ylabel('GDP per capita')
    plt.show()
