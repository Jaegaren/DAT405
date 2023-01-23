import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


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
        plt.scatter(x, y, color="blue", s =15)
    giveScatterPlotName()
    # create the color here, if the value is higher than the standard, give it another color


def giveScatterPlotName():
    global i, row
    for i, row in enumerate(correct_GDP_table.merge(correct_Life_Table, on="Entity").iterrows()):
        plt.annotate(row[1][0], (lifeExp[i], GDPCap[i] ), ha= "center")


def createScatterIndex():
    global lifeExp, i, GDPCap
    lifeExp = []
    for i in x:
        lifeExp.append(i)
    GDPCap = []
    for i in y:
        GDPCap.append(i)


if __name__ == '__main__':
    correct_GDP_table = createCorrectGDPTable(2020)
    correct_Life_Table = createCorrectLifeExpectancy(2020)

    x = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Life expectancy at birth (historical)"]
    y = correct_GDP_table.merge(correct_Life_Table, on="Entity")["GDP per capita (constant 2015 US$)"]

    createScatterIndex()

    scatterPlot()
    #   print(y_values)
    plt.yscale("log")
    plt.xlabel('Life Expectancy')
    plt.ylabel('GDP per capita')
    plt.show()
