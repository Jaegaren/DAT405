import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def createCorrectGDPTable(year):
    df = pd.read_csv('/Users/jacobwesterberg/PycharmProjects/DAT405/files/gdp-per-capita-in-us-dollar-world-bank.csv')
    return df.loc[df["Year"] == year]


def createCorrectLifeExpectancy(year):
    df = pd.read_csv("/Users/jacobwesterberg/PycharmProjects/DAT405/files/life-expectancy.csv")
    df2 = df.dropna()
    return df2.loc[df["Year"] == year]


def scatterPlot():
    global row
    sigmaAboveMedian = calculateSigmaAboveMedian()
    for index, col in enumerate(correct_GDP_table.merge(correct_Life_Table, on="Entity").iterrows()):
        if lifeExp[index] > sigmaAboveMedian:
            plt.scatter(x[index], y[index], color="green", s =10)
        else:
            plt.scatter(x[index], y[index], color="red", s=10)
    giveScatterPlotName()


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

def calculateSigmaAboveMedian():
    stdLifeExpectancy = np.std(x)
    meanLifeExpectancy = np.mean(x)
    sigmaAboveMedian = meanLifeExpectancy + stdLifeExpectancy
    return sigmaAboveMedian


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
