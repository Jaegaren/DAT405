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
    for index, row in countryIndex.items():
        plt.scatter(x[index], y[index], color="blue", s=10)
    giveScatterPlotNames()


def scatterPlotCountriesWithHighLivingExp():
    sigmaAboveMedian = calculateSigmaAboveMedian()
    for index, col in countryIndex.items():
        if x.at[index] > sigmaAboveMedian:
            plt.scatter(x[index], y[index], color="green", s=10)
    giveScatterPlotNames()


def giveScatterPlotNames():
    for index, genericTable in countryIndex.items():
        plt.annotate(countryIndex[index], (x[index], y[index]), ha="center")


def calculateSigmaAboveMedian():
    return np.median(x) + np.std(x)



if __name__ == '__main__':
    correct_GDP_table = createCorrectGDPTable(2020)
    correct_Life_Table = createCorrectLifeExpectancy(2020)

    x = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Life expectancy at birth (historical)"]
    y = correct_GDP_table.merge(correct_Life_Table, on="Entity")["GDP per capita (constant 2015 US$)"]
    countryIndex = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Entity"]
    scatterPlotCountriesWithHighLivingExp()
    plt.yscale("log")
    plt.xlabel('Life Expectancy')
    plt.ylabel('GDP per capita')
    plt.show()
