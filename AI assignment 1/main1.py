import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def createCorrectGDPTable(year):
    df = pd.read_csv('data1/gdp-per-capita-in-us-dollar-world-bank.csv')
    return df.loc[df["Year"] == year]


def createCorrectLifeExpectancy(year):
    df = pd.read_csv('data1/life-expectancy.csv')
    return df.loc[df["Year"] == year]


def scatterPlot():
    for index, row in countryIndex.items():
        plt.scatter(x[index], y[index], color="blue", s=10)
    giveScatterPlotNames()


def scatterPlotCountriesWithHighLivingExp():
    sigmaAboveMean = calculateSigmaAboveMean()
    for index, col in countryIndex.items():
        if x.at[index] > sigmaAboveMean:
            plt.scatter(x[index], y[index], color="green", s=10)
            print(countryIndex[index])
    giveScatterPlotNames()


def giveScatterPlotNames():
    for index, genericTable in countryIndex.items():
        plt.annotate(countryIndex[index], (x[index], y[index]), ha="center")


def calculateSigmaAboveMean():
    return np.mean(x) + np.std(x)



if __name__ == '__main__':
    correct_GDP_table = createCorrectGDPTable(2020)
    correct_Life_Table = createCorrectLifeExpectancy(2020)

    x = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Life expectancy at birth (historical)"]
    y = correct_GDP_table.merge(correct_Life_Table, on="Entity")["GDP per capita (constant 2015 US$)"]
    countryIndex = correct_Life_Table.merge(correct_GDP_table, on="Entity")["Entity"]
    scatterPlot()
    plt.yscale("log")
    plt.xlabel('Life Expectancy')
    plt.ylabel('GDP per capita')
    print(countryIndex)
    plt.show()
