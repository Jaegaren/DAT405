import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def readHouseTable():
    df = pd.read_csv('data2/data_assignment2.csv')
    return df


def calculateAndPlotLineRegression(xValues, yValues):
    xReg = xValues.values.reshape(-1, 1)  # These are made to reshape
    yReg = yValues.values.reshape(-1, 1)  # the data for LinearRegression()
    lineRegression = LinearRegression()
    lineRegression.fit(xReg, yReg)
    y_pred = lineRegression.predict(xReg)  # Prediction
    plt.plot(xValues, y_pred, color="red")


def annotateHousesToPandaIndex(indexTable, x, y):
    for index, genericTable in indexTable.items():
        plt.annotate(indexTable[index] - 1, (x[index], y[index]))  # minus one to account for panda 0-indexed


def dropOutliers(table, dropHouses):
    return table.drop(dropHouses)


if __name__ == '__main__':
    houseTable = readHouseTable()
    houseTable = dropOutliers(houseTable, [40, 45, 9, 24])
    x = houseTable.iloc[:, 1]  # Living_area index
    y = houseTable.iloc[:, 6]  # Selling_price index
    houseIndex = houseTable.loc[:, "ID"]

    plt.xlabel("Living Area")
    plt.ylabel("Selling Price")

    annotateHousesToPandaIndex(houseIndex, x, y)
    plt.scatter(x, y)
    calculateAndPlotLineRegression(x, y)
    #plt.show()
