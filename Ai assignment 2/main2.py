import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def readHouseTable():
    df = pd.read_csv('data2/data_assignment2.csv')
    return df


def calculateAndPlotLineRegression(x, y):
    xReg = x.values.reshape(-1, 1)
    yReg = y.values.reshape(-1, 1)
    lineRegression = LinearRegression()
    lineRegression.fit(xReg, yReg)
    y_pred = lineRegression.predict(xReg)
    plt.plot(x, y_pred, color="red")


# def plotTwoVariables(x, y):


if __name__ == '__main__':
    houseTable = readHouseTable()
    x = houseTable.iloc[:, 1]  # ID index
    y = houseTable.iloc[:, 6]  # Selling Price index
    index = houseTable.loc[:, "ID"]

    plt.xlabel("Living Area")
    plt.ylabel("Selling price")

    plt.scatter(x, y)
    calculateAndPlotLineRegression(x, y)
    plt.show()
