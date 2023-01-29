import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def readHouseTable():
    df = pd.read_csv('data2/data_assignment2.csv')
    return df


def calculateAndPlotLineRegression(xValues, yValues, returnPrints):
    xReg = xValues.values.reshape(-1, 1)  # These are made to reshape
    yReg = yValues.values.reshape(-1, 1)  # the data for LinearRegression()
    lineRegression = LinearRegression()
    lineRegression.fit(xReg, yReg)
    y_pred = lineRegression.predict(xReg)  # Prediction
    plt.plot(xValues, y_pred, color="red")
    if returnPrints:
        returnEquation(lineRegression)
        calculateCoefficientOfDetermination(yReg, y_pred)


def returnEquation(line):
    print("The line intercept is %.2f" % line.intercept_)
    print("The line coefficient is %.2f" % line.coef_)


def calculateCoefficientOfDetermination(realValues, predictedValue):
    print("The mean squared error intercept is %.2f" % mean_squared_error(realValues, predictedValue))
    print("The rSquared error is %.2f" % r2_score(realValues, predictedValue))


def annotateHousesToPandaIndex(indexTable, x, y):
    for index, genericTable in indexTable.items():
        plt.annotate(indexTable[index] - 1, (x[index], y[index]))  # minus one to account for panda 0-indexed


def dropOutliers(table, dropHouses):
    return table.drop(dropHouses)


def plotResidualGraph(table, x_col, y_col):
    sns.residplot(data=table, x=x_col, y=y_col)


if __name__ == '__main__':
    houseTable = readHouseTable()
    houseTable = dropOutliers(houseTable, [40, 45, 9, 24])
    x = houseTable.iloc[:, 1]  # Living_area index
    y = houseTable.iloc[:, 6]  # Selling_price index

    plt.xlabel("Living Area")
    plt.ylabel("Selling Price")
    plotResidualGraph(houseTable, "Living_area", "Selling_price")
    # annotateHousesToPandaIndex(houseIndex, x, y)
    # plt.scatter(x, y)
    # calculateAndPlotLineRegression(x, y, True)
    plt.show()
