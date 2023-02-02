import pandas as pd
from matplotlib import pyplot as plt


def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    return df


def create2DHistogram(x_values, y_values):
    plt.hist2d(x_values, y_values, bins=250, alpha=1, cmap='twilight')


def createPandaXYValues(table, colum1, colum2):
    return table[colum1], table[colum2]


def regularScatterPLot(x_values, y_values):
    plt.scatter(x_values, y_values)


if __name__ == '__main__':
    x, y = createPandaXYValues(readData(), "phi", "psi")
    create2DHistogram(x, y)
    # regularScatterPLot(x, y)
    plt.ylabel("Psi")
    plt.xlabel("Phi")
    plt.show()