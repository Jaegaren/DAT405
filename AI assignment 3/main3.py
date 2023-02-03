import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    return df


def normalizeData(table):
    table[['nPosition', 'Nphi', 'Npsi']] = StandardScaler().fit_transform(
        table[['position', 'phi', 'psi']])
    return table


def create2DHistogram(x_values, y_values):
    plt.hist2d(x_values, y_values, bins=350, alpha=1, cmap='plasma')


def createPandaXYValues(table, colum1, colum2):
    return table[colum1], table[colum2]


def initPlot():
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.ylabel("Psi")
    plt.xlabel("Phi")
    plt.show()


def regularScatterPLot(x_values, y_values):
    plt.scatter(x_values, y_values, marker="o", s=1)


if __name__ == '__main__':
    x, y = createPandaXYValues(readData(), "phi", "psi")
    # print(readData())
    # print(readData().describe())
    # print(normalizeData(readData()))
    create2DHistogram(x, y)
    # regularScatterPLot(x, y)
    initPlot()

