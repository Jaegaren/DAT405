import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from collections import Counter


def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    df.to_numpy()
    return df


def createPandaXYValues(table, colum1, colum2):
    return table[colum1], table[colum2]


def createPandaNPArray(dataFrame, column1, column2):
    return dataFrame[[column1, column2]].to_numpy()


def createDBSCAN(epsilon, minsamples, dataFramed):
    return DBSCAN(eps=epsilon, min_samples=minsamples).fit(dataFramed)


def plotINIT():
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel("Phi")
    plt.ylabel("Psi")
    plt.title("PRO")
    plt.show()
def calculateOptimizedEpsilon(dataset, neighbours):
    neighbors = NearestNeighbors(n_neighbors=neighbours)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    # plt.plot(distances)


def plot_outliers(data):
    data['acids'] = list(dbscan.labels_)
    outliers = data[data['acids'] == -1]
    outliers['residue name'].value_counts().sort_values().plot.bar()



if __name__ == '__main__':
    X = readData()
    X = X.loc[X["residue name"] == "PRO"]
    X = createPandaNPArray(X, "phi", "psi")

    dbscan = createDBSCAN(13, 110, X)
    snsPlot = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue= dbscan.labels_, legend="full", palette="deep")
    sns.move_legend(snsPlot, "upper right", bbox_to_anchor=(1.13, 1.15), title='Clusters')
    plt.show()
