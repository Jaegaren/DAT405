import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    df.to_numpy()
    return df

def createPandaXYValues(table, colum1, colum2):
    return table[colum1], table[colum2]

def createPandaNPArray(dataFrame, column1, column2):
    return dataFrame[[column1, column2]].to_numpy()


def createDBSCAN(dataFramed):
    clustering = DBSCAN(eps=3, min_samples=3).fit(dataFramed)
    return


def calculateOptimizedEpsilon(dataset, neighbours):
    neighbors = NearestNeighbors(n_neighbors=neighbours)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)

if __name__ == '__main__':
    X = createPandaNPArray(readData(), "phi", "psi")
    calculateOptimizedEpsilon(X, 500)
    plt.ylim(0, 4)
    m = DBSCAN(eps=3.88, min_samples=10)
    m.fit(X)
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod',
              'lightcyan', 'navy']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    clusters = m.labels_

    #plt.scatter(X[:, 0], X[:, 1], c=vectorizer(clusters))
    plt.show()

    # 1.3 for epsilon
    '''
    plt.scatter(x, y)
    plt.xlim(-180, 180)
    

    '''