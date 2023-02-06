from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    return df

def createPandaXYValues(table, colum1, colum2):
    return table[colum1], table[colum2]

def regularScatterPLot(x_values, y_values):
    plt.scatter(x_values, y_values, marker="o", s=1)


if __name__ == '__main__':
    x, y = createPandaXYValues(readData(), "phi", "psi")
    x, y = make_blobs(n_samples=10, centers=3, cluster_std=0.60, random_state=0)
    plt.scatter(x[:, 0], x[:, 1])

    # WCSS stands for within-cluster sum of squares.
    # It is a measure of the compactness of the clusters produced by a clustering algorithm.
    WCSS = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        WCSS.append(kmeans.inertia_)
        # Calculate the first point where the rate of change of WCSS begins to decrease.
        if len(WCSS) >= 2:
            rate_of_change = (WCSS[i - 1] - WCSS[i - 2]) / WCSS[i - 2]
            if rate_of_change < 0:
                best_k = i - 1
                print(best_k)
                break

    plt.plot(range(1, len(WCSS) + 1), WCSS)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    # Find the optimal k value
    k_opt = np.argmin(np.diff(WCSS)) + 1
    print(k_opt)
    print("look up")

    kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
    #kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0).fit(x)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

    plt.scatter(x[:, 0], x[:, -1])

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

    plt.title('Data points and cluster centroids')
    plt.show()

