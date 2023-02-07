import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def readData():
    df = pd.read_csv('data3/data_assignment3.csv', usecols=['phi', 'psi'])
    return df

def normalizeData(table):
    table[['Nphi', 'Npsi']] = StandardScaler().fit_transform(table[['phi', 'psi']])
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


def elbow_method(df):
    #Creating empty list to store Within-Cluster Sum of Squares (WCSS) values
    WCSS = []

    #Creating for loop that runs kmeans with different numbers of clusters 1-11
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(df)
        #Appending the inertia value to the WCSS list
        WCSS.append(kmeans.inertia_)

    #Plotting the elbow graph
    plt.plot(range(1, 11), WCSS)
    plt.title('The Elbow Method for Determining the Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

if __name__ == '__main__':
    x, y = createPandaXYValues(readData(), "phi", "psi")
    # print(readData())
    # print(readData().describe())
    # print(normalizeData(readData()))
    create2DHistogram(x, y)
    # regularScatterPLot(x, y)
    initPlot()
    elbowMethod(readData())