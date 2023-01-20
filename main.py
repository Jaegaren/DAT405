import pandas as pd
from matplotlib import pyplot as plt

def createCorrectGDPTable(year):
    df = pd.read_csv('/Users/gille/Documents/GitHub/DAT405/files/gdp-per-capita-in-us-dollar-world-bank.csv')
    return df.loc[df["Year"] == year]

if __name__ == '__main__':
    correct_table = createCorrectGDPTable(2020)
    print(correct_table)

