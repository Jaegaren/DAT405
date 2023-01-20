import pandas as pd
from matplotlib import pyplot as plt

def createCorrectTable():
    df = pd.read_csv('/Users/gille/Documents/GitHub/DAT405/files/gdp-per-capita-in-us-dollar-world-bank.csv')
    df.loc(df["Year"] == 2020)
    print(df.to_string())


if __name__ == '__main__':
    createCorrectTable()

