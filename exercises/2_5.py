import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar
from cycler import cycler
import matplotlib as mpl
from pandas.plotting import scatter_matrix

df = pd.read_csv('data/titanic-train.csv')

scatter_matrix(df.drop('PassengerId', axis=1), figsize=(10, 10))

plt.show()