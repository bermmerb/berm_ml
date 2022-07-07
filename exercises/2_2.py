import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar
from cycler import cycler
import matplotlib as mpl

df = pd.read_csv('data/weight-height.csv')

print(df.info())
print(df.head())

fig, ax = plt.subplots()
male = df[df['Gender']=='Male']
female = df[df['Gender']=='Female']
male.plot(ax=ax, x='Weight', y='Height', kind='scatter', c='red', alpha=0.3)
female.plot(ax=ax, x='Weight', y='Height', kind='scatter', c='blue', alpha=0.3)

plt.legend(["Males", "Females"])

plt.show()