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
male['Height'].plot(kind='hist',
                    bins=50,
                    range=(50, 80),
                    alpha=0.3,
                    color='blue')

female['Height'].plot(kind='hist',
                      bins=50,
                      range=(50, 80),
                      alpha=0.3,
                      color='red')

plt.title('Height distribution')
plt.legend(["Males", "Females"])
plt.xlabel("Heigth (in)")


plt.axvline(male['Height'].mean(), color='blue', linewidth=2)
plt.axvline(female['Height'].mean(), color='red', linewidth=2)

plt.show()