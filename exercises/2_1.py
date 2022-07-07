import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar
from cycler import cycler
import matplotlib as mpl

df = pd.read_csv('data/international-airline-passengers.csv')

print(df.info())
print(df.head())

df['year'] = pd.to_datetime(df['Month']).dt.year
df['month'] = pd.to_datetime(df['Month']).dt.month
df = pd.DataFrame().assign(year=df['year'], 
                           month=df['month'].apply(lambda x: calendar.month_abbr[x]), 
                           passenger=df['Thousand Passengers'])

df2 = df.pivot_table(index='year',
                     columns='month',
                     values='passenger')

mpl.rcParams["axes.prop_cycle"] = cycler('color', plt.get_cmap('tab20').colors)
plt.plot(df2)
plt.legend(df['month'].unique())
plt.title('Passenger by month')

plt.show()