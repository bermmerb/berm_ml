import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar
from cycler import cycler
import matplotlib as mpl

df = pd.read_csv('data/weight-height.csv')

dfpvt = df.pivot(columns = 'Gender', values = 'Weight')

dfpvt.head()
dfpvt.info()

dfpvt.plot(kind='box')
plt.title('Weight Box Plot')
plt.ylabel("Weight (lbs)")

plt.show()