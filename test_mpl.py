import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = np.random.normal(0, 0.1, 1000)
data2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
data3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
data4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))

data = np.vstack([data1, data2, data3, data4]).transpose()

df = pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'data4'])
print(df.head())

# df.plot()
# plt.title('Line plot')
df.plot(style='.')
plt.title('Scatter plot')


plt.show()