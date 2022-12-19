import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


df = pd.read_csv('data/d4q10m.csv', names=['weights', 'res'], delimiter=";",nrows=10000000)
res = df['res'].to_numpy()


# fixed bin size
bins = np.arange(0, 1, 0.01) # fixed bin size
plt.xlim([min(res), max(res)])
plt.hist(res, bins=bins, alpha=0.5)
plt.show()