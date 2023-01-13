import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


df = pd.read_csv('distcont_6q_2M.csv', names=['weights', 'res'], delimiter=";")
res = df['res'].to_numpy()


# fixed bin size
bins = np.arange(0, 1, 0.01) # fixed bin size
plt.xlim([min(res), max(res)])
plt.hist(res, bins=bins, alpha=0.5)
plt.show()