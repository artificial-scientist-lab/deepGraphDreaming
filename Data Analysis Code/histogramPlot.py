import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Plots the histogram of the dataset. (* TO DO *) Configure for yaml source files.

dataset = 'distcont_4q_20M'
df = pd.read_csv(f'{dataset}.csv', names=['weights', 'res'], delimiter=";")
res = df['res'].to_numpy()
print(len(res))

# fixed bin size
bins = np.arange(0, 1, 0.01) # fixed bin size
plt.xlim([min(res), max(res)])
plt.hist(res, bins=bins, alpha=0.5)
plt.yscale('log')
plt.savefig(f"{dataset}_histogram")
print(f"Highest Fidelity:{np.max(res)}")