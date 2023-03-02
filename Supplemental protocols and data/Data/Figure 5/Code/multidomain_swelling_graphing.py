'''Graphing multi-domain swelling profile with python'''

#Import packages
import pandas as pd
import matplotlib.pyplot as plt

#Import data
data = pd.read_csv('multidomain_swelling_avg.csv')
x = data['Time']
r = data['Red']
g = data['Green']
b = data['Blue']


# Plotting the data
plt.figure()
plt.plot(x, r, color='tab:red', linestyle='-', linewidth=3)
plt.plot(x, g, color='tab:green', linestyle='-.', linewidth=1)
plt.plot(x, b, color='tab:blue', linestyle='-.', linewidth=1)


# Customize graph
fs = 15
plt.xlabel('Time [hr]', fontsize=fs, weight='bold')
plt.ylabel('ΔL/L0', fontsize=fs, weight='bold')
plt.xticks(fontsize=fs - 3, weight='bold')
plt.yticks(fontsize=fs - 3, weight='bold')
plt.legend(['Red Domain', 'Green domain', 'Blue domain'],
           loc='upper left', fontsize=11)
plt.ylim(bottom=-0.05, top=1.0)
plt.xlim(left = -2.5, right=62.5)
fig = plt.gcf()
fig.savefig('multigel_swelling.svg')
