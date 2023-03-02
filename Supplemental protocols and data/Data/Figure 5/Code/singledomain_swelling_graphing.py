'''Graphing single-domain swelling profile with python'''

#Import packages
import pandas as pd
import matplotlib.pyplot as plt

#Import data
data = pd.read_csv('singledomain_swelling_avg.csv')
x = data['Time']
yavg = data['Average']
ymin = data['Min']
ymax = data['Max']


# Plotting the data
plt.figure()
plt.plot(x, yavg, color='tab:red', linestyle='-', linewidth=3)
plt.plot(x, ymin, color='red', linestyle='-.', linewidth=1)
plt.plot(x, ymax, color='red', linestyle='-.', linewidth=1)

# Customize graph
fs = 15
plt.xlabel('Time [hr]', fontsize=fs, weight='bold')
plt.ylabel('ΔL/L0', fontsize=fs, weight='bold')
plt.xticks(fontsize=fs - 3, weight='bold')
plt.yticks(fontsize=fs - 3, weight='bold')
plt.ylim(bottom=-0.05, top=1.0)
plt.xlim(left = -2.5, right=62.5)
fig = plt.gcf()
fig.savefig('singlegel_swelling.svg')
