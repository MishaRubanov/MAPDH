'''Bar graph generated from manually calculated hydrogel lengths - patterned and collected'''

#Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Data = (185, 189)
Error = (9, 14)
x = ['Patterned', 'Collected']
x_pos = np.arange(len(x))

#Customize graph
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(x_pos, Data, yerr=Error, width=0.2, align='center',
       color='tab:blue', ecolor='black', capsize=10)
fs = 15
ax.set_xticklabels(x)
ax.set_ylabel('Length (μm)', fontsize=fs - 3, weight='bold')
plt.xticks(x_pos, fontsize=fs - 3, weight='bold')
plt.yticks(fontsize=fs - 3, weight='bold')
plt.ylim(top=300)
plt.savefig('bargraph.svg')
