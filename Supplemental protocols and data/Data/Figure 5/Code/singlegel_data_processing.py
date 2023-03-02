'''This is a code for the moving average of multi-domain hydrogel swelling'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

data = pd.read_csv('singlegel_swell.csv')
t = data['Time']
avg = data['Avg']
mini = data['Min']
maxi = data['Max']

x = np.array(t)
y0 = np.array(avg)
y1 = np.array(mini)
y2 = np.array(maxi)

def moving_average(x, y):
    return np.convolve(x, np.ones(y), 'valid') / y

avg_ = moving_average(y0, 10)
avg_ = moving_average(y1, 10)
avg_y2 = moving_average(y2, 10)

avg_file = open('singledomain_swelling_avg.csv', 'w', newline='')
with avg_file:
    write = csv.writer(avg_file) 
    write.writerows([t])
    write.writerows([avg_y0])
    write.writerows([avg_y1])
    write.writerows([avg_y2])