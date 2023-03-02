'''This is a code for the moving average of multi-domain hydrogel swelling'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

data = pd.read_csv('multigel_swell_2.csv')
t = data['Time']
r = data['Red']
g = data['Green']
b = data['Blue']

x = np.array(t)
y0 = np.array(r)
y1 = np.array(g)
y2 = np.array(b)

def moving_average(x, y):
    return np.convolve(x, np.ones(y), 'valid') / y

avg_y0 = moving_average(y0, 3)
avg_y1 = moving_average(y1, 3)
avg_y2 = moving_average(y2, 3)

avg_file = open('multidomain_swelling_avg.csv', 'w', newline='')
with avg_file:
    write = csv.writer(avg_file) 
    write.writerows([t])
    write.writerows([avg_y0])
    write.writerows([avg_y1])
    write.writerows([avg_y2])