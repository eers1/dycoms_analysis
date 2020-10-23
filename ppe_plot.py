#!/usr/bin/env python2.7 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True) 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10}) 

keys = ['base', 'oat', 'ppe', 'extra', 'val', 'thin', 'kparam', 'thick']
num = [1, 4, 20, 6, 8, 5, 5, 5]

for n, key in enumerate(keys):
    for i in range(1,num[n]+1):
        series = np.loadtxt('./timeseries/{}{}_timeseries.csv'.format(key,i), delimiter=',')
        print(series[-1])
