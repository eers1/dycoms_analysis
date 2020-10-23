#!/usr/bin/env python2.7 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#rc('text', usetex=True) 
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10}) 

keys = ['base', 'ppe', 'thin', 'kparam', 'thick'] # 'oat','extra','val']
num = [1, 20, 5, 5, 5] # 4,6,8]

fig,ax = plt.subplots()
ax.plot([6,6,6],np.linspace(0,190,3),linestyle=':',color='grey',alpha=0.5) 
ax.fill_betweenx(np.linspace(0,190,10),6,8.05,color='grey',alpha=0.2)

lines=[]
for n, key in enumerate(keys):
    if key=='base':
        col = 'lime'
    elif key=='kparam':
        col = (238/255, 27/255, 155/255) 
    elif key=='thick':
        col = (255/255, 211/255, 29/255)
    elif key=='thin':
        col = (26/255, 224/255, 203/255)
    else:
        col = 'grey'

    for i in range(1,num[n]+1):
        series = np.loadtxt('./timeseries/{}{}_timeseries.csv'.format(key,i), delimiter=',')
        times = np.loadtxt('./timeseries/{}{}_times.csv'.format(key,i),delimiter=',')
        line, = ax.plot(times,series,color=col,alpha=0.3)
        lines.append(line)

ax.set_title('LWP Timeseries')
ax.set_xlabel('Time (h)')
ax.set_ylabel('Liquid Water Path ($g\; m^{-2}$)')
ax.legend((lines[0], lines[1],lines[22],lines[26],lines[31]),('DYCOMS-II RF01','PPE Simulations','','',''),loc='upper left',fontsize=10)
plt.show()
