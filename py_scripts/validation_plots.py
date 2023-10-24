import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

font = 21
inplotfont = 18
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': font})

pink = '#C2476E'

def val_plot(valpred, actualval, ax):
    trend=[]
    mean=[]
    sd=[]
    lower95=[]
    upper95=[]
    for row in valpred:
        trend.append(row[0])
        mean.append(row[1])
        sd.append(row[2])
        lower95.append(row[3])
        upper95.append(row[4])
    minX=min(actualval)
    maxX=max(actualval)
    minY=min(lower95)
    maxY=max(upper95)
    minXY=min(minY, minX)
    maxXY=max(maxY, maxX)
    seq=np.linspace(minXY,maxXY,10)
    errors=[]
    for v in range(len(upper95)):
        errors.append((upper95[v] - lower95[v])/2)
    line = ax.plot(seq,seq, c='black',label='line of equality') 
    error_lines = ax.errorbar(actualval,mean,yerr=errors,  c='black',label='prediction with\n 95\% confidence bounds',ms=S,linewidth=2,fmt='o', )
    rmse = calc_rmse(mean, actualval)
    ax.text(0.03,0.97,'RMSE = {:.3}'.format(rmse), transform=ax.transAxes, fontsize=inplotfont, horizontalalignment="left", verticalalignment="top")
    return mean, line, errors, error_lines, sd, upper95, lower95

def calc_rmse(val_pred, val_actual):
    rmse = np.mean((np.subtract(val_actual, val_pred)**2))**0.5    
    return rmse

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,7))
plt.subplots_adjust(top=0.95, left=0.1, right=0.65, wspace=0.3, hspace=0.3)
S=7
nd = 'low'
nd_ = '' if nd=='low' else f'{nd}_nd_'
noise_type = 'extras'

### LWP ###
noise_type = "max_g_048_everywhere" 
valpred = np.loadtxt(f"../predictions/lwp_cloud/pre_val_{nd_}lwp_cloud_mean_{noise_type}.csv",delimiter=",",skiprows=1)
actual = np.loadtxt(f"../data_lwp_cloud/dycoms_data_{nd}_nd_lwp_cloud_mean.csv",delimiter=",")
actualval = actual[24:32,2]
mean, line, errors, error_lines, sd, upper95, lower95 = val_plot(valpred, actualval, axes[0,0])
#print([e*100/m for e,m in zip(errors, mean)])
print(np.mean([(u95-m)*100/m for u95, m in zip(upper95, mean)]))
print(np.mean([(l95-m)*100/m for l95, m in zip(lower95, mean)]))
#print(sd)
del(valpred, actual, actualval)

### LWP Tend ###
noise_type = "none_g_all_everywhere"
valpred = np.loadtxt(f"../predictions/lwp_cloud/pre_val_{nd_}lwp_cloud_teme_{noise_type}.csv",delimiter=",",skiprows=1)
actual = np.loadtxt(f"../data_lwp_cloud/dycoms_data_{nd}_nd_lwp_cloud_teme.csv",delimiter=",")
actualval = actual[24:32,2]
mean, line, errors, error_lines, sd, upper95, lower95 = val_plot(valpred, actualval, axes[0,1])
#print([e*100/m for e,m in zip(errors, mean)])
print(np.mean([(u95-m)*100/m for u95, m in zip(upper95, mean)]))
print(np.mean([(l95-m)*100/m for l95, m in zip(lower95, mean)]))
#print(sd)
del(valpred, actual, actualval)

### CF ###
noise_type = "max_g_all_everywhere"
valpred = np.loadtxt(f"../predictions/cloud_frac/pre_val_{nd_}cloud_frac_mean_{noise_type}.csv",delimiter=",",skiprows=1)
actual = np.loadtxt(f"../data_cloud_frac/dycoms_data_{nd}_nd_cloud_frac_mean.csv",delimiter=",")
actualval = actual[24:32,2]
mean, line, errors, error_lines, sd, upper95, lower95 = val_plot(valpred, actualval, axes[1,0])
#print([e*100/m for e,m in zip(errors, mean)])
print(np.mean([(u95-m)*100/m for u95, m in zip(upper95, mean)]))
print(np.mean([(l95-m)*100/m for l95, m in zip(lower95, mean)]))
#print(sd)
del(valpred, actual, actualval)

### CF Tend ###
noise_type = "none_g_all_everywhere"
valpred = np.loadtxt(f"../predictions/cloud_frac/pre_val_{nd_}cloud_frac_teme_{noise_type}.csv",delimiter=",",skiprows=1)
actual = np.loadtxt(f"../data_cloud_frac/dycoms_data_{nd}_nd_cloud_frac_teme.csv",delimiter=",")
actualval = actual[24:32,2]

mean, line, errors, error_lines, sd, upper95, lower95 = val_plot(valpred, actualval, axes[1,1])
#print([e*100/m for e,m in zip(errors, mean)])
print(np.mean([(u95-m)*100/m for u95, m in zip(upper95, mean)]))
print(np.mean([(l95-m)*100/m for l95, m in zip(lower95, mean)]))
#print(sd)
del(valpred, actual, actualval)

### Plot additions ###
fig.text(0.37,0.03, r"MONC model output", ha='center', va='center',fontsize=font)
fig.text(0.03,0.5, r"Emulator prediction", ha='center', va='center',rotation='vertical',fontsize=font)
axes[1,1].legend(loc=(1.05,0),shadow=False,fancybox=False)

for ax,letter in zip([axes[0,0],axes[0,1],axes[1,0],axes[1,1]],['a','b','c','d']):
    ax.text(0.01,1.04,f'{letter})',transform=ax.transAxes,fontsize=font)

#fig.savefig(f'/home/rach/Emulator/paper1_plots/validation_{nd}_nd_nuggets.pdf')
#fig.savefig(f'/home/rach/Emulator/paper1_plots/validation_{nd}_nd_nuggets.png')
plt.show()
plt.close()
