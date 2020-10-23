#!/usr/bin/env python2.7 

import numpy as np
import csv
import ncplotlib as ncplt
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors
from collections import OrderedDict


from matplotlib import rc
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})

def ds_fix_dims(ds):
    ds = ds.rename({str(u'time_series_4500_1800.0'): 'time_coarse', str(u'time_series_75_60.0'): 'time_fine', str(u'time_series_75_1800.0'): 'time_mid'})
    ds['time_coarse']=ds.time_coarse/3600
    ds['time_mid']=ds.time_mid/3600
    ds['time_fine']=ds.time_fine/3600
    ds['x'] = ds.x.astype(float)/(35)
    ds['y'] = ds.y.astype(float)/(35)
    return ds

def surf_pre_calc(ds):
    '''
    surface precip domain mean in mm and summed over the simulation time
    '''
    surface_precip_mean = ds.surface_precip_mean*1000
    surface_precip_int = surface_precip_mean.sum()
    return surface_precip_int

def rwp_calc(ds):
    '''
    rwp domain mean in g m^(-2) and summed over the simulation time
    '''
    rwp_mean = ds.RWP_mean*1000
    rwp_int = rwp_mean.sum()
    return rwp_int

def rain(ds):
    rwp_int = rwp_calc(ds)
    sp_int = surf_pre_calc(ds)
    return [np.asscalar(rwp_int), np.asscalar(sp_int)]

def tke_calc(ds): 
    tke_mean = ds.reske_mean + ds.subke_mean
#    u_squared_mean = np.mean(np.mean(np.mean(ds.u**2,axis=1),axis=1),axis=1)
#    v_squared_mean = np.mean(np.mean(np.mean(ds.v**2,axis=1),axis=1),axis=1)
#    w_squared_mean = np.mean(np.mean(np.mean(ds.w**2,axis=1),axis=1),axis=1)
#
#    u_mean_squared = np.mean(np.mean(np.mean(ds.u,axis=1),axis=1),axis=1)**2
#    v_mean_squared = np.mean(np.mean(np.mean(ds.v,axis=1),axis=1),axis=1)**2
#    w_mean_squared = np.mean(np.mean(np.mean(ds.w,axis=1),axis=1),axis=1)**2
#
#    u_prime_squared_mean = u_squared_mean-u_mean_squared
#    v_prime_squared_mean = v_squared_mean-v_mean_squared
#    w_prime_squared_mean = w_squared_mean-w_mean_squared
#	
#    tke_calc = u_prime_squared_mean + v_prime_squared_mean + w_prime_squared_mean 
    time_mean = ds.reske_mean[ds.reske_mean.dims[0]].values
    time_calc = ds.u[ds.u.dims[0]].values
    return tke_mean, time_mean

def lwp_total(ds, hours):
    if include_spinup == True:
        lwp = ds.LWP_mean*1000
    else:
        lwp = ds.LWP_mean[fine_spin:]*1000
        hours -= 1
    
    lwp_last = lwp[last]
    lwp_mean = np.mean(lwp[-fine_ave:])
    lwp_tend = (lwp[last] - lwp[first])/hours
    lwp_teme = (np.mean(lwp[-fine_ave:]) - np.mean(lwp[:fine_ave]))/hours
    tendency = calc_tendency(lwp)
    return [lwp_last, lwp_mean, lwp_tend, lwp_teme], lwp.values, tendency, lwp[lwp.dims[0]].values

def lwp_smooth(ds, hours):
    '''
    Moving average of the lwp and tendency
    '''
    if include_spinup == True:
        lwp = ds.LWP_mean*1000
    else:
        lwp = ds.LWP_mean[fine_spin:]*1000
        hours-=1
    step = 55
    lwp_smooth = []
    lwp_tend = []
    lwp_tend_smooth = []
    i = 0
    for l in range(1, len(lwp)):
        lwp_tend.append((lwp[l] - lwp[l-1])/(lwp.time_fine[l] - lwp.time_fine[l-1]))
        if l>step-1:
            lwp_smooth.append(lwp[l-step:l].mean())
            if i>0:
                lwp_tend_smooth.append((lwp_smooth[i] - lwp_smooth[i-1])/(lwp.time_fine[l] - lwp.time_fine[l-1]))
                i+=1
    return [lwp_smooth[-1], np.mean(lwp_smooth[-55:]), lwp_tend[-1], np.mean(lwp_tend),
	    (lwp_smooth[-1]-lwp_smooth[0])/hours]


def lwp_cloud_calc(lmmr, lwp):   
    cloudy_lwp = []
    t = []
    #diff = lmmr[-1] - lmmr[0]
    for m in range(len(lmmr)):
        col_mask = layer_cloud_mask(lmmr, m)
        arr_mask = col_mask.values
        lwp_masked = lwp[m].where(arr_mask==1)
        if m==0:
            lwp_masked_first = lwp_masked*1000
        cloud_lwp_mean = lwp_masked.mean(axis=(0,1))
        cloudy_lwp.append(np.asscalar(cloud_lwp_mean.values)*1000)
        t.append(np.asscalar(lmmr[m][lmmr.dims[0]].values)/3600)

    lwp_masked = lwp_masked*1000
    lwp_last = lwp_masked.fillna(0)
    lwp_first = lwp_masked_first.fillna(0)
    lwp_diff = lwp_last - lwp_first
    return cloudy_lwp, t, lwp_masked, lwp_diff

def lwp_cloud(ds, hours):
    if include_spinup == True:
        lwp = ds.lwp
        lmmr = ds.q_cloud_liquid_mass
    else:
        lwp = ds.lwp[coarse_spin:]
        lmmr = ds.q_cloud_liquid_mass[coarse_spin:]
        hours -= 1
    
    cloudy_lwp, times, lwp_masked_last, lwp_diff = lwp_cloud_calc(lmmr, lwp)
    lwp_cloud_last = cloudy_lwp[last]
    lwp_cloud_mean = np.mean(cloudy_lwp[-coarse_ave:])
    lwp_cloud_tend = (cloudy_lwp[last]-cloudy_lwp[first])/hours
    lwp_cloud_teme = (np.mean(cloudy_lwp[-coarse_ave:])-np.mean(cloudy_lwp[:coarse_ave]))/hours
    tendency = calc_tendency(cloudy_lwp, times)
    return [lwp_cloud_last, lwp_cloud_mean, lwp_cloud_tend, lwp_cloud_teme],cloudy_lwp,tendency,times, lwp_masked_last, lwp_diff

def calc_tendency(dataarray, *times):
    tendency=[]
    if times:
        tseries = times[0]
    else:
        tseries = dataarray[dataarray.dims[0]].values/3600
        dataarray = dataarray.values
    for t_ind in range(1, len(dataarray), 1):
        c_step = dataarray[t_ind]
        p_step = dataarray[t_ind - 1]
        dx = (c_step) - (p_step)
        t = tseries[t_ind] - tseries[t_ind-1]
        if t != 0:
            tendency.append(dx/t)
        else:
            tendency.append(dx/0.01)
    return tendency


def column_cloud_fraction(lmmr):
    cloud_frac=[]
    t =[]
    for m in range(len(lmmr)):
        col_mask = layer_cloud_mask(lmmr, m)
        total = col_mask.sum(axis=(0,1))
        f = np.asscalar(total.values)/(250*250)
        cloud_frac.append(f)
        t.append(np.asscalar(lmmr[m][lmmr.dims[0]].values)/3600)
    return cloud_frac, t

def clfrac(ds, hours):
    if include_spinup == True:
        lmmr = ds.q_cloud_liquid_mass
    else:
        lmmr = ds.q_cloud_liquid_mass[coarse_spin:]
        hours -= 1

       
    cloud_frac, times = column_cloud_fraction(lmmr)
    cloud_frac_last = cloud_frac[last]
    cloud_frac_mean = np.mean(cloud_frac[-coarse_ave:])
    cloud_frac_tend = (cloud_frac[last] - cloud_frac[first])/hours
    cloud_frac_teme = (np.mean(cloud_frac[-coarse_ave:]) -
	    np.mean(cloud_frac[:coarse_ave]))/hours
    tendency = calc_tendency(cloud_frac, times)
    return [cloud_frac_last, cloud_frac_mean, cloud_frac_tend, cloud_frac_teme], cloud_frac, tendency, times

def layer_cloud_mask(dataarray, time):
    '''
    Applies mask to each timestep and sums
    '''
    for n in range(110):
        layer = dataarray[time,:,:,n]
        dataarray[time,:,:,n] = layer.where(layer.values<1e-5,1).where(layer.values>1e-5,0)
    col_sum = dataarray[time].sum(axis=2,skipna=True)
    col_mask = col_sum.where(col_sum.values<1,1)
    return col_mask

def lwp_different(dataarray):
    '''
    extract last lwp
    extract first lwp
    take the difference
    find masked array
    '''
    diff = dataarray[-1] - dataarray[0]
    

ppe_path = "/gws/nopw/j04/carisma/eers/dycoms_sim/PPE/ppe"
oat_path="/gws/nopw/j04/carisma/eers/dycoms_sim/OAT/oat"
val_path="/gws/nopw/j04/carisma/eers/dycoms_sim/VAL/val"
extra_path="/gws/nopw/j04/carisma/eers/dycoms_sim/EXTRA/extra"
base_path="/gws/nopw/j04/carisma/eers/dycoms_sim/BASE/base/base.nc"
design = np.loadtxt("/home/users/eers/designs/EmulatorInputsDesign2D.csv",
	delimiter=",", skiprows=1)
oat = np.array([[-7.5, 2], [-7.5, 20], [0, 8.5], [-9, 8.5]])
validation = np.loadtxt("/home/users/eers/designs/ValidationInputsDesign2D.csv", delimiter=",", skiprows=1)
extra_design = np.loadtxt("/home/users/eers/designs/extra_points.csv",delimiter=",")
base = np.array([[-7.5, 8.5]])

ppe_no = 20
oat_no = 4
val_no = 8
extra_no = 6
base_no = 1

od = OrderedDict()
od["ppe"] = [ppe_no, ppe_path, design]
od["oat"] = [oat_no, oat_path, oat]
od["val"] = [val_no, val_path, validation]
od["extra"] = [extra_no, extra_path, extra_design]
od["base"] = [base_no, base_path, base]

### Initialise np arrays ###
last = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))
mean = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))
tend = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))
teme = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))
rwp = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))
surface_precip = np.empty((ppe_no+oat_no+val_no+extra_no+base_no, 3))

lwp_last = np.empty((ppe_no+oat_no+val_no+extra_no+base_no,3))
lwp_mean_lasthr = np.empty((ppe_no+oat_no+val_no+extra_no+base_no,3))
lwp_tend_last = np.empty((ppe_no+oat_no+val_no+extra_no+base_no,3))
lwp_tend_ave = np.empty((ppe_no+oat_no+val_no+extra_no+base_no,3))
lwp_tend_diff = np.empty((ppe_no+oat_no+val_no+extra_no+base_no,3))

#ppe_clfrac = np.empty((ppe_no+val_no+base_no,12))
#ppe_cltime = np.empty((ppe_no+val_no+base_no,12))

arrays = [last, mean, tend, teme]
rain_arrays = [rwp, surface_precip]
testing_arrays = [lwp_last, lwp_mean_lasthr, lwp_tend_last, lwp_tend_ave, lwp_tend_diff]

### Create figures and ax for plotting ppe ###
#fig, ax = plt.subplots()
#fig = plt.figure(figsize=(3.2,5))
#ax = fig.add_subplot(111)
#ax1 = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)  #
#axes=[ax1,ax2]

a=0  #
#xlabel = r'x Direction (km)'
#ylabel = r'y Direction (km)'
#xlabel = 'Time (hours)'
#ylabel = 'Height (m)'

cmap = plt.cm.get_cmap('PiYG')
cmap.set_bad(color='darkslateblue')
'''
figt,((axt1,axt2,axt3,axt4),(axt5,axt6,axt7,axt8),(axt9,axt10,axt11,axt12)) = plt.subplots(nrows=3,ncols=4,sharex=True,sharey=True)
axest=[axt1,axt2,axt3,axt4,axt5,axt6,axt7,axt8,axt9,axt10,axt11,axt12]
figm,((axm1,axm2,axm3,axm4),(axm5,axm6,axm7,axm8),(axm9,axm10,axm11,axm12)) = plt.subplots(nrows=3,ncols=4,sharex=True,sharey=True)
axesm=[axm1,axm2,axm3,axm4,axm5,axm6,axm7,axm8,axm9,axm10,axm11,axm12]
figb,((axb1,axb2,axb3,axb4),(axb5,axb6,axb7,axb8),(axb9,axb10,axb11,axb12)) = plt.subplots(nrows=3,ncols=4,sharex=True,sharey=True)
axesb=[axb1,axb2,axb3,axb4,axb5,axb6,axb7,axb8,axb9,axb10,axb11,axb12]
axt=0
axm=0
axb=0

figt.suptitle('Top')
figm.suptitle('Middle-bottom-right')
figb.suptitle('Bottom-left')
figt.text(0.5, 0.04, xlabel, ha='center')
figt.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
figm.text(0.5, 0.04, xlabel, ha='center')
figm.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
figb.text(0.5, 0.04, xlabel, ha='center')
figb.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
'''
### Options ###
first = 0
last = -1
coarse_spin = 1 # 1 hour = 1, 2 hours = 2
fine_spin = 55 # 1 hour = 55, 2 hours = 111
coarse_ave = 3
fine_ave = 111

### Loop through PPE datasets ###
#calc = 'rain'
calc = 'lwp_total'
#calc = 'lwp_cloud'
#calc = 'lwp_smooth'
#calc = 'cloud_frac'
i=0
below_ppe=[5,6,11,12,14,18]
below_val=[5,6,8]
below_extra=[6] #[5,6]
lines=[]
include_spinup = True # See fine_spin and coarse_spin for setup

lwp_scenes=[]
lwp_ppe=[]
scene_values = [] 

for key in od:
    for j in range(od[key][0]):
        k = j+1
        nc = od[key][1] + str(k) + "/" + key + str(k) + ".nc"
        if key=='base':
    	    nc = base_path 
        ds = xr.open_dataset(nc)
        ds = ds_fix_dims(ds)
    
        #if key == 'oat':   ## for low nd
        #    hours = 7
        #else:
        #    hours = 8
        hours = 8  ## for high nd
            
        if calc == 'lwp_total':
            output_array, timeseries, tendency, times = lwp_total(ds, hours)
    	    #time_hrs = times[1:]
            time_hrs = times
        elif calc == 'lwp_cloud':
            output_array, timeseries, tendency, times, lwp_masked_last, lwp_diff = lwp_cloud(ds, hours)
    	    #time_hrs = times[1:]
            time_hrs = times
            #lwp_diff = lwp_diff.transpose() ## scene difference from start to end
            #np.savetxt("lwp_scenes_diff/lnd_lwp_scene_diff_%s%s.csv"%(key,k),lwp_diff.values, delimiter=',')
        elif calc == 'lwp_smooth':
            output_array = lwp_smooth(ds, hours)    
        elif calc == 'cloud_frac':
    	    output_array, timeseries, tendency, times = clfrac(ds, hours)
    	    #time_hrs = times[1:]
    	    time_hrs = times
        elif calc == 'rain':
    	    output_array = rain(ds)
        else:
            print('Select calc')
            break

        if calc in ['cloud_frac','lwp_cloud','lwp_total']:
    	    for b, array in enumerate(arrays):
                array[i, 0] = od[key][2][j][1]
                array[i, 1] = od[key][2][j][0]
                array[i, 2] = output_array[b]
        elif calc == 'rain':
    	    for b, array in enumerate(rain_arrays):
    	        array[i, 0] = od[key][2][j][1]
    	        array[i, 1] = od[key][2][j][0]
    	        array[i, 2] = output_array[b]
        elif calc == 'lwp_smooth':
            for b, array in enumerate(testing_arrays):
                array[i, 0] = od[key][2][j][1]
                array[i, 1] = od[key][2][j][0]
                array[i, 2] = output_array[b]

        ### plot lwp vs cloudy_lwp ###
        #plt.plot(np.linspace(0,30000,12),cloudy_lwp)
        #tot_lwp_mean.plot()
            	
        ### plot lwp ppe lines ### 
        #timeseries, time_hrs = tke_calc(ds)
	#timeseries = timeseries.where(timeseries.values < 1e10)
	   	
        '''#### three quotes here
        if key=="ppe":
	    if k in below_ppe:
    	        colour="green"
        	label="below line ppe"	
    	    else:
    	        colour="blue"
    	        label="above line ppe"
        elif key=="val":
    	    if k in below_val:
    	        colour="lime"
    	        label="below line val"
    	    else:
    	        colour="purple"
    	        label="above line val"
        elif key=='extra':
    	    if k in below_extra:
    	        colour="darkgreen"
    	        label="below line extra"
    	    else:
    	        colour="indigo"
    	        label="above line extra"
        elif key=='base':
    	    colour="black"
    	    label="base"
    	
        if i in [0,18,30]:
    	    colour="hotpink"
    	    label="outlier"
        elif i in [11]:
    	    colour="gold"
    	    label="corrupt"
    	
        if key!="oat":
    	    line, = ax.plot(time_hrs, timeseries, color=colour, label=label)
    	lines.append(line)
        toplot=False
        ### plot lwp scenes ###
    	
        if i in [18,8,0,12,16,6,36,32,27,30,25]:
    	    axes = axest
    	    fig = figt
    	    a=axt
    	    axt+=1
    	    toplot=True
        elif i in [15,14,3,7,9,2,19,33,34,35,24,26]:
    	    axes = axesm
    	    fig = figm
    	    a=axm
    	    axm+=1
    	    toplot=True
        elif i in [11,4,13,5,10,17,37,31,28,29]:
    	    axes = axesb
    	    fig = figb
    	    a=axb
    	    axb+=1
    	    toplot=True
        
	#if key=='base':
        toplot=True  #
        #fig_lwp,ax_lwp = plt.subplots()
        if toplot ==True:
    	    lwp_masked_last = lwp_masked_last.transpose()
	    lwp_masked_first = lwp_masked_first.transpose()
    	    #plot_obj = lwp_masked.plot(ax=axes[a],add_colorbar=False, cmap=cmap,vmin=0,vmax=700)
            np.savetxt("lwp_scenes_last/lnd_lwp_scene_last_values_%s%s.csv"%(key,k), lwp_masked_last.values, delimiter=',')
            np.savetxt("lwp_scenes_first/lnd_lwp_scene_first_values_%s%s.csv"%(key,k), lwp_masked_first.values, delimiter=',')
            #plot_obj.set_edgecolor('face')
            #rain_mmr = ds.rain_mmr_mean*1000
    	    #rain_mmr = rain_mmr.transpose()
            #surf_precip = ds.surface_precip[-1]*1000
    	    #surf_precip = surf_precip.transpose()
            #plot_obj = rain_mmr.plot(ax=axes[a],add_colorbar=False,cmap=cmap,vmin=0, vmax=0.007)
            #axes[a].set_xlabel(".",color=(0,0,0,0))
            #axes[a].set_ylabel(".",color=(0,0,0,0))
            #axes[a].set_title("%s %s"%(key, str(k)))
    	    #if a in [0,1,2,3,4,5,6,7]:
            #    plot_obj.axes.xaxis.set_visible(False)
            #if a in [1,2,3,5,6,7,9,10,11]:
            #    plot_obj.axes.yaxis.set_visible(False)
            #if a==0: #
        	#    plot_obj.axes.xaxis.set_visible(False) #
                #plot_obj.axes.set_title('%s %s'%(key,str(k)))
            #lwp_plot.colorbar.set_label("LWP (g m^(-2))")
    	    #fig_lwp.savefig("./lwp_plots/%s%s.png"%(key,str(k)))
    	    #plt.show()
            #plt.close(fig_lwp)
	#if key!='ppe':
        #    print('PPE lwp scenes saved')
        '''
        if key=='ppe':
            lwp_ppe.append(timeseries)
        elif key=='base':
            base=timeseries
	### three quotes here
        np.savetxt('{}{}_timeseries.csv'.format(key,str(k)),timeseries,delimiter=',')
        ds.close()
        print(i)
        i+=1
        a+=1 #
	

#label = 'Rain mmr (g kg^(-1))'
#label = 'Surface Precipitation (mm)'
#label = r'Liquid Water Path ($g\; m^{-2}$)'
#axes[0].set_title(r'Lowest')
#axes[1].set_title(r'Highest')
#fig.suptitle(r'LWP Scene Samples')
#fig.text(0.03, 0.5, ylabel, ha='center', va='center', rotation='vertical')
#fig.text(0.5, 0.03, xlabel, ha='center', va='center')
#fig.colorbar(plot_obj, ax=axes, extend='max', label=label)
#figt.colorbar(plot_obj, ax=axest, extend='max', label=label)
#figm.colorbar(plot_obj, ax=axesm, extend='max', label=label)
#figb.colorbar(plot_obj, ax=axesb, extend='max', label=label)

#ax.legend((lines[10],lines[7],lines[24],lines[20],lines[0],lines[11]), ('below line ppe', 'above line ppe', 'below line val', 'above line val','outlier','corrupt'), loc="upper left", fontsize=10)
#plt.title('TKE timeseries - high Nd')
#ax.set_ylabel('LWP g m^(-2)')
#ax.set_ylabel('TKE')
#ax.set_xlabel('Time (hrs)')
#plt.show()
#plt.savefig("highest_lowest.pdf")
#plt.show()
### Plotting lwp
#ax.set_xlabel("Timestep - 55=1hr")
#ax.set_ylabel("LWP (g m^(-2))")
#ax.set_title("LWP for entire PPE")
#ax.legend((lines[6],lines[7],lines[24],lines[25]), ['below line ppe', 'above line ppe', 'below line val', 'above line val'], loc="upper left", fontsize=10)
#ax.set_title(calc + ': green below k, blue above k')
#ax.set_ylabel(calc + ' tendency')
#ax.set_ylabel(calc)
#ax.set_xlabel('hours')
#plt.show()
'''
if calc == 'rain':
    np.savetxt("dycoms_data_%s_rwp.csv"%calc, rain_arrays[0], delimiter=",")
    np.savetxt("dycoms_data_%s_surfpre.csv"%calc, rain_arrays[1], delimiter=",")
elif calc == 'lwp_smooth':
    np.savetxt("dycoms_data_%s_lwp_last.csv"%calc, testing_arrays[0], delimiter=",")
    np.savetxt("dycoms_data_%s_lwp_mean_lasthr.csv"%calc, testing_arrays[1], delimiter=",")
    np.savetxt("dycoms_data_%s_lwp_tend_last.csv"%calc, testing_arrays[2], delimiter=",")
    np.savetxt("dycoms_data_%s_lwp_tend_ave.csv"%calc, testing_arrays[3], delimiter=",")
    np.savetxt("dycoms_data_%s_lwp_tend_diff.csv"%calc, testing_arrays[4], delimiter=",")
else:
    np.savetxt("dycoms_data_%s_last.csv"%calc, arrays[0], delimiter=",")
    np.savetxt("dycoms_data_%s_mean.csv"%calc, arrays[1], delimiter=",")
    np.savetxt("dycoms_data_%s_tend.csv"%calc, arrays[2], delimiter=",")
    np.savetxt("dycoms_data_%s_teme.csv"%calc, arrays[3], delimiter=",")
#np.savetxt("ppe_lwp_timeseries.csv", lwp_ppe, fmt='%s', delimiter=',')
'''

### Variability ###

var_path = "/gws/nopw/j04/carisma/eers/dycoms_sim/INT_VAR/"
points = {"thick":"Highest LWP","kparam":"On $\kappa$ line", "thin":"Lowest LWP"}

### Create lists ###
e_last=[]
e_mean=[]
e_tend=[]
e_teme=[]
lines=[]
i=0

fig, ax = plt.subplots(figsize=(7,6))
ax.plot([6,6,6],np.linspace(0,305,3),linestyle=':',color='grey',alpha=0.5)
ax.fill_betweenx(np.linspace(0,305,10),6,8.05,color='grey',alpha=0.2)
#for line in lwp_ppe:
#    legend_line, = line.plot(color='grey',alpha=0.3, label = 'PPE Simulations')

#base_line, = base.plot(color='lime',alpha=0.5,label='DYCOMS-II RF01')
for point in points:
    #fig,ax = lt.subplots()
    if point=='kparam':
        colour = (238/255, 27/255, 155/255)
    elif point=='thick':
        colour = (255/255, 211/255, 29/255)
    elif point=='thin':
        colour = (26/255, 224/255, 203/255)

    for i in range(5):
        name = point + str(i+1)
        nc = var_path + point + "/" + name + "/" + name + ".nc" 
        ds = xr.open_dataset(nc)

        hours = 8

        if calc == 'lwp_total':
            output_array, timeseries, tendency, times = lwp_total(ds, hours)
            time_hrs = times/3600
        elif calc == 'lwp_cloud':
            output_array, timeseries, tendency, times,lwp_masked = lwp_cloud(ds, hours)
            time_hrs = times
            np.savetxt("lwp_scenes_values_intvar_%s.csv"%(name), lwp_masked.values, delimiter=',')
        elif calc == 'cloud_frac':
            output_array, timeseries, tendency, times = clfrac(ds, hours)
            time_hrs = times
        else:
            print("Select calc")
            break

        #line, = ax.plot(time_hrs, timeseries, color=colour, label =points[point])
        #lines.append(line)
        np.savetxt('{}{}_timeseries.csv'.format(point,str(i+1)), timeseries,delimiter=',')
    
        e_last.append(output_array[0])
        e_mean.append(output_array[1])
        e_tend.append(output_array[2])
        e_teme.append(output_array[3])
    print(point + ' finished')    

output_type=e_mean
#thick_mean = "{:.2f}".format(np.mean(output_type[0:4]))
#kparam_mean = "{:.2f}".format(np.mean(output_type[5:9]))
#thin_mean = "{:.2f}".format(np.mean(output_type[10:14]))
#thick_var = "{:.2f}".format(np.var(output_type[0:4]))
#kparam_var = "{:.2f}".format(np.var(output_type[5:9]))
#thin_var ="{:.2f}".format(np.var(output_type[10:14]))
#labels = "%s, $\mu = %s,\; \sigma^{2} = %s$"

#ax.set_title('LWP Timeseries for PPE')
#ax.set_ylabel('Liquid Water Path ($g\; m^{-2}$)')
#ax.set_xlabel('Time (h)')
#ax.legend((lines[0],lines[5],lines[10],base_line,legend_line),[labels%(points['thick'],thick_mean,thick_var),labels%(points['kparam'],kparam_mean,kparam_var),labels%(points['thin'],thin_mean,thin_var),"DYCOMS-II RF01","PPE Simulations"],loc="upper left", fontsize=10,frameon=False)
#ax.set_xlim(0,8.05)
#ax.set_ylim(-5,195)
#fig.savefig('int_var_ensemble.pdf')
#fig.savefig('int_var_ensemble.png')
#plt.show()
#np.savetxt('ensemble_%s.csv'%calc, [e_last, e_mean, e_tend, e_teme], delimiter=',')
