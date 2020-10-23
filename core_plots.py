#!/usr/bin/env python2.7
import sys
from sys import argv
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ncplotlib as ncplt

# Open dataset
nc = sys.argv[1]   # /gws/nopw/j04/carisma
DS = xr.open_dataset(nc)

# Select specific diagnostics or take all
specific_diagnostics = 1
# Coupled?
casim = 1
socrates = 0
print("CASIM: " + str(casim))
print("SOCRATES: " + str(socrates))
# Drop spinup period?
spin_drop=True
if spin_drop==True:
    print("Dropping spinup period...")

if specific_diagnostics == 1:
    if casim == 1:
        names_1D = ['VWP_mean', 'LWP_mean', 'w_max'] #, 'reske_mean', 'subke_mean', 'senhf_mean', 'lathf_mean', 'RWP_mean']  #, 'surface_precip_mean']
        names_2D = ['theta_mean', 'vapour_mmr_mean', 'liquid_mmr_mean', 'rh_mean',
		'total_cloud_fraction', 'liquid_cloud_fraction', 'u_wind_mean', 'v_wind_mean',
		'w_wind_mean'] #, 'uu_mean', 'vv_mean', 'ww_mean']  # , 'rain_mmr_mean', 'subgrid_turbulent_transport' 'resolved_turbulent_transport', 'tke_tendency',]
	if socrates == 1:
            names_3D = ['surface_precip', 'vwp', 'lwp', 'rwp', 'reske', 'subke', 'cltop', 'clbas', 'toa_up_longwave', 'toa_down_shortwave', 'toa_up_shortwave']  
        elif socrates == 0:
            names_3D = ['vwp', 'lwp', 'cltop', 'clbas']   #'reske', 'subke',   # , 'surface_precip', 'rwp'] 
    elif casim == 0:
        names_1D = ['VWP_mean', 'LWP_mean', 'reske_mean', 'subke_mean', 'w_max', 'senhf_mean', 'lathf_mean']
        names_2D = ['theta_mean', 'vapour_mmr_mean', 'liquid_mmr_mean', 'rh_mean', 'subgrid_turbulent_transport', 'total_cloud_fraction', 'liquid_cloud_fraction', 'resolved_turbulent_transport', 'tke_tendency','u_wind_mean', 'v_wind_mean', 'w_wind_mean', 'uu_mean', 'vv_mean', 'ww_mean']
        if socrates == 1:
            names_3D = ['vwp', 'lwp', 'reske', 'subke', 'cltop', 'clbas', 'toa_up_longwave', 'toa_down_shortwave', 'toa_up_shortwave'] 
        elif socrates == 0:
            names_3D = ['vwp', 'lwp', 'reske', 'subke', 'cltop', 'clbas']
elif specific_diagnostics == 0:
    names_1D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 1]
    names_2D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 2]
    names_3D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 3]

#names_4D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 4]

#names_1D = ['LWP_mean']
#names_2D = ['liquid_mmr_mean']
#names_3D = ['lwp', 'cltop', 'clbas']
names_4D = ['q_cloud_liquid_number', 'q_cloud_liquid_mass', 'q_vapour', 'th', 'u', 'v', 'w']

# Create object list
scalars = [DS[name] for name in names_1D]
profiles = [DS[name] for name in names_2D]
scenes = [DS[name] for name in names_3D]
slices = [DS[name] for name in names_4D]

# Find key values from options database
for n, i in enumerate(DS.options_database):                                                     
    if str(i.values[0]) == "qlcrit":                                                              
        qlcritn = n                                                                                  
    elif str(i.values[0]) == "rlvap":                                                             
        rlvapn = n
    elif str(i.values[0]) == "cp":
        cpn = n

# Calculate TKE and append
tke_mean = (DS.reske_mean + DS.subke_mean)  #*1000 - I think the units on this are wrong, but if not, need to multiply by 1000 for g m-2
tke_mean.name = "TKE_mean"
scalars.append(tke_mean)

# Calculate cloud fraction and append
real_cfrac = np.mean(np.mean(DS.q_cloud_liquid_mass>=float(DS.options_database[qlcritn].values[1]), axis=3)>0.0, axis=(1,2)) 
#sumz = DS.q_cloud_liquid_mass.sum(dim="z")
#cf_time = sumz.where(sumz.values < 1e-5, 1).where(sumz.values > 1e-5,0).mean(dim=["x","y"])
#cf_time.plot()
#plt.show()
scalars.append(real_cfrac)

# Calculate liquid water potential temperature
DS_retime = DS.rename({str(DS.LWP_mean.dims[0]): 'time_1', str(DS.uu_mean.dims[0]): 'time_2', str(DS.reske.dims[0]):'time_3'})

####
thref = np.mean(DS_retime.thref)
absolute_T = np.mean(DS_retime.th, axis=(1,2)) + thref
theta_coarse = DS_retime.theta_mean.sel(time_2=DS_retime.th.time_3.values, method='nearest')
lmmr_mean_coarse = DS_retime.liquid_mmr_mean.sel(time_2=DS_retime.th.time_3.values,
	method='nearest')
# maybe this should be q_cloud_liquid_mass ^
th_c_dims=theta_coarse.assign_coords(time_2=range(len(theta_coarse['time_2']))).rename({'time_2':
'time'})
lmmr_mean_dims=lmmr_mean_coarse.assign_coords(time_2=range(len(theta_coarse['time_2']))).rename({'time_2':'time'})
abs_T = absolute_T.assign_coords(time_3=range(len(absolute_T['time_3']))).rename({'time_3': 'time'})
lpot = th_c_dims-(float(DS_retime.options_database[rlvapn].values[1])*th_c_dims/(float(DS_retime.options_database[cpn].values[1])*abs_T))*lmmr_mean_dims
lpot.name = "Theta_l"
profiles.append(lpot)

# Plot using ncplotlib
savepath = '/gws/nopw/j04/carisma/eers/dycoms_sim/' + sys.argv[2]

for var in scalars:
    if spin_drop == True:
	var = var[110:]
    fig, axes = plt.subplots()
    fig = ncplt.scalar(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')
    plt.close()

print("*** Scalars processed ***")

for var in profiles:
    if spin_drop == True:
        var = var[3:]
    fig, axes = plt.subplots()
    fig = ncplt.profile(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')
    plt.close()

print("*** Profiles processed ***")

if spin_drop == True:
    [ncplt.scene(var[2:], savepath) for var in scenes]
else:
    [ncplt.scene(var, savepath) for var in scenes]
print('*** Scenes processed ***')

if spin_drop == True:
    [ncplt.vslice(var[2:], 60, 'x', savepath) for var in slices]
else:
    [ncplt.vslice(var, 60, 'x', savepath) for var in slices]
print('*** Vertical slices processed ***')

# Calculate cloud boundaries 
clbas_ave = np.mean(DS.clbas.where(DS['clbas']!=0.0), axis=(1,2))
cltop_ave = np.mean(DS.cltop.where(DS['cltop']!=0.0), axis=(1,2))
qt_mass = DS.q_vapour + DS.q_cloud_liquid_mass + DS.q_rain_mass
qt_mass_mean = np.mean(qt_mass, axis=(1,2))
t = qt_mass_mean.dims[0]
timeseries=qt_mass_mean[t]
indices = []
# find index in each time step closest to isoline
for time in qt_mass_mean:
    idx = np.abs(time - 0.008).argmin()
    indices.append(int(idx.values))  

# find corresponding height
height = []
for i in indices:
    height.append(int(qt_mass_mean[qt_mass_mean.dims[1]][i].values))

# plot heights
fig, axes = plt.subplots()
plt.plot(qt_mass_mean[qt_mass_mean.dims[0]], height, color="navy", label="Average inversion height")
cltop_ave.plot(color="blue", label="Average cloud top")
clbas_ave.plot(color="cyan", label="Average cloud base")
plt.ylim(0, 1200)
plt.title("Cloud Boundaries")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.legend()
plt.savefig(savepath + "cloudbound.png")


