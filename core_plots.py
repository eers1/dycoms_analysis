import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ncplotlib as ncplt

# Open dataset
nc = '/work/xfc/vol1/user_cache/eers/dycoms_diurnal_test/merged.nc'
DS = xr.open_dataset(nc)

# Select specific diagnostics or take all
names_1D = ['VWP_mean', 'LWP_mean', 'RWP_mean', 'reske_mean', 'subke_mean', 'w_max', 'senhf_mean', 'lathf_mean', 'surface_precip_mean']
names_2D = ['theta_mean', 'vapour_mmr_mean', 'liquid_mmr_mean', 'rain_mmr_mean', 'rh_mean', 'subgrid_turbulent_transport', 'total_cloud_fraction', 'liquid_cloud_fraction', 'resolved_turbulent_transport', 'tke_tendency','u_wind_mean', 'v_wind_mean', 'w_wind_mean', 'uu_mean', 'vv_mean', 'ww_mean']
names_3D = ['surface_precip', 'vwp', 'lwp', 'rwp', 'reske', 'subke', 'cltop', 'clbas', 'toa_up_longwave', 'toa_down_shortwave', 'toa_up_shortwave']
names_4D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 4]

# Create object list
scalars = [DS[name] for name in names_1D]
profiles = [DS[name] for name in names_2D]
scenes = [DS[name] for name in names_3D]
slices = [DS[name] for name in names_4D]

# Plot using ncplotlib
savepath = '../../save/'

for var in scalars:
    fig, axes = plt.subplots()
    fig = ncplt.scalar(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')

for var in profiles:
    fig, axes = plt.subplots()
    fig = ncplt.profile(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')

#for var in scenes:
#    #plt.subplots()
#    ncplt.scene(var, 10)
#    plt.savefig(savepath + var.name + '.png')
#   print(var.name + ' saved')
#
#print('*** scenes processed ***')
[ncplt.scene(var,5) for var in scenes]
[ncplt.vslice(var, 'x', 60, 5) for var in slices]

