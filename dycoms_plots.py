#!/usr/bin/env python3
import sys
from sys import argv
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ncplotlib as ncplt

def compare(axes, dycoms_var, converted_monc, colour_dycoms, colour_monc, title, ylim, ylabel):
    # Average
    dycoms_var[0].plot(color=colour_dycoms, ax=axes, label="S05 Ensemble Mean")
    # First and third percentiles 
    dycoms_var[4].plot(color=colour_dycoms, ax=axes, alpha=0.5)
    dycoms_var[5].plot(color=colour_dycoms, ax=axes, alpha=0.5)
    axes.fill_between(dycoms_var.time, dycoms_var[4], dycoms_var[5], color=colour_dycoms, alpha=0.5, label="S05 Ensemble 1st & 3rd percentiles")
    # Max and min
    dycoms_var[2].plot(color=colour_dycoms, ax=axes, alpha=0.2)
    dycoms_var[3].plot(color=colour_dycoms, ax=axes, alpha=0.2)
    axes.fill_between(dycoms_var.time, dycoms_var[2], dycoms_var[3], color=colour_dycoms, alpha=0.2, label="S05 Ensemble Max & Min")
    
    # MONC simulation
    converted_monc.plot(color=colour_monc, ax=axes, label="MONC sim")
    
    axes.set_ylim(0, ylim)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel(ylabel)
    axes.set_title(title)


def comp_profile(axes, dycoms_var, converted_monc, colour_dycoms, colour_monc, title):
    # Average
    dycoms_var[0][4].plot(y=dycoms_var[0][4].dims[0], color=colour_dycoms, ax=axes, label="S05 Ensemble Mean")
    # First and third percentiles 
    dycoms_var[4][4].plot(y=dycoms_var[4][4].dims[0], color=colour_dycoms, ax=axes, alpha=0.5)
    dycoms_var[5][4].plot(y=dycoms_var[5][4].dims[0], color=colour_dycoms, ax=axes, alpha=0.5)
    axes.fill_betweenx(dycoms_var[dycoms_var[0].dims[1]], dycoms_var[4][4], dycoms_var[5][4], color=colour_dycoms, alpha=0.5, label="S05 Ensemble 1st & 3rd percentiles")
    # Max and min
    dycoms_var[2][4].plot(y=dycoms_var[2][4].dims[0], color=colour_dycoms, ax=axes, alpha=0.2)
    dycoms_var[3][4].plot(y=dycoms_var[3][4].dims[0], color=colour_dycoms, ax=axes, alpha=0.2)
    axes.fill_betweenx(dycoms_var[dycoms_var[0].dims[1]], dycoms_var[2][4], dycoms_var[3][4], color=colour_dycoms, alpha=0.2, label="S05 Ensemble Max & Min")
    # MONC simulation
    ind1 = (np.abs(converted_monc[converted_monc.dims[0]] - 10800)).argmin() # changed this to be 4th not 5th
    ind2 = (np.abs(converted_monc[converted_monc.dims[0]] - 14040)).argmin()
    
    if len(converted_monc.dims)==2:
        line = np.mean(converted_monc[[ind1,ind2],:], axis=(0)).plot(y=converted_monc.dims[-1], color=colour_monc, label="MONC sim")
    elif len(converted_monc.dims)==4:
        line = np.mean(converted_monc[[ind1,ind2],:,:,:], axis=(0,1,2)).plot(y=converted_monc.dims[-1], color=colour_monc, label="MONC sim")
    
    axes.set_ylim(0, 1200)
    axes.set_xlabel(dycoms_var.name + " [" + dycoms_var.units + "]")
    axes.set_ylabel("Height [m]")
    axes.set_title(title)

    return line



# Open datasets
nc_mnc = sys.argv[1]
nc_dcms = "./gcss7.nc"
DS_monc = xr.open_dataset(nc_mnc)
DS_dycoms = xr.open_dataset(nc_dcms)

set_monc_colour = "fuchsia"
set_dycoms_colour = "blue"

### Find key values from options database:

for n, i in enumerate(DS_monc.options_database):
    if str(i.values[0]) == "b'qlcrit'":
        qlcritn = n
    elif str(i.values[0]) == "b'rlvap'":
        rlvapn = n
    elif str(i.values[0]) == "b'cp'":
        cpn = n

### Cloud Boundaries ###
#fig, axes = plt.subplots()
fig = plt.figure(figsize=(15,12))
#fig, ax = plt.subplots(3, 5)
#ax = subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
#fig.add_subplot(3, 1, 1)
axes = fig.add_subplot(3,5,1)
# Convert clbas 
clbas_ave = np.mean(DS_monc.clbas, axis=(1,2))
#cltop_ave = np.mean(DS_monc.cltop, axis=(1,2))

# Average for cloud base and top
DS_dycoms.zb_bar[0].plot(color=set_dycoms_colour, ax=axes)
DS_dycoms.zi_bar[0].plot(color=set_dycoms_colour, ax=axes)
# First and third percentiles cloud base 
DS_dycoms.zb_bar[4].plot(color=set_dycoms_colour, ax=axes, alpha=0.5)
DS_dycoms.zb_bar[5].plot(color=set_dycoms_colour, ax=axes, alpha=0.5)
axes.fill_between(DS_dycoms.zb_bar.time, DS_dycoms.zb_bar[4], DS_dycoms.zb_bar[5], color=set_dycoms_colour, alpha=0.5)
# Max and min cloud base
DS_dycoms.zb_bar[2].plot(color=set_dycoms_colour, ax=axes, alpha=0.2)
DS_dycoms.zb_bar[3].plot(color=set_dycoms_colour, ax=axes, alpha=0.2)
axes.fill_between(DS_dycoms.zb_bar.time, DS_dycoms.zb_bar[2], DS_dycoms.zb_bar[3], color=set_dycoms_colour, alpha=0.2)
# First and third percentiles cloud top
DS_dycoms.zi_bar[4].plot(color=set_dycoms_colour, ax=axes, alpha=0.5)
DS_dycoms.zi_bar[5].plot(color=set_dycoms_colour, ax=axes, alpha=0.5)
axes.fill_between(DS_dycoms.zi_bar.time, DS_dycoms.zi_bar[4], DS_dycoms.zi_bar[5], color=set_dycoms_colour, alpha=0.5)
# Max and min cloud top
DS_dycoms.zi_bar[2].plot(color=set_dycoms_colour, ax=axes, alpha=0.2)
DS_dycoms.zi_bar[3].plot(color=set_dycoms_colour, ax=axes, alpha=0.2)
axes.fill_between(DS_dycoms.zi_bar.time, DS_dycoms.zi_bar[2], DS_dycoms.zi_bar[3], color=set_dycoms_colour, alpha=0.2)

# MONC simulation
clbas_ave.plot(color=set_monc_colour, ax=axes)
#cltop_ave.plot(color=set_monc_colour, ax=axes)

axes.set_ylim(0, 1000)
axes.set_xlabel("Time [s]")
axes.set_ylabel("Height [m]")
axes.set_title("Cloud Boundaries")


### Inversion height ### --- based on 8 g/kg isoline
qt=DS_monc.q_vapour + DS_monc.q_cloud_liquid_mass + DS_monc.q_rain_mass
qt_avexy=np.mean(qt, axis=(1,2))
t=qt_avexy.dims[0]
timeseries=qt_avexy[t]
indices = []

# find index in each time step closest to isoline
for time in qt_avexy:
    idx = np.abs(time - 0.008).argmin()
    indices.append(int(idx.values))

# find corresponding height
height = []
for i in indices:
    height.append(int(qt_avexy[qt_avexy.dims[1]][i].values))

# plot heights
plt.plot(qt_avexy[qt_avexy.dims[0]], height, color=set_monc_colour)
#plt.savefig("Cloudboundaries.png")
#plt.show()


### LWP ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3, 5, 2)
# Convert lwp_mean
clbas_avex = np.mean(DS_monc.clbas, axis=1)
lwp_mean_gkg = DS_monc.LWP_mean*1000

compare(axes, DS_dycoms.lwp_bar, lwp_mean_gkg, set_dycoms_colour, set_monc_colour, "Liquid Water Path", 80, "LWP [g kg^(-1)]")
#plt.legend()
#plt.savefig("LWP_comp.png")
#plt.show() 


### Cloud Fraction ###  --- this depends on qlcrit being index 243 in options database 
#fig, axes = plt.subplots()
axes = fig.add_subplot(3, 5, 3)
real_cfrac = np.mean(np.mean(DS_monc.q_cloud_liquid_mass>=float(DS_monc.options_database[qlcritn].values[1]), axis=3)>0.0, axis=(1,2))
compare(axes, DS_dycoms.cfrac, real_cfrac, set_dycoms_colour, set_monc_colour, "Cloud Fraction", 1, "Cloud Fraction [%]")
#plt.savefig("cloudfrac.png")
#plt.show()

###mibs_cfrac1 = np.mean(((DS_monc.q_cloud_liquid_number.sum(dim="z")>=1)+0), axis=(1,2))
###fig, axes = plt.subplots()
###mibs_cfrac2 = np.mean(DS_monc.total_cloud_fraction, axis=1)
###compare(axes, DS_dycoms.cfrac, mibs_cfrac2, set_dycoms_colour, set_monc_colour, "Cloud Fraction 2", 1, "Cloud Fraction [%]")
### TKE ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3, 5, 4)
tke_mean = (DS_monc.reske_mean + DS_monc.subke_mean)*1000   # sum of averages = average of sum
compare(axes, DS_dycoms.tke, tke_mean, set_dycoms_colour, set_monc_colour, "TKE", 1000, "tke")
#plt.savefig("tke.png")
#plt.show()
#
#### Heat Flux ###
#fig, axes = plt.subplots()
#compare(axes, DS_dycoms.lhf_bar, DS_monc.lathf_mean, set_dycoms_colour, set_monc_colour, "Latent Heat Flux", 120, "Latent Heat Flux [W m^(-2)]")
#plt.savefig("latflux.png")
#plt.show()
#
#fig, axes = plt.subplots()
#compare(axes, DS_dycoms.shf_bar, DS_monc.senhf_mean, set_dycoms_colour, set_monc_colour, "Sensible Heat Flux", 120, "Sensible Heat Flux [W m^(-2)]")
#plt.savefig("senflux.png")
#plt.show()
#
#plt.show()
### Winds ###
#fig, axes = plt.subplots()

axes = fig.add_subplot(3,5,6)
comp_profile(axes, DS_dycoms.u, DS_monc.u, set_dycoms_colour, set_monc_colour, "U Mean Wind")
#plt.savefig("uwind.png")
#plt.show()

#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,7)
comp_profile(axes, DS_dycoms.v, DS_monc.v, set_dycoms_colour, set_monc_colour, "V Mean Wind")
#plt.savefig("vwind.png")
#plt.show()


### Liquid potential temperature ###
index_list=[0, 1, 2, 3, 5, 6]
thref_reduce=[]
for i in index_list:
    thref_reduce.append(DS_monc.thref[i])

absolute_T = np.mean(DS_monc.th[:6], axis=(1,2)) + thref_reduce
mean_abs_T = np.mean(absolute_T[4:5], axis=0) # average over 4th hour so lpot calc will be with the 4th hour absT but shouldn't matter for the rest since only taking the 4th hour anyway.

#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,8)
theta=DS_monc.theta_mean
lpot=theta - (float(DS_monc.options_database[rlvapn].values[1])*theta/(float(DS_monc.options_database[cpn].values[1])*mean_abs_T))*DS_monc.liquid_mmr_mean
comp_profile(axes, DS_dycoms.thetal, lpot, set_dycoms_colour, set_monc_colour, "Theta_l")
#plt.savefig("thetal.png")
#plt.show()

'''
# Calculating theta from thetal values:
thetatop = 297.5 + (1500 - 840)**(1/3)
thetal = [289, 289, 299, thetatop]
T = 289
rl = [0.0, 0.000475, 0.0, 0.0]
theta = []
for i, n in enumerate(thetal):
    theta.append(n*(1 - (2.47e6/1015.0)*(float(rl[i])/T)))

# note that when cloud_liquid_mass profile is false then rl is all 0 so just is potential temp.
'''



#### Density ###
#fig, axes = plt.subplots()
#comp_profile(axes, DS_dycoms.dn0, DS_monc.rho, set_dycoms_colour, set_monc_colour, "Density [kg m^(-3)]")
#plt.show()
#plt.savefig("density.png")
#
### Liquid Water Mixing Ratio ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,9)
monc_lmmr = DS_monc.liquid_mmr_mean*1000
comp_profile(axes, DS_dycoms.rl, monc_lmmr, set_dycoms_colour, set_monc_colour, "Liquid Mass Mixing Ratio")
#plt.savefig("liquidmmr.png")
#plt.show()


### Total mass mixing ratio ###
#fig,axes = plt.subplots()
axes = fig.add_subplot(3,5,10)
total_mmr=DS_monc.q_vapour + DS_monc.q_cloud_liquid_mass + DS_monc.q_rain_mass
total_mmr_mean=DS_monc.vapour_mmr_mean + DS_monc.liquid_mmr_mean
comp_profile(axes, DS_dycoms.rt, total_mmr*1000, set_dycoms_colour, set_monc_colour, "Total Mass Mixing Ratio - q fields")
comp_profile(axes, DS_dycoms.rt, total_mmr_mean*1000, set_dycoms_colour, "orange", "Total Mass Mixing Ratio")
#plt.savefig("totalmmr.png")
#plt.show()

### u variance ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,11)
comp_profile(axes, DS_dycoms.u_var, DS_monc.uu_mean, set_dycoms_colour, set_monc_colour, "uu variance")
#plt.savefig("uvar.png")
#plt.show()

### v variance ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,12)
comp_profile(axes, DS_dycoms.v_var, DS_monc.vv_mean, set_dycoms_colour, set_monc_colour, "vv variance")
#plt.savefig("vvar.png")
#plt.show()

### w variance ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,13)
comp_profile(axes, DS_dycoms.w_var, DS_monc.ww_mean, set_dycoms_colour, set_monc_colour, "w variance")
#plt.savefig("wvar.png")
#plt.show()

### w third moment ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,14)
comp_profile(axes, DS_dycoms.w_skw, DS_monc.www_mean, set_dycoms_colour, set_monc_colour, "w third moment")
#plt.savefig("wwwmean.png")
#plt.show()
#
#

#### res TKE ###
#fig, axes = plt.subplots()
#tke = 0.5*(DS_monc.uu_mean + DS_monc.vv_mean + DS_monc.ww_mean)
#tkeres_mean = tke - DS_monc.tkesg_mean
#comp_profile(axes, DS_dycoms.E, tkeres_mean, set_dycoms_colour, set_monc_colour, "Resolved TKE")
#plt.savefig("restke.png")
#plt.show()
#
#### sfs TKE ###
#fig, axes = plt.subplots()
#comp_profile(axes, DS_dycoms.e, DS_monc.tkesg_mean, set_dycoms_colour, set_monc_colour, "Subfilter TKE")
#plt.savefig("subtke.png")
#plt.show()
#
#
#### Shear ###
#fig, axes = plt.subplots()
##sheartmp=DS_monc.resolved_shear_production + DS_monc.subgrid_shear_stress
#comp_profile(axes, DS_dycoms.shr_prd, DS_monc.resolved_shear_production, set_dycoms_colour, set_monc_colour, "Shear Production")
##comp_profile(axes, DS_dycoms.shr_prd, DS_monc.subgrid_shear_stress, set_dycoms_colour, set_monc_colour, "Shear Production")
##comp_profile(axes, DS_dycoms.shr_prd, sheartmp, set_dycoms_colour, set_monc_colour, "Shear Production")
#plt.savefig("shear")
#plt.show()
#
#
### Buoyancy ###
#fig, axes = plt.subplots()
axes = fig.add_subplot(3,5,15)
comp_profile(axes, DS_dycoms.boy_prd, DS_monc.resolved_buoyant_production, set_dycoms_colour, set_monc_colour, "Buoyant Production")
#plt.savefig("buoyancy")
#plt.show()
#
#### sfs Buoyancy flux ###  --- not confident about this
#fig, axes = plt.subplots()
#sfs_boy = DS_monc.uw_buoyancy + DS_monc.vw_buoyancy + DS_monc.ww_buoyancy
#comp_profile(axes, DS_dycoms.sfs_boy, sfs_boy, set_dycoms_colour, set_monc_colour, "Subfilter Buoyancy Flux")
#plt.show()
#
#
#### Transport ###  --- not confident about this
#fig, axes = plt.subplots()
#transport = DS_monc.pressure_transport + DS_monc.resolved_turbulent_transport
#comp_profile(axes, DS_dycoms.transport, transport, set_dycoms_colour, set_monc_colour, "Resolved transport")
#comp_profile(axes, DS_dycoms.transport, DS_monc.pressure_transport, set_dycoms_colour, set_monc_colour, "Resolved transport")
#comp_profile(axes, DS_dycoms.transport, DS_monc.resolved_turbulent_transport, set_dycoms_colour, set_monc_colour, "Resolved transport")
#plt.show()
#
#
#### Dissipation ###
#fig, axes = plt.subplots()
#comp_profile(axes, DS_dycoms.dissipation, DS_monc.dissipation_mean, set_dycoms_colour, set_monc_colour, "Dissipation")
#plt.show()
#
#
#### Storage ###  --- not confident about this
#fig, axes = plt.subplots()
#comp_profile(axes, DS_dycoms.storage, DS_monc.tke_tendency, set_dycoms_colour, set_monc_colour, "Storage")
#plt.show()
plt.tight_layout()
plt.savefig(nc_mnc[:-3] + "_comp.png")
plt.show()
