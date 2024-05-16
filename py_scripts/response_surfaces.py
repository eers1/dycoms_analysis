#!/usr/bin/env python3.7

import numpy as np
import matplotlib.pyplot as plt
import emlibplot as elp
from matplotlib import rc, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


font = 21
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': font})

nd = 'low'
nd_ = '' if nd=='low' else f'{nd}_nd_'
noise_type='extras'

lwp_mean = f'../data_lwp_cloud/dycoms_data_{nd}_nd_lwp_cloud_mean.csv'
lwp_tend = f'../data_lwp_cloud/dycoms_data_{nd}_nd_lwp_cloud_teme.csv'
cf_last = f'../data_cloud_frac/dycoms_data_{nd}_nd_cloud_frac_mean.csv'
cf_tend = f'../data_cloud_frac/dycoms_data_{nd}_nd_cloud_frac_teme.csv'

fig_cmap, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,9.5))
ax_dict={}
ax_dict["ax0"] = [axes[0,0]]
ax_dict["ax1"] = [axes[0,1]]
ax_dict["ax2"] = [axes[1,0]]
ax_dict["ax3"] = [axes[1,1]]
plt.subplots_adjust(top=0.95, bottom=0.18, wspace=0.2, hspace=0.2)

cbar_loc = [1.02, 0, 0.05, 0.4]

### Create samples list
new_points = 500
samples, theta_list, qt_list = elp.create_samples(-9, 0, 2, 20, new_points)

### levels
L = 20

### Create K parameter
x = np.linspace(2, 20, 10)
y = []
cp = 1015
Lv = 2.47e6
for val in x:
    y.append(val*(cp)*1e3/(Lv*(0.23-1)))


### Options Dict:
extras = True if noise_type!="exact" else False
options_dict = {
    'xlabel' : r'$\Delta \theta$ (K)',
    'ylabel' : r'$\Delta$q$_{t}$ (g kg$^{-1}$)',
    'newpoints' : new_points,
    'pltvar' : False,
    'savebool': False,
    'markerfonts' : 13,
    'font': font,
    'dsize' : 120,
    'kx' : x,
    'ky' : y,
    'samples' : samples,
    'theta_list': theta_list,
    'qt_list' : qt_list,
    'extras' : extras,
    'edgesize':1
    }


### LWP
ax = ax_dict["ax0"][0]
exact_predictions = np.loadtxt(f'../predictions/lwp_cloud/pre_tot_{nd_}lwp_cloud_mean_{noise_type}.csv',delimiter=',',skiprows=1)
vmin = 0
vmax = 180 if nd=='low' else 250
norm = colors.TwoSlopeNorm(vcenter=0.5*vmax,vmin=0,vmax=vmax)

cmap_string = "nuuk"
cm_data =np.loadtxt(f"../colour_maps/ScientificColourMaps6/{cmap_string}/{cmap_string}.txt")
cmap = colors.LinearSegmentedColormap.from_list(cmap_string, cm_data)
#cmap = cmap.reversed()

title = ''
figname = 'lwp_nv_nugget_estim_%s.png'
output_label = 'Liquid water path (g m$^{-2}$)'
diverging = False

L = (vmax-vmin)/9
levels = np.linspace(vmin,vmax,20)

Em_LWP_exact = elp.Emulator(
    lwp_mean,
    f'../predictions/lwp_cloud/pre_tot_{nd_}lwp_cloud_mean_{noise_type}.csv', **options_dict)
colour_obj = Em_LWP_exact.plot_2DCmap(title, output_label,  cmap, levels,diverging, True,norm,figname%('cmap'),fig=fig_cmap,ax=ax,extend="both", transect=False)
#cax = ax.inset_axes(cbar_loc, transform=ax.transAxes)
cbar=fig_cmap.colorbar(colour_obj, ax=ax,norm=norm,  orientation='vertical',label=output_label,format="%0.0f",)
ax.xaxis.set_visible(False)
ax.text(0.04,0.71,'A',transform=ax.transAxes,c='black')
ax.text(0.1,0.8,'B',transform=ax.transAxes,c='black')
ax.text(0.01,1.03,'a)',transform=ax.transAxes,fontsize=font)

### LWP tend
ax = ax_dict["ax1"][0]
exact_predictions = np.loadtxt(f'../predictions/lwp_cloud/pre_tot_{nd_}lwp_cloud_teme_{noise_type}.csv',delimiter=',',skiprows=1)
exact_pred_mean = exact_predictions[:,1]
cm_vmin = min(exact_pred_mean)
cm_vmax = max(exact_pred_mean)
extreme = max(abs(cm_vmin),abs(cm_vmax))
#norm = colors.TwoSlopeNorm(vcenter=0,vmin=(-18),vmax=18)
norm = colors.TwoSlopeNorm(vcenter=0,vmin=(-1)*extreme,vmax=extreme)
cm_data =np.loadtxt("../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()

figname = 'lwp_tend_nugget_estim_%s.png'
output_label = 'Liquid water path \ntendency (g m$^{-2}$ hr$^{-1}$)'
diverging = True

levels = np.linspace((-1)*extreme,extreme,20)
#levels = np.linspace(-18,18,20)

Em_LWP_tend_exact = elp.Emulator(
    lwp_tend,
    f'../predictions/lwp_cloud/pre_tot_{nd_}lwp_cloud_teme_{noise_type}.csv', **options_dict)
colour_obj = Em_LWP_tend_exact.plot_2DCmap(title, output_label, cmap,levels, diverging, False,norm,figname%('cmap'),fig=fig_cmap,ax=ax,extend="both", transect=False)
#cax = ax.inset_axes(cbar_loc, transform=ax.transAxes)
fig_cmap.colorbar(colour_obj, ax=ax, label=output_label,format="%0.0f", orientation='vertical')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.text(0.04,0.71,'A',transform=ax.transAxes,c='black')
ax.text(0.1,0.8,'B',transform=ax.transAxes,c='black')
ax.text(0.01,1.03,'b)',transform=ax.transAxes,fontsize=font)

### CF
ax = ax_dict["ax2"][0]
exact_predictions = np.loadtxt(f'../predictions/cloud_frac/pre_tot_{nd_}cloud_frac_mean_{noise_type}.csv',delimiter=',',skiprows=1)
vmin = 0.2
vmax = 1
norm = colors.TwoSlopeNorm(vcenter=vmin+(0.5*(vmax-vmin)),vmin=vmin,vmax=vmax)
cm_data =np.loadtxt(f"../colour_maps/ScientificColourMaps6/{cmap_string}/{cmap_string}.txt")
cmap = colors.LinearSegmentedColormap.from_list(cmap_string, cm_data)
#cmap = cmap.reversed()

figname = 'cloud_frac_exact_%s.png'
output_label =  'Cloud Fraction'
diverging = False
cftend = False

levels = np.linspace(vmin,vmax,16)

Em_CF_exact = elp.Emulator(
    cf_last,
    f'../predictions/cloud_frac/pre_tot_{nd_}cloud_frac_mean_{noise_type}.csv', **options_dict)

colour_obj = Em_CF_exact.plot_2DCmap(title, output_label, cmap, levels,diverging, False,norm,figname%('cmap'),fig=fig_cmap,ax=ax,extend="min", transect=False)
#cax = ax.inset_axes(cbar_loc, transform=ax.transAxes)
fig_cmap.colorbar(colour_obj, ax=ax, label=output_label,format="%0.1f", orientation='vertical')
ax.text(0.04,0.71,'A',transform=ax.transAxes,c='black')
ax.text(0.1,0.8,'B',transform=ax.transAxes,c='black')
ax.text(0.01,1.03,'c)',transform=ax.transAxes,fontsize=font)

### CF tend
ax = ax_dict["ax3"][0]
exact_predictions = np.loadtxt(f'../predictions/cloud_frac/pre_tot_{nd_}cloud_frac_teme_{noise_type}.csv',delimiter=',',skiprows=1)
exact_pred_mean = exact_predictions[:,1]
cm_vmin = min(exact_pred_mean)
cm_vmax = max(exact_pred_mean)
extreme = max(abs(cm_vmin),abs(cm_vmax))
norm = colors.TwoSlopeNorm(vcenter=0,vmin=-0.04,vmax=0.04)
cm_data =np.loadtxt("../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()

figname = 'cloud_frac_tend_nugget_estim_%s.png'
output_label = 'Cloud Fraction \nTendency (hr$^{-1}$)'
diverging = True
cftend = True

levels = np.linspace((-1)*extreme,extreme,20)

Em_CF_tend_exact = elp.Emulator(
    cf_tend,
    f'../predictions/cloud_frac/pre_tot_{nd_}cloud_frac_teme_{noise_type}.csv', **options_dict)

colour_obj = Em_CF_tend_exact.plot_2DCmap(title, output_label,cmap,levels, diverging, False,norm,figname%('cmap'),fig=fig_cmap,ax=ax,extend="both", transect=False)
ax.yaxis.set_visible(False)
ax.text(0.04,0.71,'A',transform=ax.transAxes,c='black')
ax.text(0.1,0.8,'B',transform=ax.transAxes,c='black')
ax.text(0.01,1.03,'d)',transform=ax.transAxes,fontsize=font)
#cax = ax.inset_axes(cbar_loc, transform=ax.transAxes)
fig_cmap.colorbar(colour_obj, ax=ax, label=output_label,format="%1.2f", orientation='vertical')
fig_cmap.text(0.5,0.12, options_dict['xlabel'], ha='center', va='center',fontsize=font)
fig_cmap.text(0.05,0.58, options_dict['ylabel'], ha='center', va='center',rotation='vertical',fontsize=font)

if noise_type!="exact":
    custom_lines = [Line2D([], [], c='black', lw=0, marker='v', markerfacecolor="white", markersize=11),
                Line2D([], [], c='black', lw=0, marker='o', markerfacecolor="white", markersize=11),
                Line2D([], [], c='black', lw=0, marker='s', markerfacecolor="white", markersize=11),
                Line2D([0], [0], c='black', lw=1, linestyle='--')]
    custom_labels = ["Base", "Training", "Validation", "$\kappa$"]
    location = (-1.48,-0.41)
else:
    custom_lines = [Line2D([], [], c='black', lw=0, marker='v', markerfacecolor="white", markersize=11),
                Line2D([], [], c='black', lw=0, marker='o', markerfacecolor="white", markersize=11),
                Line2D([], [], c='black', lw=0, marker='s', markerfacecolor="white", markersize=11),
                Line2D([0], [0], c='black', lw=1, linestyle='--')]
    custom_labels = ["Base", "Training", "Validation", "$\kappa$"]
    location = (4,4)
    
ax_dict['ax3'][0].legend(custom_lines, custom_labels, loc=location, ncol=5,shadow=False,fancybox=False)

fig_cmap.savefig(f'../figures/response_surfaces_{noise_type}.png')
fig_cmap.savefig(f'../figures/response_surfaces_{noise_type}.pdf')
plt.show()
plt.close()
