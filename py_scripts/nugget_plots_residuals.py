#!/usr/bin/env python
#### Using emlibplot to plot LWP with noise vectors

import numpy as np
import matplotlib.pyplot as plt
import py_scripts.emlibplot as elp
from matplotlib import rc, colors
#import sklearn.metrics as skm
import scipy.stats as sts
import matplotlib.gridspec as gridspec
from string import ascii_lowercase
import random

font=22
tickfont=18
legendfont=18
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': font})

plt.rc('font', size=font)        # controls default text sizes
plt.rc('axes', labelsize=font)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=tickfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=tickfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=legendfont)
#font = 25
#inplotfont=20
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': font})

def plot_sequence(calc,gen_type,options_dict, nv_type, fig, ax1, ax2, title, figname, data_path, transect_line, output_label, cmap, levels,diverging,climited,norm, cross_section_colour,specific_points,vmin, vmax,ylim,ensembles=None,values=None, ensemble_select=None):
    Elp_Em_obj = elp.Emulator(
        data_path,
        f'../predictions/{calc}/pre_tot_{calc}_{calc_type}_{nv_type}.csv', **options_dict)
    design_pred = np.loadtxt(f'../predictions/{calc}/pre_design_{calc}_{calc_type}_{nv_type}.csv', delimiter=',', skiprows=1)
    ax1.plot(transect_line[:,0], transect_line[:,1], color='hotpink')
    colour_obj = Elp_Em_obj.plot_2DCmap(title, output_label, cmap, levels, diverging, climited, norm, figname%('cmap'), fig=fig, ax=ax1, transect=True,extend='both',ensemble_select=ensemble_select)
    if ensembles is None: 
        transect, upper_bound, lower_bound = Elp_Em_obj.plot_transect(ax2, output_label, title,cross_section_colour,through_points=specific_points,design_predictions=design_pred,vmin=vmin*1.2, vmax=vmax*1.2,ylim=ylim)
    else:
        transect, upper_bound, lower_bound = Elp_Em_obj.plot_transect(ax2, output_label, title,cross_section_colour,ensembles=ensembles,values=values,through_points=specific_points,design_predictions=design_pred,vmin=vmin*1.2, vmax=vmax*1.2,ylim=ylim)

    return colour_obj, transect, upper_bound, lower_bound

def calc_rmse(calc, gen_type, nv_type, data_path, normal_res):
    print(nv_type)
    val_pred_data = np.loadtxt(f'../predictions/{calc}/pre_val_{calc}_{calc_type}_{nv_type}.csv', delimiter=',', skiprows=1)
    val_pred = val_pred_data[:,1]
    data_actual = np.loadtxt(data_path, delimiter=',')
    val_actual = data_actual[24:32,2]
    
    rmse = np.mean((np.subtract(val_actual, val_pred)**2))**0.5    
    print(nv_type, ": ", rmse)

    if nv_type not in ["extras", "1mag", "2mag","trial","exact"]:
        if normal_res==True:
            nugget_vec = np.loadtxt(f'../noise_files/{calc}/nv_normal_res_{calc_type}_{nv_type}.csv', delimiter=',')
        else:
            nugget_vec = np.loadtxt(f'../noise_files/{calc}/nv_res_{calc_type}_{nv_type}.csv', delimiter=',')
        val_nugget = nugget_vec[24:32]
        bool_list=[]
        for val_p,val_a,nugget in zip(val_pred, val_actual, val_nugget):
            print(val_p, val_a, nugget)
            if (val_a-nugget)<=val_p<=(val_a+nugget):
                bool_list.append(True)
            else:
                bool_list.append(False)
        print(bool_list)
        if all(bool_list):
            print("All true")
    return rmse

def ppe_residuals(ppe_predictions, ppe_actual, normal_res):
    if normal_res==True:
        residuals = [(actual - pred)/pred for pred, actual in zip(ppe_predictions, ppe_actual)]
    else:
        residuals = [(actual - pred) for pred, actual in zip(ppe_predictions, ppe_actual)]
    #residuals = ppe_predictions - ppe_actual
    return residuals

def nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, bin_lim_a, bin_lim_b, normal_res):
    exact_predictions = np.loadtxt(f'../predictions/{calc}/pre_tot_{calc}_{calc_type}_extras.csv',delimiter=',',skiprows=1)
    upper95 = exact_predictions[:,4]
    lower95 = exact_predictions[:,3]
    vmin = min(lower95)
    vmax = max(upper95)
    exact_pred_mean = exact_predictions[:,1]
    cm_vmin = min(exact_pred_mean)
    cm_vmax = max(exact_pred_mean)

    ### Create samples list
    new_points = 500
    samples, theta_list, qt_list = elp.create_samples(-9, 0, 2, 20, new_points)

    ### ensemble
    edata = np.loadtxt(f'../ensembles_csv/ensemble_{calc}_mean.csv', delimiter=',')
    if calc=='lwp_cloud':
        if calc_type=='mean':
            ensembles = edata[:,0]
            norm = colors.TwoSlopeNorm(vcenter=(cm_vmax/2),
                                       vmin=0,
                                       vmax=cm_vmax)
            levels = np.linspace(0,cm_vmax,15)
        elif calc_type=='teme':
            ensembles = edata[:,1]
            extreme = max(abs(cm_vmax),abs(cm_vmin))
            mextreme = (-1)*extreme
            vmin = -18
            vmax = 18
            norm = colors.TwoSlopeNorm(vcenter=0,vmin=(-1)*extreme,vmax=extreme)
            levels = np.linspace((-1)*extreme,extreme,20)
    elif calc=="cloud_frac":
        if calc_type=='mean':
            ensembles = edata[:,0]
            norm = colors.TwoSlopeNorm(vcenter=0.6,
                                       vmin=0.2,
                                       vmax=1)
            levels = np.linspace(0.2, 1,16)
        elif calc_type=='teme':
            ensembles = edata[:,1]
            extreme = max(abs(cm_vmax),abs(cm_vmin))
            mextreme = (-1)*extreme
            norm = colors.TwoSlopeNorm(vcenter=0,vmin=mextreme,vmax=extreme)
            levels = np.linspace(mextreme, extreme,20)
    else:
        print("switch case error")   

    ### Create K parameter
    x = np.linspace(2, 16.363636, 10)
    y = []
    for val in x:
        y.append(-0.55 * val)

    ### get transect line
    transect_line = np.loadtxt('../misc/transect_points.csv', delimiter=',')

    ### Options Dict:

    options_dict = {
        'xlabel' : r'$\Delta \theta$ (K)',
        'ylabel' : r'$\Delta q_{\textrm t}$ (g kg$^{-1}$)',
        'newpoints' : new_points,
        'pltvar' : False,
        'savebool' : False,
        'path' : '/home/rach/Emulator/Surfaces/',
        'markerfonts' : 13,
        'font': font,
        'dsize' : 120,
        'kx' : x,
        'ky' : y,
        'samples' : samples,
        'theta_list': theta_list,
        'qt_list' : qt_list,
        'extras' : True,
        'edgesize': 1
        }

#################################### Plots #############################################
    specific_points = [[4,8.47,10.86,13.90,15.62],[-8,-6.67,-4.90,-3.05,-1.94]]
    bins = np.linspace(hist_lim_x[0],hist_lim_x[1], 20)
    inplotfont = 20
    for nv_type, ax_1, ax_2, ax_3, t, selection, multiplier_p in zip(nv_type_list, top_row, bottom_row, gs1_axes, titles, ens_select, multipliers):
        if normal_res==True:
            design_res = np.loadtxt(f'../noise_files/{calc}/residuals/design_relative_residuals_{calc_type}.csv', delimiter=',')
        else:
            design_res = np.loadtxt(f'../noise_files/{calc}/residuals/design_residuals_{calc_type}.csv', delimiter=',')
        
        if nv_type not in ["extras", "1mag", "2mag","trial","exact"]:
            multiplier = f"{multiplier_p}_"
        else:
            multiplier = ""

        nv_type = f"{multiplier}{nv_type}"
        data_actual = np.loadtxt(f"../data_{calc}/dycoms_data_low_nd_{calc}_{calc_type}.csv", delimiter=',')
        data_actual = np.concatenate((data_actual[:20,-1], data_actual[32:38,-1]))        
        rmse = calc_rmse(calc, gen_type, nv_type, data_path, normal_res)
        figname = 'noise_em_%s.png'
        title = f'{t}'
        ax_2.text(0.97,0.03,'RMSE = {:.3}'.format(rmse), transform=ax_2.transAxes, fontsize=legendfont, horizontalalignment="right", verticalalignment="bottom")
        colour_obj, small_tran, small_cb, small_cb2 = plot_sequence(calc,gen_type, options_dict,nv_type,fig, ax_1, ax_2, title, figname, data_path, transect_line, output_label, cmap, levels,diverging,climited,norm,'black',specific_points,vmin, vmax,ylim,ensembles,ensemble_select=selection)

        design_pred = np.loadtxt(f'../predictions/{calc}/pre_design_{calc}_{calc_type}_{nv_type}.csv',delimiter=',',skiprows=1)
        ppe_pred = design_pred[:, 1]
        noise_res = ppe_residuals(data_actual,ppe_pred,normal_res)

        reps = 50
        ensembles = np.reshape(ensembles, [9,5])
        noise_res = np.empty((len(data_actual), reps))
        random.seed(11)
        for rep in range(reps):
            for i,j in enumerate([3,9,11,14,15,17,18,19,20]):
                data_actual[j-1] = ensembles[i][random.randint(0,4)]
            noise_res[:,rep] = ppe_residuals(data_actual,ppe_pred,normal_res)

        distro_arr = np.empty((len(bins)-1,reps))
        for i,col in enumerate(noise_res.transpose()):
            freq, bins_array = np.histogram(col,bins=bins,density=True)
            distro_arr[:,i] = freq
        n2 = np.mean(distro_arr,axis=1)

        n1, bin_array, p1 = ax_3.hist(design_res, alpha=0.5, color="#CE8964",  bins=bins, density=True, label="Model")
        bars = ax_3.bar(bin_array[:-1], n2, width=np.diff(bin_array), alpha=0.5, color="#2D93AD",align='edge', label="Emulator")

        print("n1 sum: ", sum(n1))
        print("n2 sum: ", sum(n2))
        minimums = [min(x,y) for x,y in zip(n1,n2)]
        overlap = sum(minimums)/sum(n1) # divided to make a fraction
        kstat = sts.ks_2samp(design_res, np.reshape(noise_res,np.size(noise_res)))
        ax_3.text(0.03,0.95, 'KS p-val: \n{:0.2f}'.format(kstat[1]), fontsize=legendfont, transform=ax_3.transAxes, horizontalalignment="left", verticalalignment="top")
        ax_3.text(0.97,0.95, 'Overlap: \n{:0.2f}'.format(overlap), fontsize=legendfont, transform=ax_3.transAxes, horizontalalignment="right", verticalalignment="top")
        ax_3.set_ylim(hist_lim_y)

    fig.colorbar(colour_obj, cax=cax, norm=norm, label=output_label,aspect=20)

    fig.text(0.475,0.365, options_dict['xlabel'], ha='center', va='center',fontsize=font)
    fig.text(0.012,0.85, options_dict['ylabel'], ha='center', va='center',rotation='vertical',fontsize=font)
    fig.text(0.012,0.55, output_label, ha='center', va='center',rotation='vertical',fontsize=font)
    fig.text(0.475,0.03, 'Residual Value', ha='center', va='center',fontsize=font)
    fig.text(0.012,0.2, 'Frequency', ha='center', va='center',rotation='vertical',fontsize=font)

    for i,(l1, l2, l3) in enumerate(zip(top_row,bottom_row,gs1_axes)):
        l1.xaxis.set_visible(False)
        if i != 0:
            l1.yaxis.set_visible(False)
            l2.yaxis.set_visible(False)
            l3.yaxis.set_visible(False)

    return small_tran, small_cb, ax_3, fig

#### Main ####
'''
calc = 'lwp_cloud'
data_path = '../data_lwp_cloud/dycoms_data_low_nd_lwp_cloud_mean.csv'
gen_type = 'main'
calc_type = 'mean'
multipliers = ['max']*8
normal_res = True   # change to false for unnormal
cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/tokyo/tokyo.txt")
cmap = colors.LinearSegmentedColormap.from_list('tokyo', cm_data)
cmap = cmap.reversed()
output_label = r'L (g m$^{-2}$)'
diverging = False
climited = True
ylim = [-25, 195]
hist_lim_x = (-0.2,0.2) # normal res
#hist_lim_x = (-12,12)  # unnormal res
hist_lim_y = (0,21)     # normal
#hist_lim_y = (0,0.6)   # unnormal
bin_size = 0.005

fig = plt.figure(figsize=(24,9))
gs1 = gridspec.GridSpec(2,8,left=0.048,bottom=0.42,top=0.95,right=0.93,hspace=0.1,wspace=0.1) # 4 cols 
a1 = fig.add_subplot(gs1[0])
b1 = fig.add_subplot(gs1[1])
c1 = fig.add_subplot(gs1[2])
d1 = fig.add_subplot(gs1[3])
e1 = fig.add_subplot(gs1[4])
f1 = fig.add_subplot(gs1[5])
g1 = fig.add_subplot(gs1[6])
h1 = fig.add_subplot(gs1[7])
    
a2 = fig.add_subplot(gs1[8])
b2 = fig.add_subplot(gs1[9])
c2 = fig.add_subplot(gs1[10])
d2 = fig.add_subplot(gs1[11])
e2 = fig.add_subplot(gs1[12])
f2 = fig.add_subplot(gs1[13])
g2 = fig.add_subplot(gs1[14])
h2 = fig.add_subplot(gs1[15])

gs2 = gridspec.GridSpec(1,8,left=0.048,bottom=0.09,top=0.33,right=0.93,hspace=0.1,wspace=0.1)
a3 = fig.add_subplot(gs2[0])
b3 = fig.add_subplot(gs2[1])
c3 = fig.add_subplot(gs2[2])
d3 = fig.add_subplot(gs2[3])
e3 = fig.add_subplot(gs2[4])
f3 = fig.add_subplot(gs2[5])
g3 = fig.add_subplot(gs2[6])
h3 = fig.add_subplot(gs2[7])

caxspec = gridspec.GridSpec(1,1,left=0.94,bottom=0.7,top=0.95,right=0.95)
cax = fig.add_subplot(caxspec[0])

nv_type_list = ['extras', 'g_all_everywhere', 'g_012_everywhere', 'g_048_everywhere', 'g_678_everywhere', 'ind_048_behaviour', 'ind_048_euclidean', 'two_regime']
titles = ['Extras', 'All comb.', '[012] comb.', '[048] comb.', '[678] comb.', '[048] behaviour', '[048] euclidean', 'Two regime']
ens_select = [None, np.asarray([2,8,10,13,14,16,17,18,19]), np.asarray([10,13,17]), np.asarray([17,2,18]), np.asarray([14,16,18]), np.asarray([17,2,18]), np.asarray([17,2,18]), np.asarray([2,13])]
top_row = [a1, b1, c1, d1, e1, f1, g1, h1]
bottom_row = [a2,b2,c2,d2,e2,f2,g2,h2]
gs1_axes = [a3, b3, c3, d3, e3, f3, g3, h3]

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.2, 0.2, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
gs1_axes[-1].legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_edit.png")
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_edit.pdf")
plt.show()
plt.close()
'''
'''
calc = 'lwp_cloud'
data_path = '../data_lwp_cloud/dycoms_data_low_nd_lwp_cloud_teme.csv'
gen_type = 'tend'
calc_type = 'teme'
multipliers = ['none']*8
normal_res = False
cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()

output_label = r'LWP tend $(g\; m^{-2}\; hr^{-1})$'
diverging = True
climited = False
ylim = [-7,24]
hist_lim_x = (-1.5, 1.5)
hist_lim_x = (-2, 2)
hist_lim_y = (0,3)
bin_size = 0.01

fig = plt.figure(figsize=(24,9))
gs1 = gridspec.GridSpec(2,8,left=0.048,bottom=0.42,top=0.95,right=0.93,hspace=0.1,wspace=0.1) # 4 cols 
a1 = fig.add_subplot(gs1[0])
b1 = fig.add_subplot(gs1[1])
c1 = fig.add_subplot(gs1[2])
d1 = fig.add_subplot(gs1[3])
e1 = fig.add_subplot(gs1[4])
f1 = fig.add_subplot(gs1[5])
g1 = fig.add_subplot(gs1[6])
h1 = fig.add_subplot(gs1[7])
    
a2 = fig.add_subplot(gs1[8])
b2 = fig.add_subplot(gs1[9])
c2 = fig.add_subplot(gs1[10])
d2 = fig.add_subplot(gs1[11])
e2 = fig.add_subplot(gs1[12])
f2 = fig.add_subplot(gs1[13])
g2 = fig.add_subplot(gs1[14])
h2 = fig.add_subplot(gs1[15])

gs2 = gridspec.GridSpec(1,8,left=0.048,bottom=0.09,top=0.33,right=0.93,hspace=0.1,wspace=0.1)
a3 = fig.add_subplot(gs2[0])
b3 = fig.add_subplot(gs2[1])
c3 = fig.add_subplot(gs2[2])
d3 = fig.add_subplot(gs2[3])
e3 = fig.add_subplot(gs2[4])
f3 = fig.add_subplot(gs2[5])
g3 = fig.add_subplot(gs2[6])
h3 = fig.add_subplot(gs2[7])

caxspec = gridspec.GridSpec(1,1,left=0.94,bottom=0.7,top=0.95,right=0.95)
cax = fig.add_subplot(caxspec[0])

nv_type_list = ['extras', 'g_all_everywhere', 'g_012_everywhere', 'g_048_everywhere', 'g_678_everywhere', 'ind_048_behaviour', 'ind_048_euclidean', 'two_regime']
titles = ['Extras', 'All comb.', '[012] comb.', '[048] comb.', '[678] comb.', '[048] behaviour', '[048] euclidean', 'Two regime']
ens_select = [None, np.asarray([2,8,10,13,14,16,17,18,19]), np.asarray([10,13,17]), np.asarray([17,2,18]), np.asarray([14,16,18]), np.asarray([17,2,18]), np.asarray([17,2,18]), np.asarray([2,13])]
top_row = [a1, b1, c1, d1, e1, f1, g1, h1]
bottom_row = [a2,b2,c2,d2,e2,f2,g2,h2]
gs1_axes = [a3, b3, c3, d3, e3, f3, g3, h3]

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.2, 0.2, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
gs1_axes[-1].legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.png")
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.pdf")
plt.show()
plt.close()
'''
'''
calc = 'cloud_frac'
data_path = '../data_cloud_frac/dycoms_data_low_nd_cloud_frac_mean.csv'
cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/tokyo/tokyo.txt")
cmap = colors.LinearSegmentedColormap.from_list('tokyo', cm_data)
cmap = cmap.reversed()
gen_type = 'main'
calc_type = 'mean'
multipliers = ['max']*8
normal_res = True
output_label = r'CF'
diverging = False
climited = True
ylim = [0,1.15]
hist_lim_x = (-0.1, 0.1)
hist_lim_x = (-0.08, 0.08)
hist_lim_y = (0,25)
hist_lim_y = (0,40)
bin_size = 0.01

fig = plt.figure(figsize=(24,9))
gs1 = gridspec.GridSpec(2,8,left=0.048,bottom=0.42,top=0.95,right=0.93,hspace=0.1,wspace=0.1) # 4 cols 
a1 = fig.add_subplot(gs1[0])
b1 = fig.add_subplot(gs1[1])
c1 = fig.add_subplot(gs1[2])
d1 = fig.add_subplot(gs1[3])
e1 = fig.add_subplot(gs1[4])
f1 = fig.add_subplot(gs1[5])
g1 = fig.add_subplot(gs1[6])
h1 = fig.add_subplot(gs1[7])
    
a2 = fig.add_subplot(gs1[8])
b2 = fig.add_subplot(gs1[9])
c2 = fig.add_subplot(gs1[10])
d2 = fig.add_subplot(gs1[11])
e2 = fig.add_subplot(gs1[12])
f2 = fig.add_subplot(gs1[13])
g2 = fig.add_subplot(gs1[14])
h2 = fig.add_subplot(gs1[15])

gs2 = gridspec.GridSpec(1,8,left=0.048,bottom=0.09,top=0.33,right=0.93,hspace=0.1,wspace=0.1)
a3 = fig.add_subplot(gs2[0])
b3 = fig.add_subplot(gs2[1])
c3 = fig.add_subplot(gs2[2])
d3 = fig.add_subplot(gs2[3])
e3 = fig.add_subplot(gs2[4])
f3 = fig.add_subplot(gs2[5])
g3 = fig.add_subplot(gs2[6])
h3 = fig.add_subplot(gs2[7])

caxspec = gridspec.GridSpec(1,1,left=0.94,bottom=0.7,top=0.95,right=0.95)
cax = fig.add_subplot(caxspec[0])

nv_type_list = ['extras', 'g_all_everywhere', 'g_012_everywhere', 'g_048_everywhere', 'g_678_everywhere', 'ind_048_behaviour', 'ind_048_euclidean', 'two_regime']
titles = ['Extras', 'All comb.', '[012] comb.', '[048] comb.', '[678] comb.', '[048] behaviour', '[048] euclidean', 'Two regime']
ens_select = [None, np.asarray([2,8,10,13,14,16,17,18,19]), np.asarray([10,13,17]), np.asarray([17,2,18]), np.asarray([14,16,18]), np.asarray([17,2,18]), np.asarray([17,2,18]), np.asarray([2,13])]
top_row = [a1, b1, c1, d1, e1, f1, g1, h1]
bottom_row = [a2,b2,c2,d2,e2,f2,g2,h2]
gs1_axes = [a3, b3, c3, d3, e3, f3, g3, h3]

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.2, 0.2, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
gs1_axes[-1].legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)
#fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.png")
#fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.pdf")
plt.show()
plt.close()
'''
'''
calc = 'cloud_frac'
cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()
data_path = '../data_cloud_frac/dycoms_data_low_nd_cloud_frac_teme.csv'
gen_type = 'tend'
calc_type = 'teme'
multipliers = ['none']*8
normal_res = False
output_label = r'CF tendency (hr$^{-1}$)'
diverging = True
climited = False
ylim = [-0.055,0.03]
hist_lim_x = (-1, 1)
hist_lim_x = (-0.01, 0.01)
hist_lim_y = (0,2.5)
hist_lim_y = (0,200)
bin_size = 0.01

fig = plt.figure(figsize=(24,9))
gs1 = gridspec.GridSpec(2,8,left=0.048,bottom=0.42,top=0.95,right=0.93,hspace=0.1,wspace=0.1) # 4 cols 
a1 = fig.add_subplot(gs1[0])
b1 = fig.add_subplot(gs1[1])
c1 = fig.add_subplot(gs1[2])
d1 = fig.add_subplot(gs1[3])
e1 = fig.add_subplot(gs1[4])
f1 = fig.add_subplot(gs1[5])
g1 = fig.add_subplot(gs1[6])
h1 = fig.add_subplot(gs1[7])
    
a2 = fig.add_subplot(gs1[8])
b2 = fig.add_subplot(gs1[9])
c2 = fig.add_subplot(gs1[10])
d2 = fig.add_subplot(gs1[11])
e2 = fig.add_subplot(gs1[12])
f2 = fig.add_subplot(gs1[13])
g2 = fig.add_subplot(gs1[14])
h2 = fig.add_subplot(gs1[15])

gs2 = gridspec.GridSpec(1,8,left=0.048,bottom=0.09,top=0.33,right=0.93,hspace=0.1,wspace=0.1)
a3 = fig.add_subplot(gs2[0])
b3 = fig.add_subplot(gs2[1])
c3 = fig.add_subplot(gs2[2])
d3 = fig.add_subplot(gs2[3])
e3 = fig.add_subplot(gs2[4])
f3 = fig.add_subplot(gs2[5])
g3 = fig.add_subplot(gs2[6])
h3 = fig.add_subplot(gs2[7])

caxspec = gridspec.GridSpec(1,1,left=0.94,bottom=0.7,top=0.95,right=0.95)
cax = fig.add_subplot(caxspec[0])

nv_type_list = ['extras', 'g_all_everywhere', 'g_012_everywhere', 'g_048_everywhere', 'g_678_everywhere', 'ind_048_behaviour', 'ind_048_euclidean', 'two_regime']
titles = ['Extras', 'All comb.', '[012] comb.', '[048] comb.', '[678] comb.', '[048] behaviour', '[048] euclidean', 'Two regime']
ens_select = [None, np.asarray([2,8,10,13,14,16,17,18,19]), np.asarray([10,13,17]), np.asarray([17,2,18]), np.asarray([14,16,18]), np.asarray([17,2,18]), np.asarray([17,2,18]), np.asarray([2,13])]
top_row = [a1, b1, c1, d1, e1, f1, g1, h1]
bottom_row = [a2,b2,c2,d2,e2,f2,g2,h2]
gs1_axes = [a3, b3, c3, d3, e3, f3, g3, h3]

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.2, 0.2, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
gs1_axes[-1].legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.png")
fig.savefig(f"../figures/normal_res_{multipliers[0]}_multiplier_{calc}_{calc_type}_linear_random_draws_20.pdf")
plt.show()
plt.close()
'''

### LWP emulator ###

calc = 'lwp_cloud'
data_path = '../data_lwp_cloud/dycoms_data_low_nd_lwp_cloud_mean.csv'
gen_type = 'main'
calc_type = 'mean'
normal_res = True
multipliers = ['', 'individual', 'mean', 'max', '', '']
cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/tokyo/tokyo.txt")
cmap = colors.LinearSegmentedColormap.from_list('tokyo', cm_data)
cmap = cmap.reversed()
output_label = r'L (g m$^{-2}$)'
diverging = False
climited = True
ylim = [-25, 195]
hist_lim_x = (-0.2,0.2)
hist_lim_y = (0,22)
bin_size = 0.005

fig = plt.figure(figsize=(18,9))
gs1 = gridspec.GridSpec(2,6,left=0.06,bottom=0.43,top=0.95,right=0.887,hspace=0.1,wspace=0.13)
a1 = fig.add_subplot(gs1[0])
b1 = fig.add_subplot(gs1[1])
c1 = fig.add_subplot(gs1[2])
d1 = fig.add_subplot(gs1[3])
e1 = fig.add_subplot(gs1[4])
f1 = fig.add_subplot(gs1[5])
    
a2 = fig.add_subplot(gs1[6])
b2 = fig.add_subplot(gs1[7])
c2 = fig.add_subplot(gs1[8])
d2 = fig.add_subplot(gs1[9])
e2 = fig.add_subplot(gs1[10])
f2 = fig.add_subplot(gs1[11])

gs2 = gridspec.GridSpec(1,6,left=0.06,bottom=0.1,top=0.34,right=0.887,hspace=0.1,wspace=0.13)
a3 = fig.add_subplot(gs2[0])
b3 = fig.add_subplot(gs2[1])
c3 = fig.add_subplot(gs2[2])
d3 = fig.add_subplot(gs2[3])
e3 = fig.add_subplot(gs2[4])
f3 = fig.add_subplot(gs2[5])

caxspec = gridspec.GridSpec(1,1,left=0.9,bottom=0.7,top=0.95,right=0.915)
cax = fig.add_subplot(caxspec[0])

nv_type_list = ['extras', 'g_048_everywhere', 'g_048_everywhere', 'g_048_everywhere', '1mag', '2mag']
titles = ['a) exact', 'b) proportional', 'c) mean', 'd) max', r'e) $\times 10$', r'f) $\times 100$']
ens_select = [None,  np.asarray([17,2,18]),  np.asarray([17,2,18]),  np.asarray([17,2,18]), None, None]
top_row = [a1, b1, c1, d1, e1, f1]
bottom_row = [a2, b2, c2, d2, e2, f2]
gs1_axes = [a3, b3, c3, d3, e3, f3]

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.2, 0.2, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
gs1_axes[-1].legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)
#fig.savefig(f"../figures/normal_res_all_multipliers_large_edit.png")
#fig.savefig(f"../figures/normal_res_all_multipliers_large_edit.pdf")
plt.show()
plt.close()


### other emulators ###
'''
fig = plt.figure(figsize=(18,9))
gs1 = gridspec.GridSpec(1,3,left=0.04,bottom=0.43,top=0.95,right=0.89,hspace=0.1,wspace=0.19)
gs1_1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[0], hspace=0.1,wspace=0.025)
a1 = fig.add_subplot(gs1_1[0,0])
b1 = fig.add_subplot(gs1_1[0,1])

gs1_2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[1], hspace=0.1,wspace=0.025)
c1 = fig.add_subplot(gs1_2[0,0])
d1 = fig.add_subplot(gs1_2[0,1])

gs1_3= gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[2], hspace=0.1,wspace=0.025)
e1 = fig.add_subplot(gs1_3[0,0])
f1 = fig.add_subplot(gs1_3[0,1])

a2 = fig.add_subplot(gs1_1[2])
b2 = fig.add_subplot(gs1_1[3])
c2 = fig.add_subplot(gs1_2[2])
d2 = fig.add_subplot(gs1_2[3])
e2 = fig.add_subplot(gs1_3[2])
f2 = fig.add_subplot(gs1_3[3])

gs2 = gridspec.GridSpec(1,3,left=0.04,bottom=0.1,top=0.34,right=0.89,hspace=0.1,wspace=0.19)
gs2_1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[0],wspace=0.025)
a3 = fig.add_subplot(gs2_1[0])
b3 = fig.add_subplot(gs2_1[1])
gs2_2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[1],wspace=0.025)
c3 = fig.add_subplot(gs2_2[0])
d3 = fig.add_subplot(gs2_2[1])
gs2_3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[2],wspace=0.025)
e3 = fig.add_subplot(gs2_3[0])
f3 = fig.add_subplot(gs2_3[1])

#cax = gridspec.GridSpec(1,1,left=0.9,bottom=0.7,top=0.95,right=0.915)
#cax = fig.add_subplot(caxspec[0])

multipliers = ['', 'none']

nv_type_list = ['extras', 'g_048_everywhere']
#nv_type_list = ['extras', 'g_all_everywhere', 'g_012_everywhere', 'g_048_everywhere', 'g_678_everywhere', 'ind_048_behaviour', 'ind_048_euclidean', 'two_regime']
titles = ['Exact', 'Nugget']
ens_select = 2*[None]
top_row = [a1, b1]
bottom_row = [a2, b2]
gs1_axes = [a3, b3]

cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()

calc = 'lwp_cloud'
data_path = '../data_lwp_cloud/dycoms_data_low_nd_lwp_cloud_teme.csv'
gen_type = 'tend'
calc_type = 'teme'
output_label = r''
diverging = True
climited = False
ylim = [-7,24]
hist_lim_x = (-1.5, 1.5)
hist_lim_x = (-2, 2)
hist_lim_y = (0,3)
bin_size = 0.01

multipliers = ['', 'none']
nv_type_list = ['extras', 'g_all_everywhere']
normal_res = False

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.9, 0.9, normal_res)

top_row = [c1, d1]
bottom_row = [c2, d2]
gs1_axes = [c3, d3]

cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/tokyo/tokyo.txt")
cmap = colors.LinearSegmentedColormap.from_list('tokyo', cm_data)
cmap = cmap.reversed()

calc = 'cloud_frac'
data_path = '../data_cloud_frac/dycoms_data_low_nd_cloud_frac_mean.csv'
gen_type = 'main'
calc_type = 'mean'
output_label = r'CF'
diverging = False
climited = True
ylim = [0,1.15]
hist_lim_x = (-0.1, 0.1)
hist_lim_x = (-0.08, 0.08)
hist_lim_y = (0,25)
hist_lim_y = (0,40)
bin_size = 0.01

multipliers = ['', 'max']
nv_type_list = ['extras', 'g_all_everywhere']
normal_res = True

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.9, 0.9, normal_res)

top_row = [e1, f1]
bottom_row = [e2, f2]
gs1_axes = [e3, f3]

cm_data =np.loadtxt("../../colour_maps/ScientificColourMaps6/cork/cork.txt")
cmap = colors.LinearSegmentedColormap.from_list('cork', cm_data)
cmap = cmap.reversed()

calc = 'cloud_frac'
data_path = '../data_cloud_frac/dycoms_data_low_nd_cloud_frac_teme.csv'
gen_type = 'tend'
calc_type = 'teme'
output_label = r''
diverging = True
climited = False
ylim = [-0.055,0.03]
hist_lim_x = (-1, 1)
hist_lim_x = (-0.01, 0.01)
hist_lim_y = (0,2.5)
hist_lim_y = (0,200)
bin_size = 0.01

multipliers = ['', 'none']
nv_type_list = ['extras', 'g_all_everywhere']
normal_res = False

small_tran, small_cb, ax_3, fig = nugget_figure(top_row, bottom_row, gs1_axes, cax, nv_type_list, titles, ens_select, multipliers, data_path, calc, gen_type, calc_type, cmap, diverging, output_label, ylim, bin_size, hist_lim_x, hist_lim_y, -0.9, 0.9, normal_res)
bottom_row[-1].legend([small_tran, small_cb], ["Mean", "95\% bounds"], loc=(1.001, 0), handlelength=0.75, frameon=False,fontsize=legendfont,handletextpad=0.13)
ax_3.legend(loc=(1.001, 0), handlelength=0.75, frameon=False, fontsize=legendfont,handletextpad=0.13)

a1.yaxis.set_visible(False)
c1.yaxis.set_visible(False)
e1.yaxis.set_visible(False)

fig.text(0.012,0.55, "Output value", ha='center', va='center',rotation='vertical',fontsize=font)
fig.text(0.475,0.365, r'$\Delta\theta$ (K)', ha='center', va='center',fontsize=font)
fig.text(0.012,0.85, r'$\Delta$ q$_{t}$ (g kg$^{-1}$)', ha='center', va='center',rotation='vertical',fontsize=font)
fig.text(0.475,0.03, 'Residual Value', ha='center', va='center',fontsize=font)
fig.text(0.012,0.2, 'Frequency', ha='center', va='center',rotation='vertical',fontsize=font)

for a,l in zip([a1,c1,e1], ['a)','b)','c)']):
    a.text(0.01,1.05,l,transform=a.transAxes,fontsize=font)

fig.savefig(f"../figures/other_emulators_nugget_check_mixed_normal_edit.png")
fig.savefig(f"../figures/other_emulators_nugget_check_mixed_normal_edit.pdf")
plt.show()
plt.close()
'''
