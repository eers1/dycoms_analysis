#!usr/bin/env python3.7

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statistics as stat
from matplotlib import rc
from matplotlib import rcParams

font = 25
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': font})

def load_files(calc, output_type):
    ensembles=np.loadtxt(f'../ensembles_csv/ensemble_{calc}_mean.csv',delimiter=',') # this is actually both mean and tend
    output_list = ensembles[:,0] if (output_type=='mean') else ensembles[:,1]
    
    data=np.loadtxt(f'../data_{calc}/dycoms_data_low_nd_{calc}_{output_type}.csv', delimiter=',')
    ppe=data[:,2]
    ppe_th=data[:,0]
    ppe_qt=data[:,1] 
    return output_list, ppe, ppe_th, ppe_qt

def put_in_array(values):
    array = np.reshape(values, [int(len(values)/5),5])   
    return array

def order_list(named_list, ensemble_size):
    ordered = [sorted(named_list[i:i+ensemble_size]) for i in range(0,len(named_list)-ensemble_size,ensemble_size)]
    return ordered

def order_ensemble_indices(named_array, ppe_ens_no):
    ordered_ens_ind=ppe_ens_no[named_array[:,0].argsort()]-1
    return ordered_ens_ind

def sample_var(array):
    mean = np.mean(array)
    deviations = [(element - mean)**2 for element in array]
    variance = np.sum(deviations)/(len(array)-1)
    return variance

def create_cells(multiplier, ppe_th, ppe_qt, ppe_output, ensemble_indices, sample_var, application):
    '''
    Creates cells by either similar behaviour (LWP value) or by Euclidean distance. Allocates each training data point, xi, to an ensemble point, zi, and gives xi the variance from that zi * the value of the training data squared. 
    '''
    centre_tups = [(ppe_th[i], ppe_qt[i]) for i in ensemble_indices]
    centre_behav = [ppe_output[i] for i in ensemble_indices]
    ppe_dict_exact=dict.fromkeys(range(len(ppe_th)), [])
    for i in range(len(ppe_th)):
        if application=='euclidean':
            point_ind = np.argmin([euc_distance((ppe_th[i], ppe_qt[i]), j) for j in centre_tups])
        elif application=='behaviour':
            point_ind = np.argmin([abs(j-ppe_output[i]) for j in centre_behav])
        else:
            print('Error: application must be specified as either euclidean or behaviour for creating cells')

        if multiplier=="individual":
            ppe_dict_exact[i]=sample_var[point_ind]*(ppe_output[i]**2)
        elif multiplier=="mean":
            ppe_dict_exact[i]=sample_var[point_ind]*(np.nanmean(ppe_output)**2) # mean of whole 20 point ensemble
        elif multiplier=="max":
            ppe_dict_exact[i]=sample_var[point_ind]*(np.nanmax(ppe_output)**2)  # max of whole 20 point ensemble
        elif multiplier=="none":
            ppe_dict_exact[i]=sample_var[point_ind]
        else:
            print("Please select valid multiplier")
    return ppe_dict_exact
    
def euc_distance(tup1, tup2):
    a=((tup1[0]-tup2[0])**2+(tup1[1]-tup2[1])**2)**0.5
    return a

def sample_var_15to75(named_array):
    small_var = np.empty((len(named_array)), dtype=float)
    for i, row in enumerate(named_array):
        small_var[i] = sample_var_15to75_1d(row)
    return small_var

def sample_var_15to75_1d(array1d):
    a=array1d.min()
    b=array1d.max()
    variance=((b-a)/4)**2
    return variance

def residual_cell_variance(multiplier, ppe_th, ppe_qt, ppe_output, ensemble_indices, slices, residual_variance, application):
    if slices is not None:
        ensemble_indices=ensemble_indices[slices]
    ppe_dict_exact=create_cells(multiplier, ppe_th, ppe_qt, ppe_output, ensemble_indices, residual_variance, application)
    cell_variance_vector=[ppe_dict_exact[num] for num in ppe_dict_exact]
    return cell_variance_vector

def calculate_residuals_1d(array1d, normal_bool=True):
    mean = np.mean(array1d)
    if normal_bool==True:
        residuals = np.array([(sim - mean)/mean for sim in array1d])
    else:
        residuals = np.array([(sim - mean) for sim in array1d])
    #residuals = np.array([r-min(residuals)/(max(residuals)-min(residuals)) for r in residuals])
    return residuals

def calculate_residuals(named_array, grouped, slices=None, normal_bool=True):
    '''
    Calculated the residuals of an array of ensemble members. If grouped is True, it return the residuals as a 1d array. Otherwise it will be 2d with dimensions of the number of ensemble members and the number of samples per ensemble. Slices determines which ensemble members to include. When slices==None, all members are used. 
    '''
    residuals = np.empty(np.shape(named_array), dtype=float)
    for i, row in enumerate(named_array):
        residuals[i] = calculate_residuals_1d(row, normal_bool)
        
    if slices is not None:
        residuals=residuals[slices]
    if grouped is True:
        residuals=residuals.reshape(np.size(residuals))
    return residuals

def calculate_residual_variance(named_ordered_array, grouped, slices=None, normal_bool=True):
    '''
    Calculates the residuals for an array of ensemble members (see calculate_residuals for info on grouped and slices). 
    '''
    residuals = calculate_residuals(named_ordered_array, grouped, slices, normal_bool)
    residual_variance = variance(residuals)
    return residual_variance

def variance(sample):
    if len(np.shape(sample))>1:
        variance = np.empty((len(sample)))
        for i, row in enumerate(sample):
            variance[i] = sample_var_15to75_1d(row)
    else:
        variance = sample_var_15to75_1d(sample)
    return variance

def ensemble_to_variance(output_type, vector_name, multiplier, named_list, grouped, ppe_th, ppe_qt, ppe_output, ppe_ens_no, var_dict, application, slices=None, normal_bool=True):
    array=put_in_array(named_list)
    ordered_array=array[array[:,0].argsort()]
    ensemble_indices=order_ensemble_indices(array, ppe_ens_no)
    print(ensemble_indices)

    residual_variance=calculate_residual_variance(ordered_array, grouped, slices, normal_bool)
    print(f"Slice {slices}: {residual_variance}")
    if application=='euclidean':
        residual_variance_vector=residual_cell_variance(multiplier, ppe_th, ppe_qt, ppe_output, ensemble_indices, slices, residual_variance, application)
    elif application=='behaviour':
        residual_variance_vector=residual_cell_variance(multiplier, ppe_th, ppe_qt, ppe_output, ensemble_indices, slices, residual_variance, application)
    elif application=='everywhere':
        if multiplier=="individual":
            residual_variance_vector=np.array([residual_variance]*len(ppe_output))*ppe_output**2
        elif multiplier=="mean":
            residual_variance_vector=np.array([residual_variance]*len(ppe_output))*(np.nanmean(ppe_output)**2)  # mean
        elif multiplier=="max":
            residual_variance_vector=np.array([residual_variance]*len(ppe_output))*(np.nanmax(ppe_output)**2)  # max
        elif multiplier=="none":
            residual_variance_vector=np.array([residual_variance]*len(ppe_output))
        else:
            print("Select valid multiplier")
    else:
        print('Error: application must be string selection of euclidean, behaviour or everywhere')
    var_dict[f'{output_type}_{multiplier}_{vector_name}']=residual_variance_vector
    return var_dict

def do_KS(named_list, grouped, slices=None, normal_bool=True):
    '''
    Kolmogorv-Smirnov test to check if samples are of the same distribution.
    '''    
    array=put_in_array(named_list)
    ordered_array=array[array[:,0].argsort()]

    residuals=calculate_residuals(ordered_array, grouped, None, normal_bool)

    kstats=[]
    d=[]
    for row in range(len(ordered_array)):
        for comp in range(len(ordered_array)):
            d_stat, pval=sts.ks_2samp(residuals[row], residuals[comp])
            kstats.append(pval)
            d.append(d_stat)
            if pval<0.1:
                print(f"KS test fail: {row}, {comp}")
    return kstats, d

def small_sample_mean_1d(array1d):
    '''
    Mean for small samples. Currently using the np.mean everywhere. 
    '''
    a = array1d.min()
    b = array1d.max()
    m = stat.median(array1d)
    n = len(array1d)
    m_bar = ((a+2*m+b)/4) + (a-2*m+b)/(4*n)
    return m_bar

def main(calc, output_type, savepath): 
    ensemble_list, ppe_output, ppe_th, ppe_qt = load_files(calc, output_type)
    #ensemble_list = ensemble_list+20
    #ppe_output = ppe_output+20
    ppe_ens_no=np.array([3,9,11,14,15,17,18,19,20])

    mean_output = np.nanmean(ppe_output)
    max_output = np.nanmax(ppe_output)
    print(f"PPE data: \n{ppe_output}")
    print(f"Ensemble list: \n{ensemble_list}")
    print(f"mean is {mean_output}")
    print(f"max is {max_output}")

    ### Replace the data values with the means from the ensembles. 
    #for i,ind in enumerate(ppe_ens_no):
    #    ppe_output[ind-1] = np.mean(ensemble_list[i*5:i*5+5])

    kstats, d = do_KS(ensemble_list,False)

    varper_dict={}

    #grouped=False
    #slices=[0,1,2,3,4,5,6,7,8]
    #application='euclidean'
    
    grouped=True
    slices=None
    application='everywhere'

    array = put_in_array(ensemble_list)
    ordered_array = array[array[:,0].argsort()]
    print(f"Ordered ensemble list: \n{ordered_array}")
    normal_bool = False # true for normal, false otherwise
    residuals1 = calculate_residuals(ordered_array, True, None, False)
    residuals2 = calculate_residuals(ordered_array, True, None, True)
    np.savetxt(f"{savepath}/residuals/design_relative_residuals_{output_type}.csv", residuals2.reshape(np.size(residuals2)))
    np.savetxt(f"{savepath}/residuals/design_residuals_{output_type}.csv", residuals1.reshape(np.size(residuals1)))
    print(residuals1)
    print(stat.stdev(residuals1))
    print(f"Mean of residuals: \n{np.mean(residuals1)}")
    print(f"Small mean of residuals: \n{small_sample_mean_1d(residuals1)}")
    print(residuals2)
    print(stat.stdev(residuals2))
    print(f"Mean of residuals: \n{np.mean(residuals2)}")
    print(f"Small mean of residuals: \n{small_sample_mean_1d(residuals2)}")

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7,14), sharex=True)
    plt.subplots_adjust(left=0.32,top=0.98,right=0.98,hspace=0.08)

    m_list = []
    for i,row in enumerate(ordered_array):
        mean = np.mean(row)
        m_list.append(mean)
        for j,val in enumerate(row):
            ax[0].scatter(mean, val, c='black')
            ax[1].scatter(mean, residuals1[5*i+j], c='black')
            ax[2].scatter(mean, residuals2[5*i+j], c='black')

    ax[2].set_xticks(m_list)
    ax[2].set_xticklabels([f'{val:0.0f}' for val in m_list], fontsize=10, rotation=90)
    ax[2].set_xlabel("Ensemble mean")
    ax[0].set_ylabel("Model values")
    ax[1].set_ylabel("Residuals")
    ax[2].set_ylabel("Normalised \nresiduals", horizontalalignment="center")
    rcParams['ytick.labelsize'] = 15
    fig.align_labels()
    for ax, letter in zip([ax[0], ax[1], ax[2]], ['a', 'b', 'c']):
        ax.text(0.03,0.87,f'{letter})',transform=ax.transAxes,fontsize=font)
    #plt.savefig(f"/home/rach/Emulator/ensemble_variances_{calc}_{output_type}.png")
    #plt.savefig(f"/home/rach/Emulator/ensemble_variances_{calc}_{output_type}.pdf")
    plt.show()

    print("Calculating residuals variances...")
    
    for multiplier in ["none"]: #["individual","mean","max"]:
        varper_dict=ensemble_to_variance(output_type, 'g_012_everywhere', multiplier, ensemble_list, True, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'everywhere', slices=[0,1,2], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'g_048_everywhere', multiplier, ensemble_list, True, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'everywhere', slices=[0,4,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'g_678_everywhere', multiplier, ensemble_list, True, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'everywhere', slices=[6,7,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'g_all_everywhere', multiplier, ensemble_list, True, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'everywhere', slices=None, normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'ind_048_behaviour', multiplier, ensemble_list, False, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'behaviour', slices=[0,4,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'ind_048_euclidean', multiplier, ensemble_list, False, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'euclidean', slices=[0,4,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'ind_behaviour', multiplier, ensemble_list, False, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'behaviour', slices=[0,1,2,3,4,5,6,7,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'ind_euclidean', multiplier, ensemble_list, False, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'euclidean', slices=[0,1,2,3,4,5,6,7,8], normal_bool=normal_bool)
        varper_dict=ensemble_to_variance(output_type, 'two_regime', multiplier, ensemble_list, False, ppe_th, ppe_qt, ppe_output, ppe_ens_no, varper_dict, 'euclidean', slices=[2,4], normal_bool=normal_bool)

        for k, v in varper_dict.items():
            #np.savetxt(f'{savepath}/nv_normal_res_{k}.csv', v, delimiter=',')
            np.savetxt(f'{savepath}/nv_res_{k}.csv', v, delimiter=',')

    max_g_048 = varper_dict[f'{output_type}_{multiplier}_g_048_everywhere'][0]

    print (varper_dict[f'{output_type}_{multiplier}_g_048_everywhere'])
    ## Only applies with edits on, not actually multiplying by the max
    #np.savetxt(f'{savepath}/nv_normal_res_{output_type}_unnormalised_residuals.csv',varper_dict[f'{output_type}_{multiplier}_g_048_everywhere'], delimiter=',')
    #np.savetxt(f'{savepath}/nv_normal_res_{output_type}_1mag.csv', [max_g_048*10]*len(ppe_output), delimiter=',')
    #np.savetxt(f'{savepath}/nv_normal_res_{output_type}_2mag.csv', [max_g_048*100]*len(ppe_output), delimiter=',')
    
    print(f"Whole dictionary: \n{varper_dict}")
    
    return ensemble_list


#lwp_mean_list = main('lwp_cloud', 'mean', '../noise_files/lwp_cloud')
lwp_tend_list = main('lwp_cloud', 'teme', '../noise_files/lwp_cloud/')
cf_mean_list = main('cloud_frac', 'mean', '../noise_files/cloud_frac')
cf_teme_list = main('cloud_frac', 'teme', '../noise_files/cloud_frac')


