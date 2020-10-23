'''
Library of functions for plotting straight from netCDF files
'''
from matplotlib import rc
import matplotlib.pyplot as plt

rc('font', size=15)

def scalar(fig, axes, var):
    '''    
    Plot scalar diagnostics at each simulation time
    '''
    # extra for bad value in oat1
#    good = var.where(var<10000, drop=True)
#    good.plot(ax=axes)
    var.plot(ax=axes)
    plt.title(var.name, fontsize=12)
    return fig


def profile(fig, axes, var):
    '''
    Plot profile diagnostics - through height and at each simulation time
    '''
    var_T = var.transpose()
    var_T.plot(ax=axes)
    plt.title(var.name, fontsize=12)
    return fig


def scene(var, savepath):
    ''' 
    Plot scene diagnostics - x and y domain lengths averaged over height at time snapshots.
    '''
#    print(var.name)
    t = var.dims[0] # time series
#    tlast = var.sizes[t] - 1  # change to 4 to avoid bad values at end times
    if var.sizes[t] > 9:
        g = var.isel(**{t:slice(0, (var.sizes[t]-1), int(var.sizes[t]/9))}).plot(x='x',y='y',
		col=t, col_wrap=3)
    else:
#        g = var.plot(x='x', y='y', col=t, col_wrap=3)
        g = var.plot(x='x', y='y', col=t, col_wrap=3)

    for i, ax in enumerate(g.axes.flat):
        ax.set_title('Time = %d' % int(var[t][i]) + 's')

    #plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')
#    plt.show()
    plt.close()


def vslice(var, depth, horizontal_dim, savepath):
    '''
    Plot vertical slice through 4D diagnostics - x or y vs height at time snapshots. For SHORT simulations, quick plotting
    '''
#    print(var.name)
    t = var.dims[0] # time series
    z = var.dims[3]
#    tlast = (var.sizes[t]-1)
    if var.sizes[t] > 9:
        if horizontal_dim == 'x':
            g = var[:,depth,:,:].isel(**{t:slice(0, (var.sizes[t]-1), int(var.sizes[t]/9))}).plot(x='y', y=z, col=t, col_wrap=3)
        elif horizontal_dim == 'y':
            g = var[:,:,depth,:].isel(**{t:slice(0, (var.sizes[t]-1), int(var.sizes[t]/9))}).plot(x='x', y=z, col=t, col_wrap=3)
    else:
        if horizontal_dim == 'x':
            g = var[:,depth,:,:].plot(x='y', y=z, col=t, col_wrap=3)
        elif horizontal_dim == 'y':
            g = var[:,:,depth,:].plot(x='x', y=z, col=t, col_wrap=3)
    for i, ax in enumerate(g.axes.flat):
        ax.set_title('Time = %d' % int(var[t][i]) + 's')
#    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')
#    plt.show()
    plt.close()


def hslice(var, height, savepath):
    '''
    Plot horizontal slice through 4D diagnostics - x vs y at specific height at time snapshots. For SHORT simulations, quick plotting.
    '''
    t = var.dims[0] # time series
    tlast = (var.sizes[t]-1)
    if var.sizes[t] > 5:
        if var.dims[3] == 'z':
            g = var.isel(**{t:slice(0, tlast, int(tlast/5))}).sel(z=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=3)
        else:
            g = var.isel(**{t:slice(0, tlast, int(tlast/5))}).sel(zn=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=3)
    else:
        if var.dims[3] == 'z':
            g = var.sel(z=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=3)
        else:
            g = var.sel(zn=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=3)
    for i, ax in enumerate(g.axes.flat):
        ax.set_title('Time = %d' % int(var[t][i]) + 's')
#    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')
#    plt.show()
    plt.close()

