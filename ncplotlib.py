# Library of functions for plotting straight from netCDF files
import matplotlib.pyplot as plt

def scalar(fig, axes, var):
    # Plot scalar diagnostics at each simulation time
    var.plot(ax=axes)
    plt.title(var.name, fontsize=12)
    return fig


def profile(fig, axes, var):
    # Plot profile diagnostics - through height and at each simulation time
    var_T = var.transpose()
    var_T.plot(ax=axes)
    plt.title(var.name, fontsize=12)
    return fig


def scene(var, savepath):
    # Plot scene diagnostics - x and y domain lengths averaged over height at time snapshots. For short simulations, quick plotting
    t = var.dims[0] # time series
    var.plot(x='x', y='y', col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')

    
def scene_long(var, num_plots, savepath):
    # Plot scene diagnostics - x and y domain lengths averaged over simulation time
    t = var.dims[0] # time series
    tlast = (var.sizes[t]-1)
    var.isel(**{t:slice(0, tlast, int(tlast/num_plots))}).plot(x='x',y='y', col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')


def vslice(var, depth, horizontal_dim, savepath):
    # Plot vertical slice through 4D diagnostics - x or y vs height at time snapshots. For SHORT simulations, quick plotting
    t = var.dims[0] # time series
    z = var.dims[3]
    if horizontal_dim == 'x':
        var[:,depth,:,:].plot(x='y', y=z, col=t, col_wrap=5)
    elif horizontal_dim == 'y':
        var[:,:,depth,:].plot(x='x', y=z, col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')


def vslice_long(var, horizontal_dim, depth, num_plots, savepath):
    # Plot vertical slice through 4D diagnostics - x or y vs height at time snapshots
    t = var.dims[0] # time series
    tlast = (var.sizes[t]-1)
    z = var.dims[3]
    if horizontal_dim == 'x':
        var[:,depth,:,:].isel(**{t:slice(0, tlast, int(tlast/num_plots))}).plot(x='y', y=z, col=t, col_wrap=5)
    elif horizontal_dim == 'y':
        var[:,:,depth,:].isel(**{t:slice(0, tlast, int(tlast/num_plots))}).plot(x='x', y=z, col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + 'long.png')


def hslice(var, height, savepath):
    # Plot horizontal slice through 4D diagnostics - x vs y at specific height at time snapshots. For SHORT simulations, quick plotting.
    t = var.dims[0] # time series
    if var.dims[3] == 'z':
        var.sel(z=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=5)
    else:
        var.sel(zn=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + '.png')


def hslice_long(var, height, num_plots, savepath):
    # Plot horizontal slice through 4D diagnostics - x vs y at specific height at time snapshots
    t = var.dims[0] # time series
    tlast = (var.sizes[t]-1)
    if var.dims[3] == 'z':
        var.isel(**{t:slice(0, tlast, int(tlast/num_plots))}).sel(z=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=5)
    else:
        var.isel(**{t:slice(0, tlast, int(tlast/num_plots))}).sel(zn=height, method='nearest').plot(x='x', y='y', col=t, col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.savefig(savepath + var.name + 'long.png')
