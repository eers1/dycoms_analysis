# Library of functions for plotting straight from netCDF files
import matplotlib.pyplot as plt

def scalar(var):
    # Plot scalar diagnostics at each simulation time
    var.plot()
    plt.title(var.name, fontsize=12)
    plt.show()


def profile(var):
    # Plot profile diagnostics - through height and at each simulation time
    var_T = var.transpose()
    var_T.plot()
    plt.title(var.name, fontsize=12)
    plt.show()

    
def scene(var, num_plots):
    # Plot scene diagnostics - x and y domain lengths averaged over simulation time
    t = var.dims[0] # time series
    tlast = (var.size[t]-1)
    var.isel(**{t:slice(0,tlast,int(tlast/num_plots))}).plot(x='x',y='y',col=t,col_wrap=5)
    plt.title(var.name, fontsize=12)
    plt.show()


def vslice(var, horizontal_dim, depth):
    # Plot vertical slice through 4D diagnostics - x or y vs height at time snapshots
    time = var.dims[0]
    z = var.dims[3]
    if horizontal_dim == 'x':
        var[:,depth,:,:].plot(x='y', y=z, col=time)
    elif horizontal_dim == 'y':
        var[:,:,depth,:].plot(x='x', y=z, col=time)
    plt.title(var.name, fontsize=12)
    plt.show()


def hslice(var, height):
    # Plot horizontal slice through 4D diagnostics - x vs y at specific height at time snapshots
    time = var.dims[0]
    if var.dims[3] == 'z':
        var.sel(z=height, method='nearest').plot(x='x', y='y', col=time)
    else:
        var.sel(zn=height, method='nearest').plot(x='x', y='y', col=time)
    plt.title(var.name, fontsize=12)
    plt.show()
