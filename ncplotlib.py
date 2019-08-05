# Library of functions for plotting straight from netCDF files
import matplotlib.pyplot as plt

# Plot scalar diagnostics at each simulation time
def scalar(var):
    var.plot()
    plt.title(var.name, fontsize=12)
    plt.show()


# Plot profile diagnostics - through height and at each simulation time
def profile(var):
    var_T = var.transpose()
    var_T.plot()
    plt.title(var.name, fontsize=12)
    plt.show()


# Plot scene diagnostics - x and y domain lengths averaged over simulation time
def scene(var):
    var.plot(x='x', y='y', col=var.dims[0])
    plt.title(var.name, fontsize=12)
    plt.show()


# Plot vertical slice through 4D diagnostics - x or y vs height at time snapshots
def vslice(var, horizontal_dim, depth):
    time = var.dims[0]
    z = var.dims[3]
    if horizontal_dim == 'x':
        var[:,depth,:,:].plot(x='y', y=z, col=time)
    elif horizontal_dim == 'y':
        var[:,:,depth,:].plot(x='x', y=z, col=time)
    plt.title(var.name, fontsize=12)
    plt.show()


# Plot horizontal slice through 4D diagnostics - x vs y at specific height at time snapshots
def hslice(var, height):
    time = var.dims[0]
    if var.dims[3] == 'z':
        var.sel(z=height, method='nearest').plot(x='x', y='y', col=time)
    else:
        var.sel(zn=height, method='nearest').plot(x='x', y='y', col=time)
    plt.title(var.name, fontsize=12)
    plt.show()
