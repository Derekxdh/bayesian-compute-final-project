import torch
import matplotlib.pyplot as plt
import numpy as np

# =======================================================================
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                 # Acceleration of gravity [m/s^2]
H = 100                # Depth of fluid [m]
f_0 = 1E-4              # Fixed part ofcoriolis parameter [1/s]
beta = 2E-11            # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0          # Density of fluid [kg/m^3)]
tau_0 = 0.1             # Amplitude of wind stress [kg/ms^2]
use_coriolis = True     # True if you want coriolis force
use_friction = False     # True if you want bottom friction
use_wind = False        # True if you want wind stress
use_beta = True         # True if you want variation in coriolis
use_source = False       # True if you want mass source into the domain
use_sink = False       # True if you want mass sink out of the domain
param_string = "\n================================================================"
param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
param_string += "\ng = {:g}\nH = {:g}".format(g, H)

N_x = 50 # Number of grid points in x-direction
N_y = 50 # Number of grid points in y-direction
dx = L_x/(N_x - 1) # Grid spacing in x-direction
dy = L_y/(N_y - 1) # Grid spacing in y-direction
dt = 0.1*min(dx, dy)/np.sqrt(g*H) # Time step (defined from the CFL condition)
time_step = 1 # For counting time loop steps
max_time_step = 5000 # Total number of time steps in simulation
x = np.linspace(-L_x/2, L_x/2, N_x) # Array with x-points
y = np.linspace(-L_y/2, L_y/2, N_y) # Array with y-points
X, Y = np.meshgrid(x, y) # Meshgrid for plotting
X = np.transpose(X) # To get plots right
Y = np.transpose(Y) # To get plots right
param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(dx, dy, dt)
# =======================================================================

def plot_snapshot(X, Y, eta, u, v, step, dt, suffix):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.contourf(X/1e3, Y/1e3, eta[0], levels=50, cmap='coolwarm')
    plt.colorbar()
    plt.title(f'EnKF surface at step {step}, t = {10*step*dt:.2f}s')

    plt.subplot(2, 2, 2)
    plt.quiver(X/1e3, Y/1e3, u[0]/1e1, v[0]/1e1)
    plt.title('EnKF Velocity Field')

    plt.subplot(2, 2, 3)
    plt.contourf(X/1e3, Y/1e3, eta[1], levels=50, cmap='coolwarm')
    plt.colorbar()
    plt.title(f'True surface at step {step}, t = {10*step*dt:.2f}s')

    plt.subplot(2, 2, 4)
    plt.quiver(X/1e3, Y/1e3, u[1]/1e1, v[1]/1e1)
    plt.title('True Velocity Field')

    plt.tight_layout()
    plt.savefig(f'./figs/snapshot_{step}_{suffix}.pdf')
    plt.close()

def plot_setting(filename, suffix):
    #filename = f'./enkf_{suffix}.pt'
    ensemble_list = torch.load(filename, map_location='cpu')

    mse_eta, mse_u, mse_v = 0, 0, 0
    for i, (ensemble, true_state) in enumerate(ensemble_list):
        eta = ensemble['eta'].mean(dim=0), true_state['eta']
        u = ensemble['u'].mean(dim=0), true_state['u']
        v = ensemble['v'].mean(dim=0), true_state['v']
        mse_eta += ((eta[0] - eta[1]) ** 2).mean()
        mse_u += ((u[0] - u[1]) ** 2).mean()
        mse_v += ((v[0] - v[1]) ** 2).mean()
        if i == 9:
            plot_snapshot(X, Y, eta, u, v, 10 * i, dt, suffix)
    mse_eta /= len(ensemble_list)
    mse_u /= len(ensemble_list)
    mse_v /= len(ensemble_list)
    return mse_eta, mse_u, mse_v

def plot_smc(filename, suffix):
    ensemble_list = torch.load(filename, map_location='cpu')
    print(len(ensemble_list))
    print(ensemble_list[0][0]['eta'].shape)
    print(ensemble_list[0][2]['eta'].shape)
    print(ensemble_list[0][1].shape)
    print('-------')

    mse_eta, mse_u, mse_v = 0, 0, 0
    for i, (ensemble, weight, true_state) in enumerate(ensemble_list):
        eta = (ensemble['eta']*weight[:, None, None]).sum(dim=0), true_state['eta']
        u = (ensemble['u']*weight[:, None, None]).sum(dim=0), true_state['u']
        v = (ensemble['v']*weight[:, None, None]).sum(dim=0), true_state['v']
        mse_eta += ((eta[0] - eta[1]) ** 2).mean()
        mse_u += ((u[0] - u[1]) ** 2).mean()
        mse_v += ((v[0] - v[1]) ** 2).mean()
        plot_snapshot(X, Y, eta, u, v, 10 * i, dt, suffix)
    mse_eta /= len(ensemble_list)
    mse_u /= len(ensemble_list)
    mse_v /= len(ensemble_list)
    return mse_eta, mse_u, mse_v

ns = [10, 20, 30, 40, 50, 60, 70, 80]
epss = [0.1, 0.2, 0.5, 1, 2]
def plot_vs_ensemble_size(eps):
    rmse_eta = []
    rmse_u = []
    rmse_v = []
    for n in ns:
        fn = f'./enkf_{n}_{eps}.pt'
        suffix = f'{n}_{eps}'
        re, ru, rv = plot_setting(fn, suffix)
        rmse_eta.append(re)
        rmse_u.append(ru)
        rmse_v.append(rv)
    plt.figure(figsize=(6, 4))
    plt.plot(ns, rmse_eta, 'o-', label=r"$\eta$")
    plt.plot(ns, rmse_u, 'o-', label=r"$u$")
    plt.plot(ns, rmse_v, 'o-', label=r"$v$")
    plt.title("Time averaged RMSE for different ensemble size")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"./figs/rmse_ens_{eps}.pdf")
    plt.close()

#plot_setting('./enkf_20_0.1.pt', '20_0.1')
#plot_vs_ensemble_size(0.1)
plot_smc("./SMC_100_0.1.pt", "smc_100")
