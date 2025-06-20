
import sys
import time
import numpy as np
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
import matplotlib.pyplot as plt
#import viz_tools

# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical prameters ---------------
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

# --------------- Computational prameters ---------------
N_x = 50                            # Number of grid points in x-direction
N_y = 50                            # Number of grid points in y-direction
dx = L_x/(N_x - 1)                   # Grid spacing in x-direction
dy = L_y/(N_y - 1)                   # Grid spacing in y-direction
dt = 0.1*min(dx, dy)/np.sqrt(g*H)    # Time step (defined from the CFL condition)
time_step = 1                        # For counting time loop steps
max_time_step = 5000                 # Total number of time steps in simulation
x = torch.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
y = torch.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
X, Y = torch.meshgrid(x, y)             # Meshgrid for plotting
X = X.T                  # To get plots right
Y = Y.T                  # To get plots right
param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(dx, dy, dt)

# Define friction array if friction is enabled.
if (use_friction is True):
    kappa_0 = 1/(5*24*3600)
    kappa = torch.ones((N_x, N_y))*kappa_0
    param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(kappa_0, kappa_0/(beta*1000))

# Define wind stress arrays if wind is enabled.
if (use_wind is True):
    tau_x = -tau_0*torch.cos(torch.pi*y/L_y)*0
    tau_y = torch.zeros((1, len(x)))
    param_string += "\ntau_0 = {:g}\nrho_0 = {:g} km".format(tau_0, rho_0)

# Define coriolis array if coriolis is enabled.
if (use_coriolis is True):
    if (use_beta is True):
        f = f_0 + beta*y        # Varying coriolis parameter
        L_R = np.sqrt(g*H)/f_0  # Rossby deformation radius
        c_R = beta*g*H/f_0**2   # Long Rossby wave speed
    else:
        f = f_0*torch.ones(len(y))                 # Constant coriolis parameter

    alpha = dt*f                # Parameter needed for coriolis scheme
    beta_c = alpha**2/4         # Parameter needed for coriolis scheme

    param_string += "\nf_0 = {:g}".format(f_0)
    param_string += "\nMax alpha = {:g}\n".format(alpha.max())
    param_string += "\nRossby radius: {:.1f} km".format(L_R/1000)
    param_string += "\nRossby number: {:g}".format(np.sqrt(g*H)/(f_0*L_x))
    param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(c_R)
    param_string += "\nLong Rossby transit time: {:.2f} days".format(L_x/(c_R*24*3600))
    param_string += "\n================================================================\n"

# Define source array if source is enabled.
if (use_source):
    sigma = torch.zeros((N_x, N_y))
    sigma = 0.0001*torch.exp(-((X-L_x/2)**2/(2*(1E+5)**2) + (Y-L_y/2)**2/(2*(1E+5)**2)))
    
# Define source array if source is enabled.
if (use_sink is True):
    w = torch.ones((N_x, N_y))*sigma.sum()/(N_x*N_y)

# Write all parameters out to file.
with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

#print(param_string)     # Also print parameters to screen
# ============================= Parameter stuff done ===============================

# |%%--%%| <WmT8C13pnE|3HWzhcf84v>

def swe_forward(eta0, u0, v0, iters):
    # ==================================================================================
    # ==================== Allocating arrays and initial conditions ====================
    # ==================================================================================
    u_n = u0.clone()
    v_n = v0.clone()
    u_torch1 = torch.zeros((N_x, N_y))    # To hold u at next time step
    v_torch1 = torch.zeros((N_x, N_y))    # To hold v at enxt time step
    eta_n = eta0.clone()
    eta_torch1 = torch.zeros((N_x, N_y))  # To hold eta at next time step

    # Temporary variables (each time step) for upwind scheme in eta equation
    h_e = torch.zeros((N_x, N_y))
    h_w = torch.zeros((N_x, N_y))
    h_n = torch.zeros((N_x, N_y))
    h_s = torch.zeros((N_x, N_y))
    uhwe = torch.zeros((N_x, N_y))
    vhns = torch.zeros((N_x, N_y))


    max_time_step = iters
    time_step = 1 

    #viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

    # Sampling variables.
    eta_list = list(); u_list = list(); v_list = list()         # Lists to contain eta and u,v for animation
    hm_sample = list(); ts_sample = list(); t_sample = list()   # Lists for Hovmuller and time series
    hm_sample.append(eta_n[:, int(N_y/2)])                      # Sample initial eta in middle of domain
    ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])             # Sample initial eta at center of domain
    t_sample.append(0.0)                                        # Add initial time to t-samples
    #anim_interval = 20                                         # How often to sample for time series
    #sample_interval = 1000                                      # How often to sample for time series
    # =============== Done with setting up arrays and initial conditions ===============

    # ==================================================================================
    # ========================= Main time loop for simulation ==========================
    # ==================================================================================
    while (time_step <= max_time_step):
        # ------------ Computing values for u and v at next time step --------------
        u_torch1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :])
        v_torch1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1])

        # Add friction if enabled.
        if (use_friction is True):
            u_torch1[:-1, :] =u_torch1[:-1, :] - dt*kappa[:-1, :]*u_n[:-1, :]
            v_torch1[:-1, :] =v_torch1[:-1, :] - dt*kappa[:-1, :]*v_n[:-1, :]

        # Add wind stress if enabled.
        if (use_wind is True):
            u_torch1[:-1, :] =u_torch1[:-1, :] + dt*tau_x[:]/(rho_0*H)
            v_torch1[:-1, :] =v_torch1[:-1, :] + dt*tau_y[:]/(rho_0*H)

        # Use a corrector method to add coriolis if it's enabled.
        if (use_coriolis is True):
            u_torch1[:, :] = (u_torch1[:, :] - beta_c*u_n[:, :] + alpha*v_n[:, :])/(1 + beta_c)
            v_torch1[:, :] = (v_torch1[:, :] - beta_c*v_n[:, :] - alpha*u_n[:, :])/(1 + beta_c)
        
        v_torch1[:, -1] = 0.0      # Northern boundary condition
        u_torch1[-1, :] = 0.0      # Eastern boundary condition
        # -------------------------- Done with u and v -----------------------------

        # --- Computing arrays needed for the upwind scheme in the eta equation.----
        h_e[:-1, :] = torch.where(u_torch1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
        h_e[-1, :] = eta_n[-1, :] + H

        h_w[0, :] = eta_n[0, :] + H
        h_w[1:, :] = torch.where(u_torch1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

        h_n[:, :-1] = torch.where(v_torch1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
        h_n[:, -1] = eta_n[:, -1] + H

        h_s[:, 0] = eta_n[:, 0] + H
        h_s[:, 1:] = torch.where(v_torch1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

        uhwe[0, :] = u_torch1[0, :]*h_e[0, :]
        uhwe[1:, :] = u_torch1[1:, :]*h_e[1:, :] - u_torch1[:-1, :]*h_w[1:, :]

        vhns[:, 0] = v_torch1[:, 0]*h_n[:, 0]
        vhns[:, 1:] = v_torch1[:, 1:]*h_n[:, 1:] - v_torch1[:, :-1]*h_s[:, 1:]
        # ------------------------- Upwind computations done -------------------------

        # ----------------- Computing eta values at next time step -------------------
        eta_torch1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)    # Without source/sink

        # Add source term if enabled.
        if (use_source is True):
            eta_torch1[:, :] =eta_torch1[:, :] + dt*sigma

        # Add sink term if enabled.
        if (use_sink is True):
            eta_torch1[:, :] =eta_torch1[:, :] - dt*w
        # ----------------------------- Done with eta --------------------------------

        u_n = torch.clone(u_torch1)        # Update u for next iteration
        v_n = torch.clone(v_torch1)        # Update v for next iteration
        eta_n = torch.clone(eta_torch1)    # Update eta for next iteration

        time_step += 1

        '''
        # Samples for Hovmuller diagram and spectrum every sample_interval time step.
        if (time_step % sample_interval == 0):
            hm_sample.append(eta_n[:, int(N_y/2)])              # Sample middle of domain for Hovmuller
            ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])     # Sample center point for spectrum
            t_sample.append(time_step*dt)                       # Keep track of sample times.

        # Store eta and (u, v) every anin_interval time step for animations.
        if (time_step % anim_interval == 0):
            print("Time: \t{:.2f} hours".format(time_step*dt/3600))
            print("Step: \t{} / {}".format(time_step, max_time_step))
            print("Mass: \t{}\n".format(torch.sum(eta_n)))
            u_list.append(u_n)
            v_list.append(v_n)
            eta_list.append(eta_n)
        '''
    u_n =u_n+ torch.normal(0, epsilon0, u_n.shape, device=device)
    v_n =v_n+ torch.normal(0, epsilon0, v_n.shape, device=device)
    eta_n =eta_n+ torch.normal(0, epsilon0, eta_n.shape, device=device)
    return u_n, v_n, eta_n

# |%%--%%| <3HWzhcf84v|LxyMyf80zt>

# EnKF parameters
ensemble_size = int(sys.argv[1])     # Number of ensemble members
assimilation_period = 10 # Number of time steps between assimilations
total_assimilation_steps = 200  # Total number of assimilation cycles
epsilon = float(sys.argv[2])
epsilon0 = 0.01

def initialize_true_state():
    """Initialize the true state with a Gaussian hump"""
    #u0 = torch.zeros((N_x, N_y))    
    #v0 = torch.zeros((N_x, N_y))    
    #eta0 = torch.exp(-((X-L_x/2.7)**2/(2*(0.05E+6)**2) + (Y-L_y/4)**2/(2*(0.05E+6)**2)))
    # 正弦波初始条件（满足刚性边界条件）
    k = 2 * np.pi / L_x
    l = 2 * np.pi / L_y
    A = 1.0

    omega = np.sqrt(f_0**2 + g * H * (k**2 + l**2))

    eta0 = A * torch.cos(k * X) * torch.cos(l * Y)
    u0 = (A * (g * k + f_0 * l) / omega) * torch.sin(k * X) * torch.cos(l * Y)
    v0 = (A * (g * l - f_0 * k) / omega) * torch.cos(k * X) * torch.sin(l * Y)
    
    return {'eta': eta0, 'u': u0, 'v': v0}

def generate_initial_ensemble(N, true_state):
    """Generate initial ensemble around the true state"""
    ensemble = {
        'eta': torch.zeros((N, N_x, N_y)),
        'u': torch.zeros((N, N_x, N_y)),
        'v': torch.zeros((N, N_x, N_y))
    }
    
    # Add perturbations to the true state
    for i in range(N):
        ensemble['eta'][i] = true_state['eta'] + torch.normal(0, epsilon, (N_x, N_y), device=device)
        ensemble['u'][i] = true_state['u'] + torch.normal(0, epsilon, (N_x, N_y), device=device)
        ensemble['v'][i] = true_state['v'] + torch.normal(0, epsilon, (N_x, N_y), device=device)
    
    return ensemble

def generate_observations(true_state):
    """生成观测数据：真实状态叠加1%的正态噪声"""
    nx, ny = true_state['eta'].shape[0], true_state['eta'].shape[1]
    state_dim = 3 * nx * ny  # eta, u, v
    obs_dim = state_dim #nx * ny // 5
    H = torch.zeros((obs_dim, state_dim))
    H[torch.arange(obs_dim), torch.arange(obs_dim)] = 1.0
    y = torch.concatenate([
        true_state['eta'].flatten(),
        true_state['u'].flatten(),
        true_state['v'].flatten()
    ])

    obs = H @ y + (torch.normal(0, epsilon, (H @ y).shape, device=device))
    return obs

def forecast_step(ensemble, period):
    """预测步：使用一个period前的分析状态，调用swe_forward"""
    N = ensemble['eta'].shape[0]
    forecast = {
        'eta': torch.zeros_like(ensemble['eta']),
        'u': torch.zeros_like(ensemble['u']),
        'v': torch.zeros_like(ensemble['v'])
    }
    
    for i in range(N):
        # 对每个集合成员进行前向积分
        u, v, eta = swe_forward(
            ensemble['eta'][i], 
            ensemble['u'][i], 
            ensemble['v'][i], 
            iters=period
        )
        forecast['u'][i] = u
        forecast['v'][i] = v
        forecast['eta'][i] = eta
    
    return forecast

# |%%--%%| <LxyMyf80zt|LYQFCovb22>

from tqdm import tqdm

def plot_results(rmse_eta, rmse_u, rmse_v):
    """Plot the RMSE evolution"""
    plt.figure(figsize=(12, 6))
    plt.plot(rmse_eta.cpu(), label='η RMSE (m)')
    plt.plot(rmse_u.cpu(), label='u RMSE (m/s)')
    plt.plot(rmse_v.cpu(), label='v RMSE (m/s)')
    plt.xlabel('Assimilation Cycle')
    plt.ylabel('RMSE')
    plt.title('SMC Performance for Shallow Water Equations')
    plt.legend()
    plt.grid()
    plt.savefig(f"rmse_smc_{ensemble_size}_{epsilon}.pdf", bbox_inches='tight')
    plt.show()

# |%%--%%| <LYQFCovb22|PM90Hkzsxd>

# 定义SMC参数
threshold_ess = ensemble_size / 2  # 重采样阈值

def compute_log_weights(forecast, obs, prev_log_weights):
    N = forecast['eta'].shape[0]
    log_likelihoods = torch.zeros(N, device=device)
    for i in range(N):
        x = torch.concatenate([
            forecast['eta'][i].flatten(),
            forecast['u'][i].flatten(),
            forecast['v'][i].flatten()
        ])
        total_error = torch.sum((x - obs)**2)
        log_likelihood = -total_error / (2 * epsilon**2)
        log_likelihoods[i] = log_likelihood
    return prev_log_weights + log_likelihoods

def effective_sample_size(weights):
    return 1.0 / torch.sum(weights**2)

def resample(particles, weights):
    indices = torch.multinomial(weights, ensemble_size, replacement=True)
    resampled = {
        'eta': particles['eta'][indices],
        'u': particles['u'][indices],
        'v': particles['v'][indices]
    }
    return resampled, torch.ones(ensemble_size, device=device) / ensemble_size

def run_true_state(true_state, period):
    u_new, v_new, eta_new = swe_forward(
        true_state['eta'], true_state['u'], true_state['v'], iters=period
    )
    return {'eta': eta_new, 'u': u_new, 'v': v_new}

# 主程序
def SMC_validation():

    true_state = initialize_true_state()
    particles = generate_initial_ensemble(ensemble_size, true_state)
    weights = torch.ones(ensemble_size, device=device) / ensemble_size

    # Storage for RMSE
    rmse_eta = []
    rmse_u = []
    rmse_v = []
    ensemble_list = []
    ess = []

    for step in tqdm(range(total_assimilation_steps)):

        # Calculate RMSE
        expanded_weights = weights.view(-1, 1, 1)

        analysis_mean = {
            'eta': torch.sum(expanded_weights * particles['eta'], dim=0),
            'u': torch.sum(expanded_weights * particles['u'], dim=0),
            'v': torch.sum(expanded_weights * particles['v'], dim=0)
        }
        rmse_eta.append(torch.sqrt(torch.mean((analysis_mean['eta'] - true_state['eta'])**2)))
        rmse_u.append(torch.sqrt(torch.mean((analysis_mean['u'] - true_state['u'])**2)))
        rmse_v.append(torch.sqrt(torch.mean((analysis_mean['v'] - true_state['v'])**2)))

        # 更新真实状态并生成观测
        true_state = run_true_state(true_state, assimilation_period)
        obs = generate_observations(true_state)
        
        # 预测步
        forecast = forecast_step(particles, assimilation_period)
        
        # 计算权重
        log_weights = compute_log_weights(forecast, obs, torch.log(weights))
        weights = torch.exp(log_weights - torch.max(log_weights))
        weights /= torch.sum(weights)

        if step % 10 == 0:
                tqdm.write(f"Step {step}: RMSE eta={rmse_eta[-1]:.4f} m, u={rmse_u[-1]:.4f} m/s, v={rmse_v[-1]:.4f} m/s")
                ensemble_list.append((forecast, weights, true_state))
        
        # 重采样判断
        ess.append(effective_sample_size(weights))
        if effective_sample_size(weights) < threshold_ess:
            particles, weights = resample(forecast, weights)
        else:
            particles = forecast
        
    rmse_eta = torch.as_tensor(rmse_eta)
    rmse_u = torch.as_tensor(rmse_u)
    rmse_v = torch.as_tensor(rmse_v)
    ess = torch.as_tensor(ess)

    return ensemble_list, rmse_eta, rmse_u, rmse_v, ess

# |%%--%%| <PM90Hkzsxd|yVDFAy32GG>

SMC_list, re, ru, rv, ess = SMC_validation()
torch.save(SMC_list, f"./SMC_{ensemble_size}_{epsilon}.pt")

plt.plot(ess.cpu())
plt.title("Effective sample size for SMC")
plt.savefig(f"ESS_{ensemble_size}_{epsilon}.pdf")
plt.close()

plot_results(re, ru, rv)

