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
    #kappa[0, :] = kappa_0
    #kappa[-1, :] = kappa_0
    #kappa[:, 0] = kappa_0
    #kappa[:, -1] = kappa_0
    #kappa[:int(N_x/15), :] = 0
    #kappa[int(14*N_x/15)+1:, :] = 0
    #kappa[:, :int(N_y/15)] = 0
    #kappa[:, int(14*N_y/15)+1:] = 0
    #kappa[int(N_x/15):int(2*N_x/15), int(N_y/15):int(14*N_y/15)+1] = 0
    #kappa[int(N_x/15):int(14*N_x/15)+1, int(N_y/15):int(2*N_y/15)] = 0
    #kappa[int(13*N_x/15)+1:int(14*N_x/15)+1, int(N_y/15):int(14*N_y/15)+1] = 0
    #kappa[int(N_x/15):int(14*N_x/15)+1, int(13*N_y/15)+1:int(14*N_y/15)+1] = 0
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
            u_torch1[:-1, :] -= dt*kappa[:-1, :]*u_n[:-1, :]
            v_torch1[:-1, :] -= dt*kappa[:-1, :]*v_n[:-1, :]

        # Add wind stress if enabled.
        if (use_wind is True):
            u_torch1[:-1, :] += dt*tau_x[:]/(rho_0*H)
            v_torch1[:-1, :] += dt*tau_y[:]/(rho_0*H)

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
            eta_torch1[:, :] += dt*sigma

        # Add sink term if enabled.
        if (use_sink is True):
            eta_torch1[:, :] -= dt*w
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
    u_n += torch.normal(0, epsilon0, u_n.shape, device=device)
    v_n += torch.normal(0, epsilon0, v_n.shape, device=device)
    eta_n += torch.normal(0, epsilon0, eta_n.shape, device=device)
    return u_n, v_n, eta_n



# |%%--%%| <3HWzhcf84v|LxyMyf80zt>

# EnKF parameters
ensemble_size = int(sys.argv[1])     # Number of ensemble members
assimilation_period = 10 # Number of time steps between assimilations
total_assimilation_steps = 200  # Total number of assimilation cycles
epsilon = float(sys.argv[2])
epsilon0 = 0.01

# localization
nx = N_x
ny = N_y
dx_phys = L_x / (N_x - 1)  # 实际物理间距(m)
dy_phys = L_y / (N_y - 1)
x_coords = torch.linspace(-L_x/2, L_x/2, nx, device=device)
y_coords = torch.linspace(-L_y/2, L_y/2, ny, device=device)
xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

# 展平坐标网格
points = torch.stack([xx.ravel(), yy.ravel()], dim=1)  # [nx*ny, 2]

# 计算所有点对之间的欧氏距离（高效向量化）
dist_matrix = torch.cdist(points, points)  # [nx*ny, nx*ny]

def rho(distance, L=5e4):  # L=50km局地化尺度
    return torch.exp(-distance/L)

rho_matrix = rho(dist_matrix)

state_dim = 3 * nx * ny
rho_total = torch.zeros(state_dim, state_dim, device=device)
for i in range(3):
    for j in range(3):
        rstart = i * nx * ny
        rend = (i+1) * nx * ny
        cstart = j * nx * ny
        cend = (j+1) * nx * ny
        rho_total[rstart:rend, cstart:cend] = rho_matrix

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

def generate_quad_observations(true_state):
    """生成带噪声的非线性观测 H(x) = x^2 (PyTorch GPU版本)"""
    device = true_state['eta'].device  # 自动匹配输入设备
    
    # 对真实状态施加平方非线性后添加噪声
    observations = {
        'eta': true_state['eta']**2 + torch.randn_like(true_state['eta']) * epsilon,
        'u': true_state['u']**2 + torch.randn_like(true_state['u']) * epsilon,
        'v': true_state['v']**2 + torch.randn_like(true_state['v']) * epsilon
    }
    
    return observations

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


def analysis_step(forecast, observations, localize=False):
    """分析步：使用观测更新预测"""
    N = forecast['eta'].shape[0]
    nx, ny = forecast['eta'].shape[1], forecast['eta'].shape[2]
    
    # 将状态向量化
    state_dim = 3 * nx * ny  # eta, u, v
    obs_dim = state_dim #nx * ny // 5
    H = torch.zeros((obs_dim, state_dim))
    H[torch.arange(obs_dim), torch.arange(obs_dim)] = 1.0
    
    # 观测误差协方差矩阵
    R = (epsilon**2) * torch.eye(obs_dim)
    
    # 集合均值
    x_f_mean = {
        'eta': torch.mean(forecast['eta'], axis=0),
        'u': torch.mean(forecast['u'], axis=0),
        'v': torch.mean(forecast['v'], axis=0)
    }
    
    # 将集合成员转换为矩阵形式
    X_f = torch.zeros((state_dim, N))
    for i in range(N):
        X_f[:, i] = torch.concatenate([
            forecast['eta'][i].flatten(),
            forecast['u'][i].flatten(),
            forecast['v'][i].flatten()
        ])
    
    # 计算集合协方差
    x_f_mean_vec = torch.concatenate([
        x_f_mean['eta'].flatten(),
        x_f_mean['u'].flatten(),
        x_f_mean['v'].flatten()
    ])
    X_prime = X_f - x_f_mean_vec.reshape(-1, 1)
    P_f = X_prime @ X_prime.T / (N - 1)
    if localize:
        P_f = P_f * rho_total
    
    # 观测向量
    y = observations
    
    # 卡尔曼增益
    #K = P_f @ H.T @ torch.linalg.inv(H @ P_f @ H.T + R)
    def Kalman_gain(X):
        A = H @ P_f @ H.T + R
        X1 = torch.linalg.solve(A, X)
        return P_f @ H.T @ X1
    
    # 分析更新
    #x_a_mean_vec = x_f_mean_vec + Kalman_gain(y - H @ x_f_mean_vec)
    
    # 更新集合
    X_a = X_f + Kalman_gain(y.reshape(-1, 1) - H @ X_f)
    
    # 将分析结果转换回原始形式
    analysis = {
        'eta': torch.zeros_like(forecast['eta']),
        'u': torch.zeros_like(forecast['u']),
        'v': torch.zeros_like(forecast['v'])
    }
    
    for i in range(N):
        eta_size = nx * ny
        analysis['eta'][i] = X_a[:eta_size, i].reshape(nx, ny)
        analysis['u'][i] = X_a[eta_size:2*eta_size, i].reshape(nx, ny)
        analysis['v'][i] = X_a[2*eta_size:, i].reshape(nx, ny)
    
    return analysis

def quad_analysis_step(forecast, observations, localize=False):
    """支持非线性观测H(x)=x^2的分析步（使用线性求解器优化）"""
    device = forecast['eta'].device
    N = forecast['eta'].shape[0]
    nx, ny = forecast['eta'].shape[1], forecast['eta'].shape[2]
    state_dim = 3 * nx * ny
    
    # 构建扩展状态矩阵 [x; x^2]
    X_f = torch.zeros((2 * state_dim, N), device=device)
    for i in range(N):
        x = torch.cat([forecast[k][i].flatten() for k in ['eta', 'u', 'v']])
        X_f[:, i] = torch.cat([x, x**2])
    
    # 计算统计量
    z_mean = torch.mean(X_f, dim=1, keepdim=True)
    X_prime = X_f - z_mean
    cov_zz = X_prime @ X_prime.T / (N - 1)  # 扩展状态协方差
    if localize:
        cov_zz = cov_zz * rho_total.tile((2, 2))
    
    # 观测算子与噪声
    H_ext = torch.cat([
        torch.zeros(state_dim, state_dim, device=device),
        torch.eye(state_dim, device=device)
    ], dim=1)
    R = (epsilon**2) * torch.eye(state_dim, device=device)
    y = torch.cat([observations[k].view(-1) for k in ['eta', 'u', 'v']])
    
    # 高效计算卡尔曼增益（避免显式求逆）
    S = H_ext @ cov_zz @ H_ext.T + R
    Y = cov_zz @ H_ext.T  # 临时变量
    
    # 解线性方程组 Y = K @ S （等价于 K = Y @ inv(S)）
    K = torch.linalg.solve(S.T, Y.T).T  # 等价于 solve(S.T, Y.T).T
    
    # 分析更新
    x_a_mean = z_mean[:state_dim] + K[:state_dim, :] @ (y - H_ext @ z_mean)
    X_a = X_f[:state_dim, :] + K[:state_dim, :] @ (y.view(-1, 1) - H_ext @ X_f)
    
    # 重构输出
    analysis = {
        'eta': torch.zeros_like(forecast['eta']),
        'u': torch.zeros_like(forecast['u']),
        'v': torch.zeros_like(forecast['v'])
    }
    eta_size = nx * ny
    for i in range(N):
        analysis['eta'][i] = X_a[:eta_size, i].view(nx, ny)
        analysis['u'][i] = X_a[eta_size:2*eta_size, i].view(nx, ny)
        analysis['v'][i] = X_a[2*eta_size:, i].view(nx, ny)
    
    return analysis

# |%%--%%| <LxyMyf80zt|LYQFCovb22>

from tqdm import tqdm
def enkf_validation(quad=False, localize=False):
    """EnKF validation with adjusted parameters"""
    # Initialize true state
    true_state = initialize_true_state()
    
    # Generate initial ensemble
    ensemble = generate_initial_ensemble(ensemble_size, true_state)
    
    # Storage for RMSE
    rmse_eta = []
    rmse_u = []
    rmse_v = []
    ensemble_hist = []
    
    for step in tqdm(range(total_assimilation_steps)):
        # Calculate RMSE
        analysis_mean = {
            'eta': torch.mean(ensemble['eta'], axis=0),
            'u': torch.mean(ensemble['u'], axis=0),
            'v': torch.mean(ensemble['v'], axis=0)
        }
        
        rmse_eta.append(torch.sqrt(torch.mean((analysis_mean['eta'] - true_state['eta'])**2)))
        rmse_u.append(torch.sqrt(torch.mean((analysis_mean['u'] - true_state['u'])**2)))
        rmse_v.append(torch.sqrt(torch.mean((analysis_mean['v'] - true_state['v'])**2)))

        # Advance true state
        u, v, eta = swe_forward(
            true_state['eta'], 
            true_state['u'], 
            true_state['v'], 
            iters=assimilation_period
        )
        true_state = {'eta': eta, 'u': u, 'v': v}
        
        # Forecast step
        forecast = forecast_step(ensemble, assimilation_period)
        
        if not quad:
            # Generate observations
            observations = generate_observations(true_state)
            # Analysis step
            ensemble = analysis_step(forecast, observations, localize)
        else:
            observations = generate_quad_observations(true_state)
            ensemble = quad_analysis_step(forecast, observations, localize)
        
        if step % 10 == 0:
            tqdm.write(f"Step {step}: RMSE eta={rmse_eta[-1]:.4f} m, u={rmse_u[-1]:.4f} m/s, v={rmse_v[-1]:.4f} m/s")
            ensemble_hist.append((ensemble, true_state))

    rmse_eta = torch.as_tensor(rmse_eta)
    rmse_u = torch.as_tensor(rmse_u)
    rmse_v = torch.as_tensor(rmse_v)

    return ensemble_hist, rmse_eta, rmse_u, rmse_v


def plot_results(rmse_eta, rmse_u, rmse_v, cl=False, quad=False):
    """Plot the RMSE evolution"""
    plt.figure(figsize=(12, 6))
    plt.plot(rmse_eta.cpu(), label='η RMSE (m)')
    plt.plot(rmse_u.cpu(), label='u RMSE (m/s)')
    plt.plot(rmse_v.cpu(), label='v RMSE (m/s)')
    plt.xlabel('Assimilation Cycle')
    plt.ylabel('RMSE')
    plt.title('EnKF Performance for Shallow Water Equations')
    plt.legend()
    plt.grid()
    prefix = ''
    if cl:
        prefix += '_cl'
    if quad:
        prefix += '_quad'
    plt.savefig(f"rmse{prefix}_{ensemble_size}_{epsilon}.pdf", bbox_inches='tight')
    plt.show()



# |%%--%%| <LYQFCovb22|mammst7fEH>

enkf_hist, re, ru, rv = enkf_validation(quad=True)
plot_results(re, ru, rv, quad=True)
torch.save(enkf_hist, f"./enkf_quad_{ensemble_size}_{epsilon}.pt")

enkf_hist, re, ru, rv = enkf_validation(quad=True, localize=True)
plot_results(re, ru, rv, quad=True, cl=True)
torch.save(enkf_hist, f"./enkf_cl_quad_{ensemble_size}_{epsilon}.pt")
