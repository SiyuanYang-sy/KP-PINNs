# %%
# Libraries and Dependencies
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import kron
from scipy.sparse import identity

from KP_compute_APhi import KP_compute_APhi

import torch.nn.utils as clip_grad_norm_

# %%
t0 = time.time()

np.random.seed(33)
torch.manual_seed(44)

torch.set_default_dtype(torch.float64)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# %%
# Physics-informed Neural Networks
# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh  # Tanh, Sigmoid, ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        x = x.to(torch.float64)
        out = self.layers(x)
        return out


# the physics informed neural network
class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, layers, mu):

        # data
        self.x1_u = torch.tensor(X_u[:, 0:1], requires_grad=True).double().to(device)
        self.x2_u = torch.tensor(X_u[:, 1:2], requires_grad=True).double().to(device)
        self.t_u = torch.tensor(X_u[:, 2:3], requires_grad=True).double().to(device)
        self.x1_f = torch.tensor(X_f[:, 0:1], requires_grad=True).double().to(device)
        self.x2_f = torch.tensor(X_f[:, 1:2], requires_grad=True).double().to(device)
        self.t_f = torch.tensor(X_f[:, 2:3], requires_grad=True).double().to(device)
        self.u_u = torch.tensor(u[:, 0:1]).double().to(device)
        self.v_u = torch.tensor(u[:, 1:2]).double().to(device)

        # some computation
        self.A_torch_u, self.Phi_inv_torch_u = self.computation_KP_u(self.x1_u, self.x2_u)
        self.A_torch_f, self.Phi_inv_torch_f = self.computation_KP_f(self.x1_f, self.x2_f, self.t_f)

        # layers
        self.layers = layers
        self.mu = mu

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=10000,  # 10000, 50000
            max_eval=50000,
            history_size=100,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            # tolerance_change=1e-5,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        # self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # self.optimizer = torch.optim.RMSprop(self.dnn.parameters(), lr=0.001, alpha=0.9)

        self.iter = 0

        self.Loss_u = []
        self.Loss_f = []
        self.Loss = []

    def net_psi(self, x1, x2, t):    # x1:x, x2:y, t:t
        psi = self.dnn(torch.cat([x1, x2, t], dim=1))
        return psi

    def net_f(self, x1, x2, t):      # x1:x, x2:y, t:t
        """ The pytorch autograd version of calculating residual """
        psi = self.net_psi(x1, x2, t)

        psi_x1 = torch.autograd.grad(
            psi, x1,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True
        )[0]
        psi_x2 = torch.autograd.grad(
            psi, x2,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True
        )[0]
        psi_x1x1 = torch.autograd.grad(
            psi_x1, x1,
            grad_outputs=torch.ones_like(psi_x1),
            retain_graph=True,
            create_graph=True
        )[0]
        psi_x2x2 = torch.autograd.grad(
            psi_x2, x2,
            grad_outputs=torch.ones_like(psi_x2),
            retain_graph=True,
            create_graph=True
        )[0]

        w = -(psi_x1x1 + psi_x2x2)
        u = psi_x2
        v = -psi_x1

        w_t = torch.autograd.grad(
            w, t,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        w_x1 = torch.autograd.grad(
            w, x1,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        w_x2 = torch.autograd.grad(
            w, x2,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True
        )[0]
        w_x1x1 = torch.autograd.grad(
            w_x1, x1,
            grad_outputs=torch.ones_like(w_x1),
            retain_graph=True,
            create_graph=True
        )[0]
        w_x2x2 = torch.autograd.grad(
            w_x2, x2,
            grad_outputs=torch.ones_like(w_x2),
            retain_graph=True,
            create_graph=True
        )[0]

        f = w_t + u * w_x1 + v * w_x2 - self.mu * (w_x1x1 + w_x2x2)
        return f

    def computation_KP_u(self, x1, x2):
        t1 = time.time()
        xx1, xx2 = np.unique(x1.cpu().detach().numpy()), np.unique(x2.cpu().detach().numpy())
        print(xx1.shape, xx2.shape)
        A1, Phi1 = KP_compute_APhi(xx1, 2, 1 / 2)
        A2, Phi2 = KP_compute_APhi(xx2, 2, 1 / 2)
        A_Kron = kron(A1, A2)
        lamda = 0.01
        n1, n2 = Phi1.shape[0], Phi2.shape[0]
        I1, I2 = identity(n1).tocsc(), identity(n2).tocsc()
        # Phi1, Phi2 = Phi1 + lamda * I1, Phi2 + lamda * I2
        Phi1, Phi2 = Phi1 + lamda * A1, Phi2 + lamda * A2
        Phi1_inv, Phi2_inv = spsolve(Phi1, I1), spsolve(Phi2, I2)
        Phi_inv_Kron = kron(Phi1_inv, Phi2_inv)
        A_torch, Phi_inv = torch.tensor(A_Kron.toarray()), torch.tensor(Phi_inv_Kron.toarray())
        t2 = time.time()
        print('KP_u time: ', t2 - t1)
        return A_torch.to(device), Phi_inv.to(device)

    def computation_KP_f(self, x1, x2, t):
        t1 = time.time()
        xx1, xx2, tt = np.unique(x1.cpu().detach().numpy()), np.unique(x2.cpu().detach().numpy()), np.unique(
            t.cpu().detach().numpy())
        A1, Phi1 = KP_compute_APhi(xx1, 2, 1 / 2)
        A2, Phi2 = KP_compute_APhi(xx2, 2, 1 / 2)
        A3, Phi3 = KP_compute_APhi(tt, 2, 1 / 2)
        A_Kron = kron(kron(A1, A2), A3)
        lamda = 0.01
        n1, n2, n3 = Phi1.shape[0], Phi2.shape[0], Phi3.shape[0]
        I1, I2, I3 = identity(n1).tocsc(), identity(n2).tocsc(), identity(n3).tocsc()
        # Phi1, Phi2, Phi3 = Phi1 + lamda * I1, Phi2 + lamda * I2, Phi3 + lamda * I3
        Phi1, Phi2, Phi3 = Phi1 + lamda * A1, Phi2 + lamda * A2, Phi3 + lamda * A3
        Phi1_inv, Phi2_inv, Phi3_inv = spsolve(Phi1, I1), spsolve(Phi2, I2), spsolve(Phi3, I3)
        Phi_inv_Kron = kron(kron(Phi1_inv, Phi2_inv), Phi3_inv)
        A_torch, Phi_inv = torch.tensor(A_Kron.toarray()), torch.tensor(Phi_inv_Kron.toarray())
        t2 = time.time()
        print('KP_f time: ', t2 - t1)
        return A_torch.to(device), Phi_inv.to(device)

    def loss_func(self):
        self.optimizer.zero_grad()

        #################################### loss_u #####################################
        psi_pred = self.net_psi(self.x1_u, self.x2_u, self.t_u)

        # Boundary condition - u(x1, x2, T) = g(x1, x2)
        u_pred = torch.autograd.grad(psi_pred, self.x2_u, grad_outputs=torch.ones_like(psi_pred), create_graph=True)[0]
        v_pred = -torch.autograd.grad(psi_pred, self.x1_u, grad_outputs=torch.ones_like(psi_pred), create_graph=True)[0]
        pred_u = self.u_u - u_pred
        pred_v = self.v_u - v_pred

        # KP - RKHS ≈ Y^T K^(-1) Y
        loss_u_KP1 = torch.div(torch.abs(pred_u.T @ self.A_torch_u @ self.Phi_inv_torch_u @ pred_u), len(pred_u))
        loss_u_KP2 = torch.div(torch.abs(pred_v.T @ self.A_torch_u @ self.Phi_inv_torch_u @ pred_v), len(pred_v))
        loss_u_KP = loss_u_KP1 + loss_u_KP2

        #################################### loss_f ####################################
        f_pred = self.net_f(self.x1_f, self.x2_f, self.t_f)

        # KP - RKHS ≈ Y^T K^(-1) Y
        loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch_f @ self.Phi_inv_torch_f @ f_pred), len(f_pred))

        loss_u = loss_u_KP  # loss_u_L2, loss_u_KP, loss_u_RKHS, loss_u_Sobolev
        loss_f = loss_f_KP  # loss_f_L2, loss_f_KP, loss_f_RKHS, loss_f_Sobolev
        self.Loss_u.append(float(loss_u))
        self.Loss_f.append(float(loss_f))
        w1 = 1
        w2 = 1
        loss = w1 * loss_u + w2 * loss_f
        self.Loss.append(float(loss))

        loss.backward()
        self.iter += 1
        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
            self.iter, loss.item(), loss_u.item(), loss_f.item()))
        return loss

    def train(self):
        self.dnn.train()
        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x1 = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        x2 = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).double().to(device)

        self.dnn.eval()
        psi = self.net_psi(x1, x2, t)
        u = torch.autograd.grad(psi, x2, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x1, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        f = self.net_f(x1, x2, t)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, v, f

# %%
# Configurations
mu = 0.01
noise = 0.0

N_u = 5000  # <5000
N_x = 30  # <98
N_y = 20  # <48
N_t = 10   # <20
layers = [3, 20, 20, 20, 20, 20, 20, 1]

data = pd.read_csv('2s_noise_free_NS_in_col.csv')
t = data['t'].values[:, None].astype(np.float64)   # (94080, 1)
x = data['x'].values[:, None].astype(np.float64)   # (94080, 1)
y = data['y'].values[:, None].astype(np.float64)   # (94080, 1)
u = data['u'].values[:, None].astype(np.float64)   # (94080, 1)
v = data['v'].values[:, None].astype(np.float64)   # (94080, 1)

nt = len(np.unique(t))   # 20
ny = len(np.unique(y))   # 48
nx = len(np.unique(x))   # 98

u_tyx = u.reshape((nt, ny, nx))   # (20, 48, 98)
v_tyx = v.reshape((nt, ny, nx))   # (20, 48, 98)

x_unique = np.unique(x).flatten()[:, None]   # (98, 1)
y_unique = np.unique(y).flatten()[:, None]   # (48, 1)
t_unique = np.unique(t).flatten()[:, None]   # (20, 1)
u_grid = np.transpose(u_tyx, (2, 1, 0))  # from (t, y, x) to (x, y, t), (98, 48, 20)
v_grid = np.transpose(v_tyx, (2, 1, 0))  # from (t, y, x) to (x, y, t), (98, 48, 20)

X1, X2, T = np.meshgrid(x_unique, y_unique, t_unique, indexing='ij')  # X1, X2, T: (98, 48, 20)
X_star = np.hstack((X1.flatten()[:, None], X2.flatten()[:, None], T.flatten()[:, None]))  # (94080, 3)
u_star_u = u_grid.flatten()[:, None]  # (94080, 1)
u_star_v = v_grid.flatten()[:, None]  # (94080, 1)
u_star = np.hstack([u_star_u, u_star_v])    # (94080, 2)

# IC (t=0)
data_ic = pd.read_csv('NS_ic.csv')
t_ic = data_ic['t'].values[:, None].astype(np.float64)       # (5000, 1)
x_ic = data_ic['x'].values[:, None].astype(np.float64)       # (5000, 1)
y_ic = data_ic['y'].values[:, None].astype(np.float64)       # (5000, 1)
u_ic = data_ic['u'].values[:, None].astype(np.float64)       # (5000, 1)
v_ic = data_ic['v'].values[:, None].astype(np.float64)       # (5000, 1)
xyt_ic = np.hstack([x_ic, y_ic, t_ic])    # (5000, 3)
uv_ic = np.hstack([u_ic, v_ic])           # (5000, 2)

# # BC
# data_bc = pd.read_csv('2s_NS_bc.csv')
# t_bc = data_bc['t'].values[:, None].astype(np.float64)    # (5920, 1)
# x_bc = data_bc['x'].values[:, None].astype(np.float64)    # (5920, 1)
# y_bc = data_bc['y'].values[:, None].astype(np.float64)    # (5920, 1)
# u_bc = data_bc['u'].values[:, None].astype(np.float64)    # (5920, 1)
# v_bc = data_bc['v'].values[:, None].astype(np.float64)    # (5920, 1)

# X_u_train, u_train
idx = np.sort(np.random.choice(xyt_ic.shape[0], N_u, replace=False))  # (N_u, )
X_u_train = xyt_ic[idx, :]  # (N_u, 3)
u_train = uv_ic[idx, :]     # (N_u, 2)

# X_f_train
idx = np.sort(np.random.choice(x_unique.shape[0], N_x, replace=False))
Xx1 = x_unique[idx]  # len(Xx1) = N_x
idx = np.sort(np.random.choice(y_unique.shape[0], N_y, replace=False))
Xx2 = y_unique[idx]  # len(Xx2) = N_y
idx = np.sort(np.random.choice(t_unique.shape[0], N_t, replace=False))
Xt = t_unique[idx]  # len(Xt) = N_t
X1_f, X2_f, T_f = np.meshgrid(Xx1, Xx2, Xt, indexing='ij')     # (N_x, N_y, N_t)
X_f_train = np.hstack((X1_f.flatten()[:, None], X2_f.flatten()[:, None], T_f.flatten()[:, None]))   # (N_x*N_y*N_t, 3)

# print(u_star.size, np.linalg.norm(u_star))     # Relative L2 Error, 188160, 288.63528968965886

# Training
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, mu)
t11 = time.time()
print('Running time 11 (init):', t11 - t0)
model.train()

t1 = time.time()
print('Running time 1 (train):', t1 - t0)

# predict
u_pred_u, u_pred_v, f_pred = model.predict(X_star)
u_pred = np.hstack([u_pred_u, u_pred_v])

t2 = time.time()
print('Running time 2 (predict):', t2 - t0)

error_u = np.mean(np.abs(u_star - u_pred) ** 2)  # L2
print('Error u: %e' % (error_u))

# %%
# Visualization - loss
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure()
w = np.linspace(0, model.iter, model.iter)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title(r'\textbf{Loss function}')
plt.plot(w, model.Loss_u, label='Loss_u')
plt.plot(w, model.Loss_f, label='Loss_f')
plt.plot(w, model.Loss, label='Loss')
plt.legend()

# plt.savefig('loss.png', dpi=600)
# plt.savefig('loss.pdf', dpi=600)

# %%
plt.show()

t3 = time.time()
print('Running time 3:', t3 - t0)


