# %%
# Libraries and Dependencies
import torch
from collections import OrderedDict
import numpy as np
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

np.random.seed(12)
torch.manual_seed(12)

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


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X_u, u, layers):

        # data
        self.x1_u = torch.tensor(X_u[:, 0:1], requires_grad=True).double().to(device)
        self.x2_u = torch.tensor(X_u[:, 1:2], requires_grad=True).double().to(device)
        self.t_u = torch.tensor(X_u[:, 2:3], requires_grad=True).double().to(device)
        self.u = torch.tensor(u).double().to(device)

        # some computation
        self.A_torch, self.Phi_inv_torch = self.computation_KP(self.x1_u, self.x2_u, self.t_u)

        # layers
        self.layers = layers

        # parameter setting
        self.mu = torch.tensor([torch.rand(1)], requires_grad=True).to(device)  # initial mu, mu_exact = 1.0
        self.mu = torch.nn.Parameter(self.mu)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('mu', self.mu)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=10000,  # 50000
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
        self.Mu = []

    def net_u(self, x1, x2, t):
        u = self.dnn(torch.cat([x1, x2, t], dim=1))
        return u

    def net_f(self, x1, x2, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x1, x2, t)

        u_x1 = torch.autograd.grad(
            u, x1,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x2 = torch.autograd.grad(
            u, x2,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x1x1 = torch.autograd.grad(
            u_x1, x1,
            grad_outputs=torch.ones_like(u_x1),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x2x2 = torch.autograd.grad(
            u_x2, x2,
            grad_outputs=torch.ones_like(u_x2),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u_x1x1 + u_x2x2 - self.mu * (u_x1 ** 2 + u_x2 ** 2)
        return f

    def computation_KP(self, x1, x2, t):
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
        Phi1, Phi2, Phi3 = Phi1 + lamda * I1, Phi2 + lamda * I2, Phi3 + lamda * I3
        # Phi1, Phi2, Phi3 = Phi1 + lamda * A1, Phi2 + lamda * A2, Phi3 + lamda * A3
        Phi1_inv, Phi2_inv, Phi3_inv = spsolve(Phi1, I1), spsolve(Phi2, I2), spsolve(Phi3, I3)
        Phi_inv_Kron = kron(kron(Phi1_inv, Phi2_inv), Phi3_inv)
        A_torch, Phi_inv = torch.tensor(A_Kron.toarray()), torch.tensor(Phi_inv_Kron.toarray())
        t2 = time.time()
        print('KP time: ', t2 - t1)
        return A_torch.to(device), Phi_inv.to(device)

    def loss_func(self):
        self.optimizer.zero_grad()

        #################################### loss_u #####################################
        u_pred = self.net_u(self.x1_u, self.x2_u, self.t_u)

        # Boundary condition - u(x1, x2, T) = g(x1, x2)
        pred = self.u - u_pred
        # KP - RKHS ≈ Y^T K^(-1) Y
        loss_u_KP = torch.div(torch.abs(pred.T @ self.A_torch @ self.Phi_inv_torch @ pred), len(pred))

        #################################### loss_f ####################################
        f_pred = self.net_f(self.x1_u, self.x2_u, self.t_u)

        # KP - RKHS ≈ Y^T K^(-1) Y
        loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch @ self.Phi_inv_torch @ f_pred), len(f_pred))

        loss_u = loss_u_KP  # loss_u_L2, loss_u_KP, loss_u_RKHS, loss_u_Sobolev
        loss_f = loss_f_KP  # loss_f_L2, loss_f_KP, loss_f_RKHS, loss_f_Sobolev
        self.Loss_u.append(float(loss_u))
        self.Loss_f.append(float(loss_f))
        w1 = 1
        w2 = 1
        loss = w1 * loss_u + w2 * loss_f
        self.Loss.append(float(loss))
        self.Mu.append(float(self.mu))

        loss.backward()
        self.iter += 1
        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e, μ: %.5e' % (
            self.iter, loss.item(), loss_u.item(), loss_f.item(), self.mu))
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
        u = self.net_u(x1, x2, t)
        f = self.net_f(x1, x2, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


# %%
# Configurations
mu_exact = 1.0
T = 0.0
noise = 0.0

N_x1 = 10     # <100
N_x2 = 10     # <80
N_t = 10      # <50
layers = [3, 20, 20, 20, 20, 20, 20, 1]

data = np.load('LQG_solution.npz', allow_pickle=True)

x1 = data['x1'].flatten()[:, None].astype(np.float64)  # (100, 1)
x2 = data['x2'].flatten()[:, None].astype(np.float64)  # (80, 1)
t = data['t'].flatten()[:, None].astype(np.float64)  # (50, 1)
Exact = np.real(data['u']).astype(np.float64)  # (100, 80, 50)

X1, X2, T = np.meshgrid(x1, x2, t, indexing='ij')  # X1, X2, T: (100, 80, 50)
X_star = np.hstack((X1.flatten()[:, None], X2.flatten()[:, None], T.flatten()[:, None]))  # (400000, 3)
u_star = Exact.flatten()[:, None]  # (400000,1)

# X_train
idx1 = np.sort(np.random.choice(x1.shape[0], N_x1, replace=False))
Xx1 = x1[idx1]  # len(Xx1) = N_x1
idx2 = np.sort(np.random.choice(x2.shape[0], N_x2, replace=False))
Xx2 = x2[idx2]  # len(Xx2) = N_x2
idx3 = np.sort(np.random.choice(t.shape[0], N_t, replace=False))
Xt = t[idx3]  # len(Xt) = N_t
X1_f, X2_f, T_f = np.meshgrid(Xx1, Xx2, Xt, indexing='ij')
idx = np.ix_(idx1, idx2, idx3)
u_data = Exact[idx]
X_train = np.hstack((X1_f.flatten()[:, None], X2_f.flatten()[:, None], T_f.flatten()[:, None]))
u_train = u_data.flatten()[:, None]

# print(u_star.size, np.linalg.norm(u_star))     # Relative L2 Error, 400000, 198.61099142740932

# Training
model = PhysicsInformedNN(X_train, u_train, layers)
t11 = time.time()
print('Running time 11 (init):', t11 - t0)
model.train()

t1 = time.time()
print('Running time 1 (train):', t1 - t0)

mu_pre = model.Mu
print('mu_pre: ', mu_pre[-1])
print('mu_exact: ', mu_exact)
error_mu = np.abs(mu_pre[-1] - mu_exact) / np.abs(mu_exact)
print('Error mu: %e' % error_mu)

# predict
u_pred, f_pred = model.predict(X_star)

t2 = time.time()
print('Running time 2 (predict):', t2 - t0)

error_u = np.mean(np.abs(u_star - u_pred) ** 2)  # L2
print('Error u: %e' % (error_u))

# %%
# save data
# np.savez('data.npz',
#          X_train=X_train,
#          u_train=u_train,
#          X_star=X_star,
#          u_star=u_star,
#          Exact=Exact,
#          u_pred=u_pred,
#          f_pred=f_pred,
#          iter=model.iter,
#          Loss_u=model.Loss_u,
#          Loss_f=model.Loss_f,
#          Loss=model.Loss,
#          Mu=model.Mu
#          )

# load data
# data = np.load('data.npz')
# X_u_train = data['X_train']


# %%
# Visualization - loss
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure()
w = np.linspace(0, model.iter, model.iter)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title(r'\textbf{Loss function}')
# plt.plot(w, model.Loss, label='Loss')
plt.plot(w, model.Loss_u, label='Loss_u')
plt.plot(w, model.Loss_f, label='Loss_f')
plt.plot(w, model.Loss, label='Loss')
plt.legend()

# plt.savefig('loss.png', dpi=600)
# plt.savefig('loss.pdf', dpi=600)

#%%
# Visualization - μ
fig = plt.figure()
w = np.linspace(0, model.iter, model.iter)
fs = 16
plt.xlabel('Iter', fontsize=fs)
plt.ylabel('$\mu$', fontsize=fs)
plt.title('${\mu}_{pred}$: %.5f, ${\mu}_{true}$: %.5f' % (model.Mu[-1], mu_exact), fontsize=fs)
plt.plot(w, np.full_like(model.Mu, mu_exact), color='silver', linewidth=5, label='Truth')
plt.plot(w, model.Mu, label='Prediction')
plt.legend()

# plt.savefig('μ.png', dpi=600)
# plt.savefig('μ.pdf', dpi=600)


# %%
plt.show()

t3 = time.time()
print('Running time 3:', t3 - t0)





