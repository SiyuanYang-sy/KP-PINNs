# %%
# Libraries and Dependencies
import torch
from collections import OrderedDict
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io

from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.sparse import kron
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from KP_compute_APhi import KP_compute_APhi
from matplotlib import cm
import plotly.graph_objects as go
from matplotlib.ticker import FormatStrFormatter


# %%
t0 = time.time()

np.random.seed(1234)
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
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).double().to(device)
        self.y_u = torch.tensor(X_u[:, 1:2], requires_grad=True).double().to(device)
        self.u = torch.tensor(u).double().to(device)

        # some computation
        self.A_torch, self.Phi_inv = self.computation_KP(self.x_u, self.y_u)

        # layers
        self.layers = layers

        # parameter setting
        self.k = torch.tensor([torch.rand(1)], requires_grad=True).to(device)  # initial k, k_exact = 1.0
        self.k = torch.nn.Parameter(self.k)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('k', self.k)

        # self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=10000,  # 50000
            max_eval=50000,
            history_size=100,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            # tolerance_change=1e-5,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.iter = 0

        self.Loss_u = []
        self.Loss_f = []
        self.Loss = []
        self.K = []

    def net_u(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u

    def net_f(self, x, y):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, y)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        p1 = (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        p2 = 2 * ((math.pi) ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        p3 = 2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
        p4 = 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
        p = p1 - p2 + p3 + p4
        f = u_xx + u_yy + ((self.k) ** 2) * u - p
        return f

    def computation_KP(self, x1, x2):  # x1:x, x2:y
        t1 = time.time()
        xx1, xx2 = np.unique(x1.cpu().detach().numpy()), np.unique(x2.cpu().detach().numpy())
        A1, Phi1 = KP_compute_APhi(xx1, 2, 1 / 2)
        A2, Phi2 = KP_compute_APhi(xx2, 2, 1 / 2)
        A_Kron = kron(A1, A2)
        lamda = 0.01
        n1, n2 = Phi1.shape[0], Phi2.shape[0]
        I1, I2 = identity(n1).tocsc(), identity(n2).tocsc()
        Phi1, Phi2 = Phi1 + lamda * I1, Phi2 + lamda * I2
        # Phi1, Phi2 = Phi1 + lamda * A1, Phi2 + lamda * A2
        Phi1_inv, Phi2_inv = spsolve(Phi1, I1), spsolve(Phi2, I2)
        Phi_inv_Kron = kron(Phi1_inv, Phi2_inv)
        A_torch, Phi_inv = torch.tensor(A_Kron.toarray()), torch.tensor(Phi_inv_Kron.toarray())
        t2 = time.time()
        print('KP time: ', t2 - t1)
        return A_torch.to(device), Phi_inv.to(device)

    def loss_func(self):
        self.optimizer.zero_grad()

        #################################### loss_u #####################################
        u_pred = self.net_u(self.x_u, self.y_u)

        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u - u_pred
        # b = torch.linalg.solve(self.Phi_torch, pred)
        # loss_u_KP = torch.div(torch.abs(pred.T @ self.A_torch @ b), len(pred))
        loss_u_KP = torch.div(torch.abs(pred.T @ self.A_torch @ self.Phi_inv @ pred), len(pred))

        #################################### loss_f ####################################
        f_pred = self.net_f(self.x_u, self.y_u)

        # KP - RKHS ≈ Y^T K^(-1) Y
        # b = torch.linalg.solve(self.Phi_torch, f_pred)
        # loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch @ b), len(f_pred))
        loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch @ self.Phi_inv @ f_pred), len(f_pred))

        loss_u = loss_u_KP  # loss_u_L2, loss_u_KP, loss_u_RKHS, loss_u_Sobolev
        loss_f = loss_f_KP  # loss_f_L2, loss_f_KP, loss_f_RKHS, loss_f_Sobolev
        self.Loss_u.append(float(loss_u))
        self.Loss_f.append(float(loss_f))
        w1 = 1
        w2 = 1
        loss = w1 * loss_u + w2 * loss_f
        self.Loss.append(float(loss))
        self.K.append(float(abs(self.k)))
        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e, k: %.5f' % (
            self.iter, loss.item(), loss_u.item(), loss_f.item(), abs(self.k)))

        loss.backward()
        self.iter += 1
        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_func)

        '''# Backward and optimize
        self.optimizer.step(self.loss_func)'''

        # # Set the maximum number of iterations
        # max_iter = 3000
        # # Train the model for max_iter iterations
        # for _ in range(max_iter):
        #     # Backward and optimize
        #     self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)

        self.dnn.eval()
        u = self.net_u(x, y)
        f = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


# %%
# Configurations
k_exact = 1.0
noise = 0.0

N_1 = 30  # <=500, 50
N_2 = 30  # <=600, 80
layers = [2, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('helmholtz_solution.mat')

x = data['x'].flatten()[:, None].astype(np.float64)  # (500, 1)
y = data['y'].flatten()[:, None].astype(np.float64)  # (600, 1)
Exact = np.real(data['usol']).T.astype(np.float64)  # (600, 500)

X, Y = np.meshgrid(x, y)  # X: (600, 500), T: (600, 500)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))  # (30000, 2)
u_star = Exact.flatten()[:, None]  # (30000, 1)

# X_train
idx1 = np.sort(np.random.choice(x.shape[0], N_1, replace=False))
Xx = x[idx1]  # len(Xt) = N_1
idx2 = np.sort(np.random.choice(y.shape[0], N_2, replace=False))
Xy = y[idx2]  # len(Xx) = N_2
u_train = Exact[idx2[:, None], idx1]
X_train = np.column_stack((np.repeat(Xx, Xy.shape[0], axis=0), np.tile(Xy, (Xx.shape[0], 1))))
u_train = u_train.flatten(order='F').reshape(-1, 1)

print(u_star.size, np.linalg.norm(u_star))     # Relative L2 Error, 300000, 205.53730207912182

# %%
# Training
model = PhysicsInformedNN(X_train, u_train, layers)
t11 = time.time()
print('Running time 11 (init):', t11 - t0)
model.train()

t1 = time.time()
print('Running time 1 (train):', t1 - t0)

k_pre = model.K
print('k_pre: ', k_pre[-1])
print('k_exact: ', k_exact)
error_k = np.abs(k_pre[-1] - k_exact) / np.abs(k_exact)
print('Error k: %e' % error_k)

# predict
u_pred, f_pred = model.predict(X_star)

t2 = time.time()
print('Running time 2 (predict):', t2 - t0)

error_u = np.mean(np.abs(u_star - u_pred) ** 2)  # L2
# relative_L2 = np.sqrt(error_u * u_star.size) / np.linalg.norm(u_star)
# error_u = np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)  # Relative L2 Error
print('Error u: %e' % (error_u))

U_pred = np.reshape(u_pred, Exact.shape)  # U_pred.shape: (600, 500)
Error = np.abs(Exact - U_pred)


# %%
# save data
# np.savez('data.npz',
#          X_train=X_train,
#          X=X,
#          Y=Y,
#          Exact=Exact,
#          U_pred=U_pred,
#          iter=model.iter,
#          Loss_u=model.Loss_u,
#          Loss_f=model.Loss_f,
#          Loss=model.Loss,
#          K=model.K
#          )

# load data
# data = np.load('data.npz')
# X_u_train = data['X_u_train']


# %%
# Visualization - training data
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

fig = plt.figure()
plt.scatter(X_train[:, 0:1], X_train[:, 1:2], c='r', label='X_train', s=1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'\textbf{Training data}')
plt.legend()

# plt.savefig('training_data.png', dpi=600)
# plt.savefig('training_data.pdf', dpi=600)

# %%
# Visualization - loss
fig = plt.figure()
w = np.linspace(0, model.iter, model.iter)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Loss function')
plt.plot(w, model.Loss_u, label='Loss_u')
plt.plot(w, model.Loss_f, label='Loss_f')
plt.plot(w, model.Loss, label='Loss')
plt.legend()

# plt.savefig('loss.png', dpi=600)
# plt.savefig('loss.pdf', dpi=600)

# %%
# Visualization - k
fig = plt.figure(figsize=(5, 4))
w = np.linspace(0, model.iter, model.iter)
fs = 20
plt.xlabel('Iter', fontsize=fs)
plt.ylabel('$k$', fontsize=fs)
plt.title('${k}_{pred}$: %.5f, ${k}_{true}$: %.5f' % (model.K[-1], k_exact), fontsize=fs)
plt.plot(w, np.full_like(model.K, k_exact), color='silver', linewidth=5, label='Truth')
plt.plot(w, model.K, label='Prediction')
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# plt.savefig('k.png', dpi=600)
# plt.savefig('k.pdf', dpi=600, bbox_inches='tight')

# %%
# Visualization - plot_surface

# together

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

surf1 = ax1.plot_surface(X, Y, Exact, cmap=cm.coolwarm, antialiased=False)
ax1.set_xlabel(r'\textit{x}')
ax1.set_ylabel(r'\textit{y}')
ax1.set_zlabel(r'\textit{u}')
ax1.set_title(r'\textbf{Truth}')

surf2 = ax2.plot_surface(X, Y, U_pred, cmap=cm.coolwarm, antialiased=False)
ax2.set_xlabel(r'\textit{x}')
ax2.set_ylabel(r'\textit{y}')
ax2.set_zlabel(r'\textit{u}')
ax2.set_title(r'\textbf{Prediction}')

plt.tight_layout()

# plt.savefig('plot_surface.png', dpi=600)
# plt.savefig('plot_surface.pdf', dpi=600)



# %%
# Visualization - go_surface
# fig = go.Figure(data=[go.Surface(x = X, y = Y, z = Exact)])
# fig.update_layout(
#                     title='Truth',
#                     scene = dict(
#                     xaxis_title='x',
#                     yaxis_title='y',
#                     zaxis_title='u'
#                     ),
#                   width=800, height=800,
#                   )
# fig.show()
#
# fig = go.Figure(data=[go.Surface(x = X, y = Y, z = U_pred)])
# fig.update_layout(
#                     title='Prediction',
#                     scene = dict(
#                     xaxis_title='x',
#                     yaxis_title='y',
#                     zaxis_title='u'
#                     ),
#                   width=800, height=800,
#                   )
# fig.show()


# %%
# Visualization - u(t, x)

# together

fig, axs = plt.subplots(1, 3, figsize=(12, 6))   # (24, 8)

h1 = axs[0].imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.10)
# cbar1 = fig.colorbar(h1, cax=cax1)
cbar1 = fig.colorbar(h1, cax=cax1, ticks=[-1.00, 0.00, 1.00])
cbar1.ax.tick_params(labelsize=20)
cbar1.ax.set_yticklabels(['-1.00', '0.00', '1.00'])

axs[0].set_xlabel('$x$', size=20)
axs[0].set_ylabel('$y$', size=20)

# axs[0].plot(X_train[:,0], X_train[:,1], 'kx', label='Data (%d points)' % (X_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)
#
# axs[0].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[0].set_title(r'\textbf{Prediction}', fontsize=20)
axs[0].tick_params(labelsize=20)

h2 = axs[1].imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.10)
cbar2 = fig.colorbar(h2, cax=cax2)
cbar2.ax.tick_params(labelsize=20)

axs[1].set_xlabel('$x$', size=20)
axs[1].set_ylabel('$y$', size=20)

# axs[1].plot(X_train[:,0], X_train[:,1], 'kx', label='Data (%d points)' % (X_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)
#
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[1].set_title(r'\textbf{Truth}', fontsize=20)
axs[1].tick_params(labelsize=20)

h3 = axs[2].imshow(np.abs(U_pred - Exact).T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider3 = make_axes_locatable(axs[2])
cax3 = divider3.append_axes("right", size="5%", pad=0.10)
cbar3 = fig.colorbar(h3, cax=cax3)
cbar3.ax.tick_params(labelsize=20)
cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))

axs[2].set_xlabel('$x$', size=20)
axs[2].set_ylabel('$y$', size=20)

# axs[2].plot(X_train[:,0], X_train[:,1], 'kx', label='Data (%d points)' % (X_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)
#
# axs[2].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[2].set_title(r'\textbf{Absolute error}', fontsize=20)
axs[2].tick_params(labelsize=20)

plt.tight_layout()

# plt.savefig('u(x, y).png', dpi=600)
# plt.savefig('u(x, y).pdf', dpi=600, bbox_inches='tight')


# #%% Save the data of the above figure and k
# np.savez('data_helmholtz_KP12_inverse.npz',
#          iter=model.iter,
#          K=model.K,
#          x=x,
#          y=y,
#          U_pred=U_pred,
#          Exact=Exact
#          )

plt.show()

t3 = time.time()

print('Running time 3:', t3 - t0)




