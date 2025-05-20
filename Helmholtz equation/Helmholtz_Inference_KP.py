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


# %%
t0 = time.time()

np.random.seed(111)
torch.manual_seed(222)

torch.set_default_dtype(torch.float64)

# Initialize the cuda environment
torch.cuda.init()

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
    def __init__(self, X_u, u, X_f, layers, k):

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).double().to(device)
        self.y_u = torch.tensor(X_u[:, 1:2], requires_grad=True).double().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).double().to(device)
        self.y_f = torch.tensor(X_f[:, 1:2], requires_grad=True).double().to(device)
        self.u = torch.tensor(u).double().to(device)

        # some computation
        # loss_u
        self.idx1, _ = self.BC(self.x_u, self.y_u, 1)
        self.idx2, _ = self.BC(self.x_u, self.y_u, 2)
        self.idx3, _ = self.BC(self.x_u, self.y_u, 3)
        self.idx4, _ = self.BC(self.x_u, self.y_u, 4)
        self.A_torch_u1, self.Phi_inv_u1 = self.computation_KP_u(self.x_u, self.y_u, 1)
        self.A_torch_u2, self.Phi_inv_u2 = self.computation_KP_u(self.x_u, self.y_u, 2)
        self.A_torch_u3, self.Phi_inv_u3 = self.computation_KP_u(self.x_u, self.y_u, 3)
        self.A_torch_u4, self.Phi_inv_u4 = self.computation_KP_u(self.x_u, self.y_u, 4)
        # loss_f
        self.A_torch_f, self.Phi_inv_f = self.computation_KP_f(self.x_f, self.y_f)

        # layers
        self.layers = layers
        self.k = k

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
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
        # self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # self.optimizer = torch.optim.RMSprop(self.dnn.parameters(), lr=0.001, alpha=0.9)

        self.iter = 0

        self.Loss_u = []
        self.Loss_f = []
        self.Loss = []

    def net_u(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u

    def net_f(self, x, y):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, y)  # x.shape = (2500, 1), u.shape = (2500, 1)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]  # u_x.shape = (2500, 1)
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


    def BC(self, x1, x2, bc):  # x1:x, x2:y
        idx = 0
        if bc == 1:  # BC1: x = -1
            idx = np.where(x1.cpu().detach().numpy() == -1)[0]
            x = np.sort(x2[idx].cpu().reshape(-1).detach().numpy())
        elif bc == 2:  # BC2: x = 1
            idx = np.where(x1.cpu().detach().numpy() == 1)[0]
            x = np.sort(x2[idx].cpu().reshape(-1).detach().numpy())
        elif bc == 3:  # BC3: y = -1
            idx = np.where(x2.cpu().detach().numpy() == -1)[0]
            x = np.sort(x1[idx].cpu().reshape(-1).detach().numpy())
        elif bc == 4:  # BC4: y = 1
            idx = np.where(x2.cpu().detach().numpy() == 1)[0]
            x = np.sort(x1[idx].cpu().reshape(-1).detach().numpy())
        return idx, x

    def computation_KP_u(self, x1, x2, a):  # x1:x, x2:y
        _, x = self.BC(x1, x2, a)
        A, Phi = KP_compute_APhi(x, 10, 3 / 2)
        lamda = 0.01
        n = Phi.shape[0]
        I = identity(n).tocsc()
        Phi = Phi + lamda * I
        # Phi = Phi + lamda * A
        Phi_inv = spsolve(Phi, I)
        A_torch, Phi_inv_torch = torch.tensor(A.toarray()), torch.tensor(Phi_inv.toarray())
        return A_torch.to(device), Phi_inv_torch.to(device)

    def computation_KP_f(self, x1, x2):  # x1:x, x2:y
        t1 = time.time()
        xx1, xx2 = np.unique(x1.cpu().detach().numpy()), np.unique(x2.cpu().detach().numpy())
        A1, Phi1 = KP_compute_APhi(xx1, 10, 3 / 2)
        A2, Phi2 = KP_compute_APhi(xx2, 10, 3 / 2)
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

        # L2 norm
        loss_u_L2 = torch.mean(torch.abs(self.u - u_pred) ** 2)

        # BC1: x = -1
        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u[self.idx1] - u_pred[self.idx1]
        # b = torch.linalg.solve(self.Phi_torch_u1, pred)
        # loss_u_KP1 = torch.div(torch.abs(pred.T @ self.A_torch_u1 @ b), len(pred))
        loss_u_KP1 = torch.div(torch.abs(pred.T @ self.A_torch_u1 @ self.Phi_inv_u1 @ pred), len(pred))

        # BC2: x = 1
        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u[self.idx2] - u_pred[self.idx2]
        # b = torch.linalg.solve(self.Phi_torch_u2, pred)
        # loss_u_KP2 = torch.div(torch.abs(pred.T @ self.A_torch_u2 @ b), len(pred))
        loss_u_KP2 = torch.div(torch.abs(pred.T @ self.A_torch_u2 @ self.Phi_inv_u2 @ pred), len(pred))

        # BC3: y = -1
        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u[self.idx3] - u_pred[self.idx3]
        # b = torch.linalg.solve(self.Phi_torch_u3, pred)
        # loss_u_KP3 = torch.div(torch.abs(pred.T @ self.A_torch_u3 @ b), len(pred))
        loss_u_KP3 = torch.div(torch.abs(pred.T @ self.A_torch_u3 @ self.Phi_inv_u3 @ pred), len(pred))

        # BC4: y = 1
        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u[self.idx4] - u_pred[self.idx4]
        # b = torch.linalg.solve(self.Phi_torch_u4, pred)
        # loss_u_KP4 = torch.div(torch.abs(pred.T @ self.A_torch_u4 @ b), len(pred))
        loss_u_KP4 = torch.div(torch.abs(pred.T @ self.A_torch_u4 @ self.Phi_inv_u4 @ pred), len(pred))

        loss_u_KP = loss_u_KP1 + loss_u_KP2 + loss_u_KP3 + loss_u_KP4


        #################################### loss_f ####################################
        f_pred = self.net_f(self.x_f, self.y_f)

        # L2 norm
        tL2 = time.time()
        loss_f_L2 = torch.mean(torch.abs(f_pred) ** 2)

        # KP - RKHS ≈ Y^T K^(-1) Y
        # b = torch.linalg.solve(self.Phi_torch_f, f_pred)
        # loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch_f @ b), len(f_pred))
        loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch_f @ self.Phi_inv_f @ f_pred), len(f_pred))


        loss_u = loss_u_KP  # loss_u_L2, loss_u_KP, loss_u_RKHS, loss_u_Sobolev
        loss_f = loss_f_KP  # loss_f_L2, loss_f_KP, loss_f_RKHS, loss_f_Sobolev
        self.Loss_u.append(float(loss_u))
        self.Loss_f.append(float(loss_f))
        w1 = 1
        w2 = 1
        loss = w1 * loss_u + w2 * loss_f
        self.Loss.append(float(loss))
        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
            self.iter, loss.item(), loss_u.item(), loss_f.item()))

        loss.backward()
        self.iter += 1
        # print('Iter %d, Loss_u_L2: %.5e, Loss_f_L2: %.5e' % (self.iter, loss_u_L2.item(), loss_f_L2.item()))
        return loss

    def train(self):
        self.dnn.train()
        # Backward and optimize
        self.optimizer.step(self.loss_func)

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
k = 1
noise = 0.0

N_u = 100  # <=2200
N_1 = 100  # <=500
N_2 = 100  # <=600
layers = [2, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('helmholtz_solution.mat')

x = data['x'].flatten()[:, None].astype(np.float64)  # x.shape: (500, 1)
y = data['y'].flatten()[:, None].astype(np.float64)  # y.shape: (600, 1)
Exact = np.real(data['usol']).T.astype(np.float64)  # Exact.shape: (600, 500)

X, Y = np.meshgrid(x, y)  # X: (600, 500), T: (600, 500)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))  # (30000, 2)
u_star = Exact.flatten()[:, None]  # (30000, 1)

# Doman bounds
# x = -1
xx1 = np.stack((X[:, 0], Y[:, 0]), axis=1)  # (600, 2)
uu1 = Exact[:, 0:1]  # (600, 1)
# x = 1
xx2 = np.stack((X[:, -1], Y[:, 0]), axis=1)  # (600, 2)
uu2 = Exact[:, -1:]  # (600, 1)
# y = -1
xx3 = np.stack((X[0, :].T, Y[0, :].T), axis=1)  # (500, 2)
uu3 = Exact[0:1, :].T  # (500, 1)
# y = 1
xx4 = np.stack((X[0, :].T, Y[-1, :].T), axis=1)  # (500, 2)
uu4 = Exact[-1:, :].T  # (500, 1)

# X_u_train, u_train
X_u_train = np.vstack([xx1, xx2, xx3, xx4])  # (2200, 2)
u_train = np.vstack([uu1, uu2, uu3, uu4])  # (2200, 1)
idx = np.sort(np.random.choice(X_u_train.shape[0], N_u, replace=False))  # (N_u, )
X_u_train = X_u_train[idx, :]  # (N_u, 2)
u_train = u_train[idx, :]  # (N_u, 1)

# X_f_train
idx = np.sort(np.random.choice(x.shape[0], N_1, replace=False))
Xx = x[idx]  # len(Xt) = N_1
idx = np.sort(np.random.choice(y.shape[0], N_2, replace=False))
Xy = y[idx]  # len(Xx) = N_2
X_f_train = np.column_stack((np.repeat(Xx, Xy.shape[0], axis=0), np.tile(Xy, (Xx.shape[0], 1))))

print(u_star.size, np.linalg.norm(u_star))     # Relative L2 Error, 300000, 205.53730207912182

# %%
# Training
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, k)
t11 = time.time()
print('Running time 11 (init):', t11 - t0)
model.train()

t1 = time.time()
print('Running time 1 (train):', t1 - t0)

# predict
u_pred, f_pred = model.predict(X_star)  # u_pred.shape: (300000, 1), f_pred.shape: (300000, 1)

t2 = time.time()
print('Running time 2 (predict):', t2 - t0)

error_u = np.mean(np.abs(u_star - u_pred) ** 2)  # L2
print('Error u: %e' % (error_u))

U_pred = np.reshape(u_pred, Exact.shape)  # U_pred.shape: (600, 500)
Error = np.abs(Exact - U_pred)

# %%
# save data
# np.savez('data.npz',
#          X_u_train=X_u_train,
#          X_f_train=X_f_train,
#          X=X,
#          Y=Y,
#          Exact=Exact,
#          U_pred=U_pred,
#          iter=model.iter,
#          Loss_u=model.Loss_u,
#          Loss_f=model.Loss_f,
#          Loss=model.Loss
#          )

# load data
# data = np.load('data.npz')
# X_u_train = data['X_u_train']


# %%
# Visualization - training data
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('font', family='serif', size=20)

fig = plt.figure()
plt.scatter(X_u_train[:, 0:1], X_u_train[:, 1:2], c='r', label='X_u_train', s=1)
plt.scatter(X_f_train[:, 0:1], X_f_train[:, 1:2], c='b', label='X_f_train', s=1)
plt.xlabel(r'\textit{x}')
plt.ylabel(r'\textit{y}')
plt.title(r'\textbf{Training data}')
plt.legend()

# plt.savefig('training_data.png', dpi=600)
# plt.savefig('training_data.pdf', dpi=600)

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
# Visualization - loss
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
# Visualization - u(t, x)

# together

fig, axs = plt.subplots(1, 3, figsize=(24, 8))

h1 = axs[0].imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.10)
cbar1 = fig.colorbar(h1, cax=cax1)
cbar1.ax.tick_params(labelsize=22)

axs[0].set_xlabel('$x$', size=22)
axs[0].set_ylabel('$y$', size=22)

# axs[0].plot(X_u_train[:,0], X_u_train[:,1], 'kx', label='Data (%d points)' % (X_u_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)

# axs[0].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[0].set_title(r'\textbf{Prediction}', fontsize=22)
axs[0].tick_params(labelsize=22)

h2 = axs[1].imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.10)
cbar2 = fig.colorbar(h2, cax=cax2)
cbar2.ax.tick_params(labelsize=22)

axs[1].set_xlabel('$x$', size=22)
axs[1].set_ylabel('$y$', size=22)

# axs[1].plot(X_u_train[:,0], X_u_train[:,1], 'kx', label='Data (%d points)' % (X_u_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)

# axs[1].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[1].set_title(r'\textbf{Truth}', fontsize=22)
axs[1].tick_params(labelsize=22)

h3 = axs[2].imshow(np.abs(U_pred - Exact).T, interpolation='nearest', cmap='rainbow',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='equal')
divider3 = make_axes_locatable(axs[2])
cax3 = divider3.append_axes("right", size="5%", pad=0.10)
cbar3 = fig.colorbar(h3, cax=cax3)
cbar3.ax.tick_params(labelsize=22)

axs[2].set_xlabel('$x$', size=22)
axs[2].set_ylabel('$y$', size=22)

# axs[2].plot(X_u_train[:,0], X_u_train[:,1], 'kx', label='Data (%d points)' % (X_u_train.shape[0]),
#             markersize=4, clip_on=False, alpha=1.0)

# axs[2].legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
axs[2].set_title(r'\textbf{Absolute error}', fontsize=22)
axs[2].tick_params(labelsize=22)

plt.tight_layout()

# plt.savefig('u(x, y).png', dpi=600)
# plt.savefig('u(x, y).pdf', dpi=600)


plt.show()

t3 = time.time()
print('Running time 3:', t3 - t0)





