# %%
# Libraries and Dependencies
import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from KP_compute_APhi import KP_compute_APhi


# %%
t0 = time.time()

np.random.seed(1234)
torch.manual_seed(1234)
torch.set_default_dtype(torch.float64)

t1 = time.time()

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
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers):

        # data
        self.x = torch.tensor(X, requires_grad=True).double().to(device)
        self.u = torch.tensor(u).double().to(device)

        # some computation
        self.A_torch, self.Phi_inv_torch = self.computation_KP(self.x)

        # layers
        self.layers = layers

        # parameter setting
        '''self.lamda = torch.tensor([0.001], requires_grad=True).to(device)  # initial lamda
        self.lamda = torch.nn.Parameter(self.lamda)'''
        self.lamda = torch.nn.Parameter(torch.tensor(torch.rand(1), requires_grad=True).to(device))

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lamda', self.lamda)

        # optimizer
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=500,
            max_eval=1000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        # self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.iter = 0

        self.Loss_u = []
        self.Loss_f = []
        self.Loss = []
        self.Lamda = []

    def net_u(self, x):
        u = self.dnn(x)
        return u

    def net_f(self, x):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        mu = 2
        f = u_x - self.lamda * u - torch.exp(-x)
        return f

    def computation_KP(self, x):
        t1 = time.time()
        xx = np.unique(x.cpu().detach().numpy())
        A, Phi = KP_compute_APhi(xx, 1, 3 / 2)
        lamda = 0.01
        Phi = Phi + lamda * A
        n = Phi.shape[0]
        I = identity(n).tocsc()
        Phi_inv = spsolve(Phi, I)
        A_torch, Phi_inv_torch = torch.tensor(A.toarray()), torch.tensor(Phi_inv.toarray())
        t2 = time.time()
        print('KP time: ', t2 - t1)
        return A_torch.to(device), Phi_inv_torch.to(device)

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x)
        f_pred = self.net_f(self.x)

        # L2 norm
        loss_u_L2 = torch.mean(torch.abs(self.u - u_pred) ** 2)
        loss_f_L2 = torch.mean(torch.abs(f_pred) ** 2)

        # KP - RKHS ≈ Y^T K^(-1) Y
        pred = self.u - u_pred
        # b = torch.linalg.solve(self.Phi_torch, pred)
        loss_u_KP = torch.div(torch.abs(pred.T @ self.A_torch @ self.Phi_inv_torch @ pred), len(pred))
        # b = torch.linalg.solve(self.Phi_torch, f_pred)
        loss_f_KP = torch.div(torch.abs(f_pred.T @ self.A_torch @ self.Phi_inv_torch @ f_pred), len(f_pred))

        loss_u = loss_u_KP  # loss_u_L2, loss_u_KP, loss_u_RKHS, loss_u_Sobolev
        loss_f = loss_f_KP  # loss_f_L2, loss_f_KP, loss_f_RKHS, loss_f_Sobolev
        self.Loss_u.append(float(loss_u))
        self.Loss_f.append(float(loss_f))
        w1 = 1
        w2 = 1
        loss = w1 * loss_u + w2 * loss_f
        self.Loss.append(float(loss))
        self.Lamda.append(float(self.lamda))

        loss.backward(retain_graph=True)
        # self.optimizer.step()
        self.iter += 1
        if self.iter % 10 == 0:
            print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e, lamda: %.5f' % (
            self.iter, loss.item(), loss_u.item(), loss_f.item(), self.lamda))

        return loss

    def train(self):
        self.dnn.train()

        # Backward and optimize
        self.optimizer.step(self.loss_func)

        '''# Set the maximum number of iterations
        max_iter = 5000
        # Train the model for max_iter iterations
        for _ in range(max_iter):
            # Backward and optimize
            self.optimizer.step(self.loss_func)'''

    def predict(self, X):
        x = torch.tensor(X, requires_grad=True).double().to(device)

        self.dnn.eval()
        u = self.net_u(x)
        f = self.net_f(x)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


# %%
# Configurations
noise = 0.0

N = 50  # <=2000
layers = [1, 20, 20, 20, 20, 20, 20, 1]

lamda_exact = -2

data = scipy.io.loadmat('stiff_solution.mat')

x = data['x'].flatten()[:, None].astype(np.float64)  # x.shape: (2000, 1)
Exact = np.real(data['usol']).T.astype(np.float64)  # Exact.shape: (2000, 1)

X_star = x
u_star = Exact

# X_train
idx = np.sort(np.random.choice(x.shape[0], N, replace=False))
X_train = x[idx]  # shape: [N, 1]
u_train = Exact[idx]

# print(u_star.size, np.linalg.norm(u_star))     # Relative L2 Error, 2000, 23.840652264224452

# Training
model = PhysicsInformedNN(X_train, u_train, layers)
model.train()

t1 = time.time()
print('Running time 1 (train):', t1 - t0)

lamda_pre = model.lamda.detach().cpu().numpy()
print('lamda_pre: ', lamda_pre[0])
print('lamda_exact: ', lamda_exact)
error_lamda = np.abs(lamda_pre - lamda_exact) / np.abs(lamda_exact)
print('Error lamda: %e' % error_lamda)

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
#          iter=model.iter,
#          Loss_u=model.Loss_u,
#          Loss_f=model.Loss_f,
#          Loss=model.Loss,
#          Lamda=model.Lamda,
#          X_star=X_star,
#          u_star=u_star,
#          u_pred=u_pred,
#          f_pred=f_pred
#          )

# load data
# data = np.load('data.npz')
# X_u_train = data['X_u_train']


# %%
# Visualization - training data
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure()
plt.scatter(X_train, u_train, label='X_train', s=1)
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.title('Training data')
plt.legend()

# plt.savefig('training_data.png', dpi=600)
# plt.savefig('training_data.pdf', dpi=600)

# %%
# Visualization - loss
fig = plt.figure()
w = np.linspace(0, model.iter, model.iter)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Loss')
plt.plot(w, model.Loss_u, label='Loss_u')
plt.plot(w, model.Loss_f, label='Loss_f')
plt.plot(w, model.Loss, label='Loss')
plt.legend()

# plt.savefig('loss.png', dpi=600)
# plt.savefig('loss.pdf', dpi=600)

# %%
# Visualization - lamda
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(5, 4))
w = np.linspace(0, model.iter, model.iter)
plt.xlabel('Iter', fontsize=20)
plt.ylabel('$\lambda$', fontsize=20)
plt.title('${\lambda}_{pred}$: %.5f, ${\lambda}_{true}$: %.5f' % (model.Lamda[-1], lamda_exact))
plt.plot(w, np.full_like(model.Lamda, lamda_exact), color='silver', linewidth=5, label='Truth')
plt.plot(w, model.Lamda, label='Prediction')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

# plt.savefig('lambda.png', dpi=600, bbox_inches='tight')
# plt.savefig('lambda.pdf', dpi=600, bbox_inches='tight')

# %%
# Visualization - u(x)
fig = plt.figure(figsize=(5, 4))
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$u(t)$', fontsize=20)
# plt.title('$u(t)$')
plt.plot(X_star, u_star, color='silver', linewidth=5, label='Truth')
plt.plot(X_star, u_pred, label='Prediction')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

# plt.savefig('u(t).png', dpi=600, bbox_inches='tight')
# plt.savefig('u(t).pdf', dpi=600, bbox_inches='tight')

# #%% Save the data of the above two figures
# np.savez('data_stiff_KP32_inverse.npz',
#          iter=model.iter,
#          Lamda=model.Lamda,
#          X_star=X_star,
#          u_star=u_star,
#          u_pred=u_pred
#          )

# %%
# Visualization - f(x)
fig = plt.figure()
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
# plt.title('$f(t)$')
plt.plot(X_star, np.zeros_like(f_pred), color='silver', linewidth=5, label='Truth')
plt.plot(X_star, f_pred, label='Prediction')
plt.legend()

# plt.savefig('f(t).png', dpi=600)
# plt.savefig('f(t).pdf', dpi=600)

plt.show()

t3 = time.time()
print('Running time 3:', t3 - t0)


