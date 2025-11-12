# MPC usando la NN como modelo (linealización online)
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import matplotlib.pyplot as plt

# ---- Configuración ----
dt = 0.02
nx = 2
nu = 1
m = 0.04
R = 0.05
Jb = 2/5 * m * R**2

g = 9.81
a = (m * g)/(m+(Jb/(R**2)))

# Cargar modelo si no está en memoria:
# Importa tu clase MLPModel

# 1. Crear la instancia del modelo con la misma arquitectura
# --- Definir arquitectura EXACTA ---


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=1)
        return self.net(xu)


# --- Cargar el state_dict ---
model = MLPModel()  # crear modelo vacío
state_dict = torch.load("ball_beam_model.pth")  # esto es OrderedDict
model.load_state_dict(state_dict)
model.eval()

# MPC params
Np = 20
Q = np.diag([200.0, 1.0])
R = 0.1
u_min, u_max = -0.25, 0.25

# simulación
sim_time = 6.0
steps = int(sim_time / dt)
x0 = np.array([0.05, 0.0])   # estado inicial
xref = np.array([0.25, 0.0])  # setpoint

# helper: obtener f(x,u), Jx, Ju (f devuelve delta_x)


def linearize_nn(model, x_np, u_np):
    # x_np: (nx,), u_np: (1,)
    x_t = torch.tensor(x_np.reshape(
        1, -1), dtype=torch.float32, requires_grad=True)
    u_t = torch.tensor(u_np.reshape(
        1, -1), dtype=torch.float32, requires_grad=True)
    # forward
    f = model(x_t, u_t)  # shape (1, nx) -> delta x
    f0 = f.detach().numpy().squeeze()  # (nx,)
    # jacobianos
    Jx = np.zeros((nx, nx), dtype=float)
    Ju = np.zeros((nx, nu), dtype=float)
    for i in range(nx):
        model.zero_grad()
        if x_t.grad is not None:
            x_t.grad.zero_()
        if u_t.grad is not None:
            u_t.grad.zero_()
        f[0, i].backward(retain_graph=True)
        # grad wrt x
        grad_x = x_t.grad.detach().numpy().squeeze(
        ).copy() if x_t.grad is not None else np.zeros(nx)
        grad_u = u_t.grad.detach().numpy().squeeze(
        ).copy() if u_t.grad is not None else np.zeros(nu)
        Jx[i, :] = grad_x
        Ju[i, :] = grad_u
        # clear grads for next output
        x_t.grad.zero_()
        u_t.grad.zero_()
    # Note: above fills rows as ∂f_i/∂x_j; we want Jx as matrix with shape (nx,nx)
    # but built as rows already. Keep as is.
    # Return Jx (nx x nx) and Ju (nx x nu)
    return f0, Jx, Ju


# almacenamiento
X_nn = np.zeros((nx, steps+1))
U_hist = np.zeros(steps)
X_nn[:, 0] = x0.copy()
xk = x0.copy()
u_prev = 0.0

for k in range(steps):
    # linearizar la NN en (xk, u_prev)
    f0, Jx, Ju = linearize_nn(model, xk, np.array([u_prev], dtype=float))

    # construir A, B, c para el modelo afín: x_{k+1} ≈ A x_k + B u_k + c
    A_lin = np.eye(nx) + Jx         # I + Jx
    B_lin = Ju                      # Ju
    c_lin = f0 - Jx @ xk - Ju.flatten() * u_prev

    # Precomputar matrices para horizonte (usaremos A_lin,B_lin,c_lin constantes)
    # Variables CVX
    X = cp.Variable((nx, Np+1))
    U = cp.Variable((Nu := nu, Np))
    cost = 0
    constr = []
    constr += [X[:, 0] == xk]

    for i in range(Np):
        # cost
        err = X[:, i] - xref
        cost += cp.quad_form(err, Q) + R * cp.sum_squares(U[:, i])
        # dynamics: X_{i+1} = A_lin X_i + B_lin U_i + c_lin
        constr += [X[:, i+1] == A_lin @ X[:, i] + B_lin @ U[:, i] + c_lin]
        constr += [u_min <= U[:, i], U[:, i] <= u_max]
    # terminal cost
    errN = X[:, Np] - xref
    cost += cp.quad_form(errN, Q * 0.5)

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if U.value is None:
        u_apply = 0.0
    else:
        u_apply = float(U.value[0, 0])

    # aplicar control en la planta NN: x_{k+1} = x_k + f_nn(x_k, u_apply)
    with torch.no_grad():
        x_t = torch.tensor(xk.reshape(1, -1), dtype=torch.float32)
        u_t = torch.tensor(np.array([[u_apply]], dtype=np.float32))
        dx = model(x_t, u_t).numpy().squeeze()
    xk = xk + dx

    # guardar
    X_nn[:, k+1] = xk
    U_hist[k] = u_apply
    u_prev = u_apply

# Graficar resultados
t = np.arange(0, sim_time + dt/2, dt)
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(t, X_nn[0, :], label='p (NN planta)')
plt.axhline(xref[0], color='k', linestyle='--',
            linewidth=0.7, label='setpoint')
plt.ylabel('p (m)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t[:-1], U_hist, label='u (alpha)')
plt.ylabel('alpha (rad)')
plt.xlabel('tiempo (s)')
plt.legend()
plt.grid(True)

plt.suptitle('MPC linealizado sobre NN - Ball and Beam')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
