import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import math

# ============================================================
#   Cargar modelo entrenado (DELTA MPC)
# ============================================================


class MLPModel(nn.Module):
    def __init__(self, input_dim=5, output_dim=4, hidden=[64, 64, 64]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel().to(device)
model.load_state_dict(torch.load(
    "best_ball_beam_model_delta.pt", map_location=device))
model.eval()

# ============================================================
#   Cargar normalización guardada desde el archivo original
# ============================================================

with open("norms_delta.pkl", "rb") as f:
    data = pickle.load(f)
    X_mean = data["X_mean"]
    X_std = data["X_std"]
    Y_mean = data["Y_mean"]
    Y_std = data["Y_std"]

# ============================================================
#   Parámetros físicos y simulación real
# ============================================================

m = 0.04
R = 0.05
Jb = 2/5 * m * R**2
J = 0.1
g = 9.81

dt = 0.01
x_real = np.array([0.05, 0, 0, 0], dtype=np.float64)   # estado inicial


def dynamics(x, u):
    p, p_dot, th, th_dot = x
    denom_p = (m + Jb / R**2)
    p_ddot = (m*p*(th_dot**2) - m*g*math.sin(th)) / denom_p
    denom_th = (J + m*p**2)
    th_ddot = (u - 2*m*p*p_dot*th_dot - m*g*p*math.cos(th)) / denom_th
    return np.array([p_dot, p_ddot, th_dot, th_ddot], dtype=np.float64)


def rk4_step(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*dt*k1, u)
    k3 = dynamics(x + 0.5*dt*k2, u)
    k4 = dynamics(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# ============================================================
#   Función NN predictiva (delta model)
# ============================================================


def nn_step(x, u):
    inp = np.concatenate([x, [u]]).astype(np.float32)
    inp_n = (inp - X_mean) / X_std
    t_inp = torch.from_numpy(inp_n).unsqueeze(0).to(device)

    with torch.no_grad():
        delta_n = model(t_inp).cpu().numpy().squeeze(0)

    delta = delta_n * Y_std + Y_mean
    return x + delta

# ============================================================
#   MPC setup
# ============================================================


N = 12              # horizonte
grad_iters = 15     # iteraciones descenso gradiente por paso
tau_limit = 0.08    # límite torque físico
max_steps = 500     # duración total (~5s)
x = x_real.copy()

traj = np.zeros((max_steps+1, 4))
u_log = np.zeros(max_steps)
traj[0] = x

print("\n=========== INICIANDO MPC ===========")

for step in range(max_steps):
    # variables de control optimizables
    u_seq = torch.zeros(N, dtype=torch.float32,
                        requires_grad=True, device=device)
    opt = torch.optim.Adam([u_seq], lr=0.05)

    for it in range(grad_iters):
        x_sim = torch.tensor(x, dtype=torch.float32, device=device)
        cost = 0.0

        for k in range(N):
            inp = torch.cat([x_sim, u_seq[k].unsqueeze(0)])
            inp_n = (inp - torch.tensor(X_mean, device=device)) / \
                torch.tensor(X_std, device=device)
            delta_n = model(inp_n.unsqueeze(0))
            delta = delta_n * \
                torch.tensor(Y_std, device=device) + \
                torch.tensor(Y_mean, device=device)

            x_sim = x_sim + delta.squeeze(0)

            cost += 10.0*x_sim[0]**2 + 0.1*x_sim[2]**2 + 0.001*u_seq[k]**2

        opt.zero_grad()
        cost.backward()
        opt.step()

    u = float(torch.clamp(u_seq[0], -tau_limit, tau_limit).item())

    x = rk4_step(x, u, dt)

    traj[step+1] = x
    u_log[step] = u

    if step % 50 == 0:
        print(
            f"[t = {step*dt:4.2f}s] p={x[0]:+.3f}m | th={x[2]:+.3f}rad | u={u:+.3f}")

print("=========== MPC FINALIZADO ===========")

# ============================================================
#   Gráficas
# ============================================================

t = np.arange(max_steps+1)*dt

plt.figure(figsize=(9, 8))
plt.subplot(3, 1, 1)
plt.title("Respuesta del sistema controlado (MPC + NN)")
plt.plot(t, traj[:, 0])
plt.grid()
plt.ylabel("p (m)")
plt.subplot(3, 1, 2)
plt.plot(t, traj[:, 2])
plt.grid()
plt.ylabel("θ (rad)")
plt.subplot(3, 1, 3)
plt.plot(t[:-1], u_log)
plt.grid()
plt.ylabel("Torque (Nm)")
plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()
