# ball_beam_nn_mpc_model_fixed.py
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import time
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Parámetros físicos
m = 0.04
R = 0.05
Jb = 2/5 * m * R**2
J = 0.1
g = 9.81

# Simulación
dt = 0.01
x0 = np.array([0.05, 0.0, 0.0, 0.0])  # [p, p_dot, theta, theta_dot]

process_noise_std = 1e-5
measurement_noise_std = 1e-5


def dynamics(x, u):
    p, p_dot, th, th_dot = x
    denom_p = (m + Jb / R**2)
    # Nota: sin restricciones físicas, p puede crecer infinitamente en simulación
    p_ddot = (m * p * (th_dot**2) - m * g * math.sin(th)) / denom_p
    denom_th = (J + m * p**2)
    th_ddot = (u - 2*m*p*p_dot*th_dot - m*g*p*math.cos(th)) / denom_th
    return np.array([p_dot, p_ddot, th_dot, th_ddot], dtype=np.float64)


def rk4_step(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*dt*k1, u)
    k3 = dynamics(x + 0.5*dt*k2, u)
    k4 = dynamics(x + dt*k3, u)
    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


def generate_torque_sequence(total_steps, segment_len=100, tau_limit=0.06):
    seq = np.zeros(total_steps, dtype=np.float64)
    i = 0
    while i < total_steps:
        val = np.random.uniform(-tau_limit, tau_limit)
        l = min(segment_len, total_steps - i)
        seq[i:i+l] = val
        i += l
    return seq


def generate_dataset(n_trajectories=200, steps_per_traj=500):
    X = []
    Y_delta = []  # CAMBIO: Usaremos Y para guardar deltas
    for traj in range(n_trajectories):
        x = x0.copy() + np.random.normal(scale=1e-4, size=4)
        taus = generate_torque_sequence(
            steps_per_traj, segment_len=50, tau_limit=0.1)
        # Aumenté un poco el tau_limit y variabilidad para cubrir más espacio de estados

        for k in range(steps_per_traj):
            u = taus[k]
            x_next = rk4_step(x, u, dt)
            x_next_noisy = x_next + \
                np.random.normal(scale=process_noise_std, size=4)

            # Guardamos estado actual + control como entrada
            X.append(np.concatenate([x, [u]]))
            # CAMBIO CLAVE: Guardamos la DIFERENCIA (delta) como objetivo
            Y_delta.append(x_next_noisy - x)

            x = x_next_noisy

    return np.array(X, dtype=np.float32), np.array(Y_delta, dtype=np.float32)


# ---------------------------
# Generación de datos
print("Generando datos (con deltas)...")
X, Y = generate_dataset(
    n_trajectories=200, steps_per_traj=500)  # ~100k muestras

# Normalización
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-9
Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-9

Xn = (X - X_mean) / X_std
Yn = (Y - Y_mean) / Y_std

# Datasets PyTorch
X_tensor = torch.from_numpy(Xn)
Y_tensor = torch.from_numpy(Yn)

dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(
    dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)

# ---------------------------
# Modelo


class MLPModel(nn.Module):
    def __init__(self, input_dim=5, output_dim=4, hidden=[16, 16, 16]):
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
# LR un poco más alto para empezar
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ---------------------------
# Entrenamiento
n_epochs = 50  # Suele ser suficiente con buen LR
print(f"Entrenando en {device}...")
for epoch in range(1, n_epochs+1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= val_size

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e}")

torch.save(model.state_dict(), "best_ball_beam_model_delta.pt")
with open("norms_delta.pkl", "wb") as f:
    pickle.dump({"X_mean": X_mean, "X_std": X_std,
                "Y_mean": Y_mean, "Y_std": Y_std}, f)


# ---------------------------
# Predicción y Rollout (Adaptados a Delta)


def model_predict_step(x, u):
    inp = np.concatenate([x, [u]]).astype(np.float32)
    inp_n = (inp - X_mean) / X_std
    with torch.no_grad():
        t_inp = torch.from_numpy(inp_n).unsqueeze(0).to(device)
        delta_n = model(t_inp).cpu().numpy().squeeze(0)

    # Desnormalizar delta
    delta = delta_n * Y_std + Y_mean
    # CAMBIO CLAVE: El estado siguiente es estado actual + delta predicho
    return x + delta


def rollout_model(x_init, tau_sequence, steps):
    traj = np.zeros((steps+1, 4), dtype=np.float64)
    traj[0] = x_init
    x = x_init.copy()
    for k in range(steps):
        x = model_predict_step(x, tau_sequence[k])
        traj[k+1] = x
    return traj


# ---------------------------
# Visualización
# Reduje steps_vis a 1000 (10s) para que sea más legible y realista
# antes de que el sistema inestable diverja caóticamente.
steps_vis = 500
taus_vis = generate_torque_sequence(steps_vis, segment_len=100, tau_limit=0.05)

x_real = x0.copy()
traj_real = np.zeros((steps_vis+1, 4))
traj_real[0] = x_real
for k in range(steps_vis):
    x_real = rk4_step(x_real, taus_vis[k], dt)
    traj_real[k+1] = x_real

traj_pred = rollout_model(x0, taus_vis, steps_vis)
err_norm = np.linalg.norm(traj_real - traj_pred, axis=1)

t_axis = np.arange(steps_vis+1) * dt

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.title("Comparación dinámica real vs modelo NN (Delta Prediction)")
plt.plot(t_axis, traj_real[:, 0], label="p real")
plt.plot(t_axis, traj_pred[:, 0], '--', label="p pred NN")
plt.ylabel("Posición p (m)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_axis, traj_real[:, 2], label="θ real")
plt.plot(t_axis, traj_pred[:, 2], '--', label="θ pred NN")
plt.ylabel("Ángulo θ (rad)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_axis, err_norm, 'r', label="|| error ||")
plt.xlabel("Tiempo (s)")
plt.ylabel("Error norma L2")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# --- Cálculo de Métricas de Error (Rollout) ---
# Comparamos la trayectoria real completa con la trayectoria predicha

# MAE (Error Absoluto Medio)
mae = mean_absolute_error(traj_real, traj_pred)

# MSE (Error Cuadrático Medio)
mse = mean_squared_error(traj_real, traj_pred)

# RMSE (Raíz del Error Cuadrático Medio)
rmse = np.sqrt(mse)
# Alternativa: rmse = mean_squared_error(traj_real, traj_pred, squared=False)

print("\n--- MÉTRICAS DE ERROR (ROLLOUT) ---")
print(f"Pasos de simulación: {steps_vis}")
print(f"  MAE (global):  {mae:.6e}")
print(f"  MSE (global):  {mse:.6e}")
print(f"  RMSE (global): {rmse:.6e}")
print("-------------------------------------\n")
