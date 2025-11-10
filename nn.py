import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import numpy as np

# ----------------------
# Modelo lineal continuo
# ----------------------
m = 0.04
R = 0.05
Jb = 2/5 * m * R**2

g = 9.81
a = (m * g)/(m+(Jb/(R**2)))

# Discretización
dt = 0.02
A = np.array([[1, dt],
              [0, 1]])
B = np.array([[0],
              [a*dt]])

# ----------------------
# Generar dataset
# ----------------------


def simulate_linear(n_samples=5000):
    X, U, Y = [], [], []
    for _ in range(n_samples):
        # muestreo de estados y entradas
        x = np.array([
            np.random.uniform(-0.4, 0.4),   # p
            np.random.uniform(-1.0, 1.0)    # p_dot
        ])
        u = np.random.uniform(-0.25, 0.25)  # theta (rad)

        # dinámica discreta
        x_next = A @ x + (B.flatten() * u)

        X.append(x)
        U.append([u])
        Y.append(x_next)
    return np.array(X), np.array(U), np.array(Y)


X, U, Y = simulate_linear()

# Convertir a tensores
X_t = torch.tensor(X, dtype=torch.float32)
U_t = torch.tensor(U, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

# Dataset target como Δx (opcional mejora)
dY_t = Y_t - X_t  # deltas

# ----------------------
# Red neuronal
# ----------------------


class DynNN(nn.Module):
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


model = DynNN()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ----------------------
# Entrenamiento
# ----------------------
epochs = 5000
for e in range(epochs):
    optim.zero_grad()
    pred = model(X_t, U_t)
    loss = loss_fn(pred, dY_t)
    loss.backward()
    optim.step()
    if (e+1) % 100 == 0:
        print(f"Epoch {e+1}, Loss: {loss.item()}")

# ----------------------
# Prueba
# ----------------------
with torch.no_grad():
    idx = np.random.randint(0, len(X))
    x0 = X_t[idx:idx+1]
    u0 = U_t[idx:idx+1]
    true_next = Y_t[idx]
    nn_next = x0 + model(x0, u0)
    print("\nEjemplo:")
    print("Estado actual:", x0.numpy())
    print("Entrada:", u0.numpy())
    print("Real x_next:", true_next.numpy())
    print("NN x_next:", nn_next.numpy())


with torch.no_grad():
    pred_d = model(X_t, U_t)           # predicción Δx
    pred = X_t + pred_d                # x_pred = x + Δx_pred
    true = Y_t                         # x_next real

pred_np = pred.numpy()
true_np = true.numpy()

mse = mean_squared_error(true_np, pred_np)
mae = mean_absolute_error(true_np, pred_np)
r2 = r2_score(true_np, pred_np)

print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"R2:  {r2:.4f}")


# Elegimos un subconjunto para visualizar (por ejemplo, 150 muestras)
idx = np.random.choice(len(X), 150, replace=False)

true_plot = true_np[idx]
pred_plot = pred_np[idx]
t = np.arange(len(idx))

# -------------------------
# Simulación temporal NN vs modelo real
# -------------------------
steps = 200
x_real = X_t[0].numpy()       # estado inicial
x_nn = X_t[0].numpy()
u_const = 0.1                 # entrada constante de prueba

traj_real = []
traj_nn = []

with torch.no_grad():
    for k in range(steps):
        # guardar
        traj_real.append(x_real.copy())
        traj_nn.append(x_nn.copy())

        # sistema real lineal
        x_real = A @ x_real + B.flatten()*u_const

        # sistema NN
        x_tensor = torch.tensor(x_nn, dtype=torch.float32).unsqueeze(0)
        u_tensor = torch.tensor([[u_const]], dtype=torch.float32)
        dx = model(x_tensor, u_tensor).numpy().squeeze()
        x_nn = x_nn + dx

traj_real = np.array(traj_real)
traj_nn = np.array(traj_nn)
t = np.arange(steps) * dt

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(t, traj_real[:, 0], label='Real p')
plt.plot(t, traj_nn[:, 0], '--', label='NN p')
plt.xlabel('tiempo (s)')
plt.ylabel('p (m)')
plt.title('Posición vs tiempo')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, traj_real[:, 1], label='Real p_dot')
plt.plot(t, traj_nn[:, 1], '--', label='NN p_dot')
plt.xlabel('tiempo (s)')
plt.ylabel('p_dot (m/s)')
plt.title('Velocidad vs tiempo')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


torch.save(model.state_dict(), "ball_beam_model.pth")
