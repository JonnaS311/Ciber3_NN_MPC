import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # Para dividir datos
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# Fijar semillas
# -------------------------
seed = 50
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# Modelo lineal continuo
# -------------------------
m = 0.04
R = 0.05
Jb = 2/5 * m * R**2
g = 9.81
a = (m * g)/(m+(Jb/(R**2)))

dt = 0.02
A = np.array([[1, dt], [0, 1]])
B = np.array([[0], [a*dt]])

# -------------------------
# Dataset
# -------------------------


def simulate_linear(n_samples=5000):
    X, U, Y = [], [], []
    for _ in range(n_samples):
        x = np.array([
            np.random.uniform(-0.4, 0.4),
            np.random.uniform(-1.0, 1.0)
        ])
        u = np.random.uniform(-0.25, 0.25)
        x_next = A @ x + (B.flatten() * u)
        X.append(x)
        U.append([u])
        Y.append(x_next)
    return np.array(X), np.array(U), np.array(Y)


X, U, Y = simulate_linear()

# Convertir a Tensores
X_t_all = torch.tensor(X, dtype=torch.float32)
U_t_all = torch.tensor(U, dtype=torch.float32)
Y_t_all = torch.tensor(Y, dtype=torch.float32)
dY_t_all = Y_t_all - X_t_all  # Δx (lo que el modelo aprenderá)

# ----------------------------------------------------
# AÑADIDO: División en Entrenamiento y Validación (como en MATLAB)
# ----------------------------------------------------
X_train, X_val, U_train, U_val, dY_train, dY_val = train_test_split(
    X_t_all, U_t_all, dY_t_all, test_size=0.3, random_state=seed
)
# (Usamos 30% para validación, similar al 500/1000 de tu ejemplo MG)
print(f"Datos entrenamiento: {X_train.shape[0]}")
print(f"Datos validación: {X_val.shape[0]}")


# -------------------------
# ANFIS CORREGIDO
# -------------------------
class ANFIS(nn.Module):

    def __init__(self, n_inputs=3, mfs_per_input=2):
        super().__init__()
        self.n_inputs = n_inputs
        self.mfs_per_input = mfs_per_input
        self.n_rules = mfs_per_input ** n_inputs
        self.means = nn.Parameter(torch.randn(n_inputs, mfs_per_input) * 0.1)
        self.log_sigmas = nn.Parameter(
            torch.randn(n_inputs, mfs_per_input) * 0.1)
        self.n_cons_params = n_inputs + 1

        output_dim = 2

        self.consequents = nn.Parameter(
            torch.randn(self.n_rules, self.n_cons_params, output_dim) * 0.1
        )

        combos = []

        for i in range(self.n_rules):
            idx = []
            tmp = i
            for _ in range(n_inputs):
                idx.append(tmp % mfs_per_input)
                tmp //= mfs_per_input
            combos.append(idx)

        self.register_buffer("combos", torch.tensor(combos, dtype=torch.long))

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=1)
        batch = xu.shape[0]
        device = xu.device
        means = self.means
        sigmas = F.softplus(self.log_sigmas) + 1e-6
        xu_exp = xu.unsqueeze(2)
        means_exp = means.unsqueeze(0)
        sigmas_exp = sigmas.unsqueeze(0)
        mfg = torch.exp(-0.5 * ((xu_exp - means_exp) / sigmas_exp) ** 2)
        combos = self.combos.to(device)

        # expandir mfg para todas las reglas
        mfg_exp = mfg.unsqueeze(1).expand(
            batch, self.n_rules, self.n_inputs, self.mfs_per_input)

        # expandir combos para indexar
        combos_exp = combos.unsqueeze(0).unsqueeze(-1).expand(
            batch, self.n_rules, self.n_inputs, 1
        )

        # seleccionar MF de cada regla
        selected = torch.gather(mfg_exp, 3, combos_exp).squeeze(-1)
        firing = torch.prod(selected, dim=2)

        # normalización
        w = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)
        regressor = torch.cat([torch.ones(batch, 1, device=device), xu], dim=1)
        y_r = torch.einsum("bq,rqo->bro", regressor, self.consequents)
        out = torch.sum(w.unsqueeze(-1) * y_r, dim=1)

        return out


# Instanciar
model = ANFIS(n_inputs=3, mfs_per_input=2)
optim = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()


# ---------------------------------------------
# AÑADIDO: Función para graficar MFs (para reutilizar)
# (Basada en tu código original)
# ---------------------------------------------
def plot_mfs(model_to_plot, title):
    trained_means = model_to_plot.means.detach()
    trained_sigmas = (F.softplus(model_to_plot.log_sigmas) + 1e-6).detach()

    input_names = [
        'Entrada 1: x1 (Posición)', 'Entrada 2: x2 (Velocidad)', 'Entrada 3: u (Control)']
    ranges = [
        torch.linspace(-0.5, 0.5, 100), torch.linspace(-1.2,
                                                       1.2, 100), torch.linspace(-0.3, 0.3, 100)
    ]

    def gaussian_mf(x, mean, sigma):
        return torch.exp(-0.5 * ((x - mean) / sigma) ** 2)

    fig, axs = plt.subplots(model_to_plot.n_inputs, 1, figsize=(8, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(model_to_plot.n_inputs):
        axs[i].set_title(input_names[i])
        x_vals = ranges[i]
        for j in range(model_to_plot.mfs_per_input):
            mean = trained_means[i, j]
            sigma = trained_sigmas[i, j]
            y_vals = gaussian_mf(x_vals, mean, sigma)
            axs[i].plot(x_vals.numpy(), y_vals.numpy(),
                        label=f"MF {j+1} (μ={mean:.2f}, σ={sigma:.2f})")
        axs[i].set_ylabel("Grado de Membresía")
        axs[i].set_xlabel("Valor de Entrada")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---------------------------------------------
# AÑADIDO: Graficar MFs ANTES (como figure(2) en MATLAB)
# ---------------------------------------------
print("Mostrando funciones de membresía iniciales...")
plot_mfs(model, "Funciones de Membresía (MFs) Iniciales (Antes de Entrenar)")


# -------------------------
# Entrenamiento (MODIFICADO)
# -------------------------
epochs = 3000
# AÑADIDO: Listas para guardar los errores
train_losses = []
val_losses = []  # (Checking error en MATLAB)

print("Iniciando entrenamiento...")
for e in range(epochs):
    # Fase de entrenamiento
    model.train()  # Activar modo entrenamiento
    optim.zero_grad()
    pred_train = model(X_train, U_train)
    loss_train = loss_fn(pred_train, dY_train)
    loss_train.backward()
    optim.step()
    train_losses.append(loss_train.item())

    # Fase de validación (checking)
    model.eval()  # Desactivar modo entrenamiento (para dropout, batchnorm, etc.)
    with torch.no_grad():  # No calcular gradientes en validación
        pred_val = model(X_val, U_val)
        loss_val = loss_fn(pred_val, dY_val)
        val_losses.append(loss_val.item())

    if (e+1) % 200 == 0:
        print(
            f"Epoch {e+1}, Loss Train: {loss_train.item()}, Loss Val: {loss_val.item()}")

print("Entrenamiento finalizado.")

# ---------------------------------------------
# AÑADIDO: Graficar Curvas de Error (como figure(4) en MATLAB)
# ---------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Error de Entrenamiento")
plt.plot(val_losses, label="Error de Validación (Checking)")
plt.legend()
plt.xlabel("Epochs")
# MSELoss es MSE, pero su raíz (RMSE) es común
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Curvas de Error (Entrenamiento vs. Validación)")
plt.grid(True)
plt.show()


# -------------------------
# Pruebas (usando todo el dataset)
# -------------------------
model.eval()  # Asegurarse de que el modelo esté en modo evaluación
with torch.no_grad():
    pred_d = model(X_t_all, U_t_all)
    pred = X_t_all + pred_d
    true = Y_t_all

pred_np = pred.numpy()
true_np = true.numpy()

mse = mean_squared_error(true_np, pred_np)
mae = mean_absolute_error(true_np, pred_np)
r2 = r2_score(true_np, pred_np)

print("\nResultados finales (sobre todos los datos):")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2:  {r2}")

# -------------------------
# Simulación temporal (como en tu script)
# -------------------------
steps = 200
x_real = X_t_all[0].numpy()  # Empezar desde el mismo punto
x_nn = X_t_all[0].numpy()
u_const = 0.1

traj_real = []
traj_nn = []

model.eval()
with torch.no_grad():
    for k in range(steps):
        traj_real.append(x_real.copy())
        traj_nn.append(x_nn.copy())

        # Sistema real (lineal)
        x_real = A @ x_real + B.flatten()*u_const

        # Sistema ANFIS (no lineal)
        x_tensor = torch.tensor(x_nn, dtype=torch.float32).unsqueeze(0)
        u_tensor = torch.tensor([[u_const]], dtype=torch.float32)
        dx = model(x_tensor, u_tensor).numpy().squeeze()
        x_nn = x_nn + dx  # El modelo predice dx

traj_real = np.array(traj_real)
traj_nn = np.array(traj_nn)
t = np.arange(steps) * dt

# Gráfica de simulación (como figure(5) en MATLAB, pero para tu sistema)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, traj_real[:, 0], 'b-', label="Real (Modelo Lineal)")
plt.plot(t, traj_nn[:, 0], 'r--', label="ANFIS (Modelo Neuronal)")
plt.xlabel("Tiempo (s)")
plt.ylabel("p (m)")
plt.title("Simulación: Posición")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, traj_real[:, 1], 'b-', label="Real (Modelo Lineal)")
plt.plot(t, traj_nn[:, 1], 'r--', label="ANFIS (Modelo Neuronal)")
plt.xlabel("Tiempo (s)")
plt.ylabel("p_dot (m/s)")
plt.title("Simulación: Velocidad")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# AÑADIDO: Gráfica de Errores de Simulación (como figure(6) en MATLAB)
# ---------------------------------------------
error_p = traj_real[:, 0] - traj_nn[:, 0]
error_v = traj_real[:, 1] - traj_nn[:, 1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, error_p, 'r-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Error (m)")
plt.title("Error de predicción (Posición)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, error_v, 'r-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Error (m/s)")
plt.title("Error de predicción (Velocidad)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------------
# AÑADIDO: Graficar MFs DESPUÉS (como figure(3) en MATLAB)
# ---------------------------------------------
print("Mostrando funciones de membresía finales...")
plot_mfs(model, "Funciones de Membresía (MFs) Entrenadas (Después de Entrenar)")

torch.save(model.state_dict(), "ball_beam_anfis.pth")
