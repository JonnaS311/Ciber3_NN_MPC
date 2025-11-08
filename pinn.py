import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros físicos
# -----------------------------
m = 0.04      # masa de la bola
R = 0.05     # radio de la bola
Jb = 2/5*m*R**2  # inercia de la bola (esfera sólida)
J = 0.1      # inercia de la viga
g = 9.81     # gravedad

# Torque externo (aquí lo ponemos 0)


def tau(t, A=0.2, t_step=1.0):
    # escalón que vale 0 antes de t_step y A después (use torch.where)
    return torch.where(t >= t_step, A*torch.ones_like(t), torch.zeros_like(t))
    # return torch.zeros_like(t)
# -----------------------------
# Red neuronal
# -----------------------------


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = torch.tanh

    def forward(self, t):
        out = t
        for i, layer in enumerate(self.layers[:-1]):
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        # Salidas: p(t), theta(t)
        return out


# -----------------------------
# Entrenamiento
# -----------------------------
# Red: entrada t -> salida [p, theta]
model = PINN([1, 64, 64, 64, 2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Puntos de entrenamiento en el tiempo
t_train = torch.linspace(0, 5, 200).view(-1, 1).requires_grad_(True)

# Condiciones iniciales
p0, dp0 = 0.1, 0.0
th0, dth0 = 0.05, 0.0

for epoch in range(10000):
    optimizer.zero_grad()

    # Predicciones
    out = model(t_train)
    p = out[:, 0:1]
    th = out[:, 1:2]

    # Derivadas
    dp = autograd.grad(p, t_train, torch.ones_like(p), create_graph=True)[0]
    d2p = autograd.grad(dp, t_train, torch.ones_like(dp), create_graph=True)[0]

    dth = autograd.grad(th, t_train, torch.ones_like(th), create_graph=True)[0]
    d2th = autograd.grad(
        dth, t_train, torch.ones_like(dth), create_graph=True)[0]

    # Ecuación 1: bola
    eq1 = (m + Jb/R**2)*d2p - m*p*dth**2 + m*g*torch.sin(th)

    # Ecuación 2: viga
    eq2 = (J + m*p**2)*d2th + 2*m*p*dp*dth + m*g*p*torch.cos(th) - tau(t_train)

    # Pérdida ecuaciones
    loss_phys = torch.mean(eq1**2) + torch.mean(eq2**2)

    # Condiciones iniciales en t=0
    t0 = torch.tensor([[0.0]], requires_grad=True)
    out0 = model(t0)
    p0_pred, th0_pred = out0[:, 0], out0[:, 1]

    dp0_pred = autograd.grad(
        out0[:, 0], t0, torch.ones_like(p0_pred), create_graph=True)[0]
    dth0_pred = autograd.grad(
        out0[:, 1], t0, torch.ones_like(th0_pred), create_graph=True)[0]

    loss_ic = (p0_pred - p0)**2 + (dp0_pred - dp0)**2 + \
        (th0_pred - th0)**2 + (dth0_pred - dth0)**2

    # Pérdida total
    loss = loss_phys + loss_ic
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -----------------------------
# Visualización
# -----------------------------
t_test = torch.linspace(0, 5, 200).view(-1, 1)
out = model(t_test).detach().numpy()
p_pred = out[:, 0]
th_pred = out[:, 1]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t_test, p_pred, 'b-', label="p(t) PINN")
plt.xlabel("t [s]")
plt.ylabel("Posición bola p")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_test, th_pred, 'r-', label="θ(t) PINN")
plt.xlabel("t [s]")
plt.ylabel("Ángulo viga θ")
plt.grid(True)
plt.legend()

plt.suptitle("PINN para el sistema bola–viga")
plt.show()
