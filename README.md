# PINN - Ball and Beam System Control

Proyecto que implementa una red neuronal para modelar la dinámica de un sistema ball-beam (bola y viga) y utiliza Control Predictivo por Modelo (MPC) para su control.

## Descripción

Este proyecto consiste en:

1. **Entrenamiento de Red Neuronal (`nn.py`)**: Entrena una red neuronal feedforward para aprender la dinámica discreta del sistema ball-beam a partir de datos simulados del modelo físico linealizado.

2. **Control Predictivo por Modelo (`MPC.py`)**: Implementa un controlador MPC que utiliza la red neuronal entrenada como modelo predictivo, linearizando el modelo online en cada paso de tiempo.

## Características

- Modelo físico del sistema ball-beam con parámetros realistas
- Red neuronal feedforward (MLP) con arquitectura 3→32→32→2
- Generación de dataset sintético para entrenamiento
- Evaluación de métricas (MSE, MAE, R²)
- Visualización de trayectorias temporales
- Controlador MPC con linearización online de la red neuronal
- Optimización usando CVXPY con solver OSQP

## Requisitos

### Dependencias Python

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- CVXPY
- OpenCV (opcional, para cvxpy)

### Instalación

1. Clona o descarga el repositorio:

```bash
cd PINN
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
pip install cvxpy  # Si no está incluido en requirements.txt
```

## Estructura del Proyecto

```
PINN/
│
├── nn.py                      # Entrenamiento de la red neuronal
├── MPC.py                     # Controlador MPC usando la NN
├── ball_beam_model.pth        # Modelo entrenado guardado
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
├── Control MPC.png           # Visualización del control MPC
└── Modelo de red neuronal.png # Arquitectura de la red neuronal
```

## Uso

### 1. Entrenar la Red Neuronal

Ejecuta el script de entrenamiento:

```bash
python nn.py
```

Este script:

- Genera 5000 muestras de entrenamiento del sistema ball-beam
- Entrena la red neuronal durante 5000 épocas
- Evalúa el modelo con métricas (MSE, MAE, R²)
- Visualiza comparación de trayectorias (real vs NN)
- Guarda el modelo entrenado en `ball_beam_model.pth`

### 2. Ejecutar Control MPC

Una vez entrenado el modelo, ejecuta el controlador:

```bash
python MPC.py
```

Este script:

- Carga el modelo entrenado (`ball_beam_model.pth`)
- Simula el sistema durante 6 segundos
- Aplica control MPC con horizonte de predicción de 20 pasos
- Lineariza la red neuronal online en cada iteración
- Visualiza la respuesta del sistema y la señal de control

## Parámetros del Sistema

### Modelo Físico

- Masa de la bola: `m = 0.04 kg`
- Radio de la bola: `R = 0.05 m`
- Momento de inercia: `Jb = 2/5 * m * R²`
- Gravedad: `g = 9.81 m/s²`
- Paso de tiempo: `dt = 0.02 s`

### Red Neuronal

- Arquitectura: 3 entradas → 32 neuronas (ReLU) → 32 neuronas (ReLU) → 2 salidas
- Entradas: [posición, velocidad, ángulo]
- Salidas: [Δposición, Δvelocidad]
- Optimizador: Adam (lr=1e-3)
- Función de pérdida: MSE

### Control MPC

- Horizonte de predicción: `Np = 20`
- Matriz de peso de estados: `Q = diag([200.0, 1.0])`
- Peso de control: `R = 0.1`
- Límites de control: `u ∈ [-0.25, 0.25] rad`
- Solver: OSQP

## Resultados

El modelo entrenado debería lograr:

- **MSE**: Muy bajo (orden de 1e-6 o menor)
- **MAE**: Muy bajo
- **R²**: Cercano a 1.0

El controlador MPC debería ser capaz de llevar la bola desde una posición inicial hasta el setpoint deseado con buen comportamiento transitorio.

## Visualizaciones

El proyecto genera las siguientes visualizaciones:

1. **nn.py**:

   - Comparación de posición y velocidad real vs predicha por la NN
   - Trayectorias temporales para validación

2. **MPC.py**:
   - Evolución de la posición de la bola
   - Señal de control (ángulo de la viga)
   - Comparación con el setpoint

## Notas Técnicas

- La red neuronal predice **incrementos de estado** (Δx) en lugar del estado absoluto, lo que mejora la estabilidad numérica y el aprendizaje.
- El MPC lineariza la red neuronal en cada paso usando diferenciación automática (autograd de PyTorch).
- El modelo se lineariza alrededor del punto de operación actual: `x_{k+1} ≈ A_lin * x_k + B_lin * u_k + c_lin`
- Se utiliza warm-start en el solver OSQP para mejorar la eficiencia computacional.

## Reproducibilidad

El código incluye fijación de semillas aleatorias para garantizar la reproducibilidad de los resultados:

- Semilla: 50
- Semillas fijadas para: `random`, `numpy`, `torch`

## Autor

Proyecto desarrollado para modelado y control de sistemas dinámicos usando redes neuronales.

## Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.
