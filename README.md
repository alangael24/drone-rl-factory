# Drone RL Factory

**Un motor de neuro-evolución para drones autónomos.**

Sistema que usa LLMs (Gemini/GPT-4) para generar funciones de recompensa en C, compilarlas dinámicamente, y entrenar políticas neuronales de control de drones en segundos.

## Arquitectura

```
┌─────────────────┐
│   LLM (Gemini)  │ genera código C
│   "Arquitecto"  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GCC COMPILER  │ compila a .so
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHYSICS IN C   │ 2M+ pasos/seg
│   (libdrone)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PPO TRAINER   │ entrena en 40s
│   (PyTorch)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NEURAL POLICY  │ cerebro del dron
│   (.pt file)    │
└─────────────────┘
```

## Rendimiento

| Métrica | Valor |
|---------|-------|
| Velocidad de simulación | 2,000,000+ pasos/seg |
| Tiempo de entrenamiento (500K pasos) | ~40 segundos |
| Tiempo de compilación C | <1 segundo |

## Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install torch numpy gymnasium matplotlib google-generativeai

# Compilar kernel C
cd c_src
gcc -O3 -fPIC -shared -o libdrone.so drone_dynamics.c -lm
cd ..
```

## Uso

### 1. Entrenar una política

```bash
# Con función de recompensa mock
python train.py hover 500000

# Con LLM (requiere API key en config.py)
python train.py waypoint 500000 --api
```

### 2. Visualizar resultados

```bash
python visualize.py --model models/drone_policy_hover.pt --episodes 5
```

### 3. Generar y compilar recompensa personalizada

```bash
python compiler.py  # Compila función de recompensa mock
```

## Estructura del Proyecto

```
sprint3_rl/
├── c_src/
│   ├── drone_dynamics.c   # Kernel de física en C
│   └── libdrone.so        # Librería compilada
├── models/
│   ├── drone_policy_hover.pt
│   └── drone_policy_waypoint.pt
├── config.py              # Configuración y prompts
├── architect.py           # Generador de recompensas C con LLM
├── compiler.py            # Compilador dinámico GCC
├── drone_env.py           # Puente Python-C (ctypes)
├── train.py               # Entrenamiento PPO
└── visualize.py           # Visualizador 3D
```

## API de Recompensas en C

El LLM genera funciones con esta firma:

```c
typedef struct {
    float x, y, z;           // Posición
    float vx, vy, vz;        // Velocidad
    float roll, pitch, yaw;  // Orientación
    float roll_rate, pitch_rate, yaw_rate;
    float target_x, target_y, target_z;
    int steps, collisions;
    float total_reward;
} DroneState;

float calculate_reward(DroneState* state) {
    // Lógica de recompensa generada por LLM
    return reward;
}
```

## Tareas Disponibles

- **hover**: Mantener altura estable a 1 metro
- **waypoint**: Volar hacia un objetivo 3D
- **smooth**: Vuelo suave con mínima oscilación

## Requisitos

- Python 3.10+
- GCC
- PyTorch
- NumPy
- Matplotlib
- API Key de Gemini (opcional, para generación con LLM)

## Licencia

MIT

## Autor

Generado con Claude Code
