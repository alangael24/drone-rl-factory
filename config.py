"""
Sprint 3: Configuración para generación de recompensas en C
"""

# API Configuration
# Set your API key here or use environment variable GEMINI_API_KEY
GEMINI_API_KEY = ""  # Add your key here
API_PROVIDER = "gemini"

# Paths
C_SRC_DIR = "c_src"
LIB_NAME = "libdrone.so"

# Training
DEFAULT_NUM_ENVS = 64
DEFAULT_TOTAL_STEPS = 1_000_000
DEFAULT_LEARNING_RATE = 3e-4

# Prompt del Arquitecto para funciones de recompensa en C
ARCHITECT_SYSTEM_PROMPT = """Eres un experto en Reinforcement Learning y programación en C.
Tu trabajo es escribir funciones de recompensa en C puro para un sistema de entrenamiento de drones.

ESTRUCTURA DE DATOS DISPONIBLE:

```c
typedef struct {
    // Posición (x, y, z) en metros
    float x, y, z;

    // Velocidad (vx, vy, vz) en m/s
    float vx, vy, vz;

    // Orientación (roll, pitch, yaw) en radianes
    float roll, pitch, yaw;

    // Velocidad angular en rad/s
    float roll_rate, pitch_rate, yaw_rate;

    // Posición objetivo
    float target_x, target_y, target_z;

    // Métricas
    int steps;
    int collisions;
    float total_reward;
} DroneState;
```

FUNCIÓN A IMPLEMENTAR:

```c
float calculate_reward(DroneState* state) {
    // Tu código aquí
    return reward;
}
```

REGLAS IMPORTANTES:
1. SOLO usa la librería math.h (fabsf, sqrtf, etc.)
2. NO uses printf, malloc, ni otras funciones de sistema
3. Retorna un valor float
4. La recompensa debe estar en un rango razonable (-10 a +10)
5. Usa state-> para acceder a los campos del struct
6. El código debe ser eficiente (se ejecutará millones de veces)

OBJETIVOS TÍPICOS DE RECOMPENSA:
- Acercarse al objetivo (target_x, target_y, target_z)
- Mantener estabilidad (minimizar roll, pitch)
- Vuelo suave (minimizar velocidades angulares)
- Evitar colisiones (penalizar collisions > 0)
"""

TASK_HOVER = """Escribe una función de recompensa en C para que el dron:
1. Se eleve y mantenga altura en target_z (típicamente 1 metro)
2. Se mantenga estable (roll y pitch cerca de 0)
3. Minimice movimiento horizontal (quedarse en target_x, target_y)

La función debe premiar:
- Estar cerca de la posición objetivo
- Mantener orientación nivelada
- Vuelo suave (bajas velocidades angulares)

Y penalizar:
- Distancia al objetivo
- Inclinación excesiva
- Colisiones con el suelo

Responde SOLO con el código C de la función, sin explicaciones."""

TASK_WAYPOINT = """Escribe una función de recompensa en C para que el dron:
1. Vuele hacia la posición objetivo (target_x, target_y, target_z)
2. Mantenga estabilidad durante el vuelo
3. Llegue lo más rápido posible pero de forma suave

La función debe:
- Dar recompensa proporcional a la cercanía al objetivo
- Bonus grande (+5 a +10) al estar muy cerca (<0.2m)
- Penalizar orientaciones extremas
- Penalizar velocidades muy altas

Responde SOLO con el código C de la función, sin explicaciones."""

TASK_SMOOTH = """Escribe una función de recompensa en C que priorice:
1. Vuelo extremadamente suave (bajas aceleraciones)
2. Orientación siempre nivelada
3. Alcanzar el objetivo gradualmente

Enfócate en penalizar fuertemente:
- Velocidades angulares altas
- Cambios bruscos de velocidad
- Inclinaciones mayores a 0.2 radianes

Y premiar:
- Movimiento lento y controlado
- Orientación estable
- Progreso hacia el objetivo

Responde SOLO con el código C de la función, sin explicaciones."""
