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


# ============================================================
# GENERACIÓN DE TAREAS (RoboGen-style)
# ============================================================

TASK_PROPOSAL_PROMPT = """Eres un arquitecto de simulaciones de drones con experiencia en reinforcement learning.

CAPACIDADES DEL DRON SIMULADO:
- Vuelo 3D en un espacio de 20x20x5 metros
- Control de empuje (thrust): 0-20 N
- Control de torque (roll, pitch, yaw): -5 a +5 Nm
- Sensores: posición, velocidad, orientación, velocidad angular
- Física simplificada pero realista (gravedad, arrastre, colisiones)

TAREAS EXISTENTES:
1. HOVER: Mantener posición estática en el aire
2. WAYPOINT: Volar hacia un punto objetivo
3. SMOOTH: Vuelo suave con mínima oscilación

INSTRUCCIONES:
Propón 3 tareas NUEVAS y DESAFIANTES que sean:
1. Compatibles con las capacidades físicas del dron
2. Progresivamente más difíciles
3. Diferentes a las existentes
4. Entrenables con reinforcement learning

Para cada tarea proporciona:
- Nombre corto (1-2 palabras)
- Descripción técnica (2-3 oraciones)
- Criterio de éxito medible
- Componentes clave de la función de recompensa

Formato de respuesta:
TAREA 1: [nombre]
Descripción: [descripción técnica]
Éxito: [criterio medible]
Recompensa: [componentes clave]

TAREA 2: ...
"""

# Tareas adicionales propuestas por LLM (se pueden agregar dinámicamente)
TASK_LAND = """Escribe una función de recompensa en C para aterrizaje suave:
1. El dron debe descender desde su posición actual hasta z=0
2. La velocidad vertical al tocar el suelo debe ser menor a 0.5 m/s
3. Debe mantener posición horizontal estable durante el descenso

Premiar:
- Reducción gradual de altura
- Baja velocidad vertical cerca del suelo
- Estabilidad horizontal

Penalizar:
- Velocidad de impacto alta
- Deriva horizontal excesiva
- Inclinación durante el descenso

Responde SOLO con el código C de la función, sin explicaciones."""

TASK_FIGURE8 = """Escribe una función de recompensa en C para vuelo en forma de 8:
1. El dron debe seguir una trayectoria en forma de 8 en el plano XY
2. Mantener altura constante (target_z)
3. Completar el patrón de forma suave

El patrón de 8 se puede aproximar con:
- Objetivo X = sin(t) donde t incrementa con steps
- Objetivo Y = sin(2*t)

Premiar:
- Seguimiento de la trayectoria
- Velocidad constante a lo largo del camino
- Suavidad en las curvas

Penalizar:
- Desviación de la trayectoria ideal
- Oscilaciones en altura
- Movimientos bruscos

Responde SOLO con el código C de la función, sin explicaciones."""

TASK_AVOID = """Escribe una función de recompensa en C para evasión de obstáculos:
1. El dron debe ir hacia el objetivo (target_x, target_y, target_z)
2. Pero debe evitar pasar por zonas prohibidas (centro del mapa)
3. La zona prohibida es un cilindro de radio 2m centrado en (0, 0)

Premiar:
- Acercarse al objetivo
- Mantenerse fuera de la zona prohibida
- Ruta eficiente (no dar vueltas innecesarias)

Penalizar:
- Entrar en la zona prohibida (distancia al centro < 2m)
- Quedarse estático sin avanzar
- Tiempo excesivo para llegar al objetivo

Responde SOLO con el código C de la función, sin explicaciones."""


# Diccionario de todas las tareas disponibles
AVAILABLE_TASKS = {
    "hover": TASK_HOVER,
    "waypoint": TASK_WAYPOINT,
    "smooth": TASK_SMOOTH,
    "land": TASK_LAND,
    "figure8": TASK_FIGURE8,
    "avoid": TASK_AVOID,
}


# ============================================================
# CONFIGURACIÓN DE DOMAIN RANDOMIZATION (DrEureka)
# ============================================================

# Configuración por defecto para Domain Randomization
DEFAULT_DR_CONFIG = {
    "mass_min": 0.8,       # Masa mínima (kg)
    "mass_max": 1.2,       # Masa máxima (kg)
    "drag_min": 0.05,      # Arrastre mínimo
    "drag_max": 0.2,       # Arrastre máximo
    "wind_max": 1.0,       # Fuerza máxima de viento (N)
    "gravity_var": 0.05,   # Variación de gravedad (±5%)
    "motor_noise_max": 0.1, # Ruido máximo en motores (±10%)
    "enabled": False,      # Desactivado por defecto
}

# Configuración agresiva para transferencia sim-to-real
AGGRESSIVE_DR_CONFIG = {
    "mass_min": 0.6,
    "mass_max": 1.5,
    "drag_min": 0.02,
    "drag_max": 0.3,
    "wind_max": 3.0,
    "gravity_var": 0.1,
    "motor_noise_max": 0.2,
    "enabled": True,
}

# Prompt para que el LLM proponga configuración de DR
DR_PROPOSAL_PROMPT = """Eres un experto en transferencia sim-to-real para drones.

Dado el siguiente escenario de despliegue:
{deployment_scenario}

Propón rangos de Domain Randomization para entrenar una política robusta.

PARÁMETROS DISPONIBLES:
- mass_min, mass_max: Rango de masa del dron (nominal: 1.0 kg)
- drag_min, drag_max: Rango de coeficiente de arrastre (nominal: 0.1)
- wind_max: Fuerza máxima de viento en cualquier dirección (N)
- gravity_var: Variación porcentual de gravedad (ej: 0.05 = ±5%)
- motor_noise_max: Ruido máximo en actuadores (ej: 0.1 = ±10%)

Responde en formato JSON:
{
    "mass_min": X.X,
    "mass_max": X.X,
    "drag_min": X.X,
    "drag_max": X.X,
    "wind_max": X.X,
    "gravity_var": X.X,
    "motor_noise_max": X.X,
    "reasoning": "Explicación breve de por qué estos rangos"
}
"""
