/**
 * Sprint 3: Drone Dynamics - High-Speed Physics Kernel in C
 *
 * Física simplificada de quadcopter para entrenamiento RL ultrarrápido.
 * Diseñado para ser llamado millones de veces por segundo.
 *
 * Implementa Domain Randomization (DrEureka) para robustez sim-to-real.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Constantes físicas por defecto (pueden ser sobrescritas por DR)
#define DEFAULT_GRAVITY 9.81f
#define DEFAULT_MASS 1.0f
#define DT 0.02f  // 50 Hz simulation
#define MAX_THRUST 20.0f
#define MAX_TORQUE 5.0f

// Límites del mundo
#define WORLD_SIZE 10.0f
#define MAX_HEIGHT 5.0f
#define MIN_HEIGHT 0.0f

// Estado del dron con parámetros de Domain Randomization
typedef struct {
    // Posición (x, y, z)
    float x, y, z;

    // Velocidad (vx, vy, vz)
    float vx, vy, vz;

    // Orientación (roll, pitch, yaw) en radianes
    float roll, pitch, yaw;

    // Velocidad angular
    float roll_rate, pitch_rate, yaw_rate;

    // Objetivo
    float target_x, target_y, target_z;

    // Métricas
    int steps;
    int collisions;
    float total_reward;

    // === DOMAIN RANDOMIZATION (DrEureka) ===
    // Parámetros físicos aleatorizados por entorno
    float mass;           // Masa del dron (varía entre entornos)
    float drag_coeff;     // Coeficiente de arrastre aerodinámico
    float wind_x;         // Fuerza de viento en X
    float wind_y;         // Fuerza de viento en Y
    float wind_z;         // Fuerza de viento en Z
    float gravity;        // Gravedad (puede variar ligeramente)
    float motor_noise;    // Ruido en los motores (0-1)
} DroneState;

// Configuración de Domain Randomization
typedef struct {
    // Rangos para aleatorización de masa
    float mass_min;
    float mass_max;
    // Rangos para arrastre
    float drag_min;
    float drag_max;
    // Fuerza máxima de viento
    float wind_max;
    // Variación de gravedad (±%)
    float gravity_var;
    // Ruido de motor máximo
    float motor_noise_max;
    // Flag para activar/desactivar DR
    int enabled;
} DomainRandomConfig;

// Configuración del entorno
typedef struct {
    int num_envs;           // Número de entornos paralelos
    DroneState* states;     // Array de estados
    float* observations;    // Buffer de observaciones (num_envs * obs_size)
    float* rewards;         // Buffer de recompensas
    int* dones;             // Buffer de terminaciones
    DomainRandomConfig dr_config;  // Configuración de Domain Randomization
} DroneEnvs;

// ============================================================
// FUNCIONES DE DOMAIN RANDOMIZATION
// ============================================================

/**
 * Genera un número aleatorio uniforme en [0, 1]
 */
static inline float rand_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

/**
 * Genera un número aleatorio uniforme en [min, max]
 */
static inline float rand_range(float min_val, float max_val) {
    return min_val + rand_uniform() * (max_val - min_val);
}

/**
 * Aplica Domain Randomization a un estado
 */
void apply_domain_randomization(DroneState* state, DomainRandomConfig* config) {
    if (!config->enabled) {
        // Usar valores por defecto
        state->mass = DEFAULT_MASS;
        state->drag_coeff = 0.1f;
        state->wind_x = 0.0f;
        state->wind_y = 0.0f;
        state->wind_z = 0.0f;
        state->gravity = DEFAULT_GRAVITY;
        state->motor_noise = 0.0f;
        return;
    }

    // Aleatorizar masa
    state->mass = rand_range(config->mass_min, config->mass_max);

    // Aleatorizar arrastre
    state->drag_coeff = rand_range(config->drag_min, config->drag_max);

    // Aleatorizar viento (dirección y magnitud)
    state->wind_x = rand_range(-config->wind_max, config->wind_max);
    state->wind_y = rand_range(-config->wind_max, config->wind_max);
    state->wind_z = rand_range(-config->wind_max * 0.5f, config->wind_max * 0.5f);

    // Aleatorizar gravedad (pequeña variación)
    float grav_variation = 1.0f + rand_range(-config->gravity_var, config->gravity_var);
    state->gravity = DEFAULT_GRAVITY * grav_variation;

    // Aleatorizar ruido de motor
    state->motor_noise = rand_range(0.0f, config->motor_noise_max);
}

/**
 * Configura los parámetros de Domain Randomization
 */
void set_domain_randomization(
    DroneEnvs* envs,
    float mass_min, float mass_max,
    float drag_min, float drag_max,
    float wind_max,
    float gravity_var,
    float motor_noise_max,
    int enabled
) {
    envs->dr_config.mass_min = mass_min;
    envs->dr_config.mass_max = mass_max;
    envs->dr_config.drag_min = drag_min;
    envs->dr_config.drag_max = drag_max;
    envs->dr_config.wind_max = wind_max;
    envs->dr_config.gravity_var = gravity_var;
    envs->dr_config.motor_noise_max = motor_noise_max;
    envs->dr_config.enabled = enabled;
}

// ============================================================
// FUNCIONES DE FÍSICA
// ============================================================

/**
 * Limita un valor a un rango
 */
static inline float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * Actualiza la física del dron por un paso de tiempo
 * Usa parámetros de Domain Randomization del estado
 */
void physics_step(DroneState* state, float thrust, float roll_cmd, float pitch_cmd, float yaw_cmd) {
    // Limitar comandos
    thrust = clamp(thrust, 0.0f, MAX_THRUST);
    roll_cmd = clamp(roll_cmd, -MAX_TORQUE, MAX_TORQUE);
    pitch_cmd = clamp(pitch_cmd, -MAX_TORQUE, MAX_TORQUE);
    yaw_cmd = clamp(yaw_cmd, -MAX_TORQUE, MAX_TORQUE);

    // Aplicar ruido de motor (Domain Randomization)
    if (state->motor_noise > 0.0f) {
        float noise = (rand_uniform() - 0.5f) * 2.0f * state->motor_noise;
        thrust *= (1.0f + noise);
        roll_cmd *= (1.0f + noise * 0.5f);
        pitch_cmd *= (1.0f + noise * 0.5f);
    }

    // Actualizar velocidades angulares (simplificado: torque directo)
    float damping = 0.95f;
    state->roll_rate = state->roll_rate * damping + roll_cmd * 0.1f;
    state->pitch_rate = state->pitch_rate * damping + pitch_cmd * 0.1f;
    state->yaw_rate = state->yaw_rate * damping + yaw_cmd * 0.1f;

    // Integrar orientación
    state->roll += state->roll_rate * DT;
    state->pitch += state->pitch_rate * DT;
    state->yaw += state->yaw_rate * DT;

    // Limitar orientación
    state->roll = clamp(state->roll, -1.0f, 1.0f);
    state->pitch = clamp(state->pitch, -1.0f, 1.0f);

    // Calcular aceleración (simplificado)
    // El thrust se aplica en la dirección Z del dron
    float cos_roll = cosf(state->roll);
    float sin_roll = sinf(state->roll);
    float cos_pitch = cosf(state->pitch);
    float sin_pitch = sinf(state->pitch);

    // Fuerza en frame mundo
    float thrust_world_x = thrust * sin_pitch;
    float thrust_world_y = -thrust * sin_roll * cos_pitch;
    float thrust_world_z = thrust * cos_roll * cos_pitch;

    // Usar masa y gravedad del estado (Domain Randomization)
    float mass = state->mass > 0.0f ? state->mass : DEFAULT_MASS;
    float gravity = state->gravity > 0.0f ? state->gravity : DEFAULT_GRAVITY;

    // Aceleración = Fuerza/Masa - Gravedad + Viento
    float ax = (thrust_world_x + state->wind_x) / mass;
    float ay = (thrust_world_y + state->wind_y) / mass;
    float az = ((thrust_world_z + state->wind_z) / mass) - gravity;

    // Damping aerodinámico (usa coeficiente del estado)
    float air_damping = state->drag_coeff > 0.0f ? state->drag_coeff : 0.1f;
    ax -= state->vx * air_damping;
    ay -= state->vy * air_damping;
    az -= state->vz * air_damping;

    // Integrar velocidad
    state->vx += ax * DT;
    state->vy += ay * DT;
    state->vz += az * DT;

    // Integrar posición
    state->x += state->vx * DT;
    state->y += state->vy * DT;
    state->z += state->vz * DT;

    // Detectar colisión con el suelo
    if (state->z < MIN_HEIGHT) {
        state->z = MIN_HEIGHT;
        state->vz = 0.0f;
        state->collisions++;
    }

    // Limitar altura máxima
    if (state->z > MAX_HEIGHT) {
        state->z = MAX_HEIGHT;
        state->vz = 0.0f;
    }

    // Limitar posición horizontal (mundo toroidal o con paredes)
    state->x = clamp(state->x, -WORLD_SIZE, WORLD_SIZE);
    state->y = clamp(state->y, -WORLD_SIZE, WORLD_SIZE);

    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA (PLACEHOLDER - SERÁ REEMPLAZADA POR LLM)
// ============================================================

/**
 * Calcula la recompensa para el estado actual.
 * Esta función será generada por el LLM y compilada dinámicamente.
 */
float calculate_reward(DroneState* state) {
    // Recompensa por acercarse al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa inversamente proporcional a la distancia
    float reward = 1.0f / (1.0f + distance);

    // Bonus por estar muy cerca
    if (distance < 0.1f) {
        reward += 10.0f;
    }

    // Penalización por velocidad alta (queremos vuelo suave)
    float speed = sqrtf(state->vx*state->vx + state->vy*state->vy + state->vz*state->vz);
    reward -= speed * 0.01f;

    // Penalización por orientación extrema
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 0.1f;
    reward -= orientation_penalty;

    // Penalización por colisión
    if (state->z <= MIN_HEIGHT && state->vz < -0.1f) {
        reward -= 5.0f;
    }

    return reward;
}

// ============================================================
// API PÚBLICA (LLAMADA DESDE PYTHON)
// ============================================================

/**
 * Crea un conjunto de entornos paralelos
 */
DroneEnvs* create_envs(int num_envs) {
    DroneEnvs* envs = (DroneEnvs*)malloc(sizeof(DroneEnvs));
    envs->num_envs = num_envs;
    envs->states = (DroneState*)calloc(num_envs, sizeof(DroneState));
    envs->observations = (float*)calloc(num_envs * 15, sizeof(float));  // 15 observaciones
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

    // Inicializar configuración de Domain Randomization por defecto (desactivado)
    envs->dr_config.mass_min = 0.8f;
    envs->dr_config.mass_max = 1.2f;
    envs->dr_config.drag_min = 0.05f;
    envs->dr_config.drag_max = 0.2f;
    envs->dr_config.wind_max = 1.0f;
    envs->dr_config.gravity_var = 0.05f;
    envs->dr_config.motor_noise_max = 0.1f;
    envs->dr_config.enabled = 0;  // Desactivado por defecto

    // Inicializar estados
    for (int i = 0; i < num_envs; i++) {
        envs->states[i].target_z = 1.0f;  // Objetivo: hovering a 1m
        // Aplicar DR inicial
        apply_domain_randomization(&envs->states[i], &envs->dr_config);
    }

    return envs;
}

/**
 * Libera memoria
 */
void destroy_envs(DroneEnvs* envs) {
    if (envs) {
        free(envs->states);
        free(envs->observations);
        free(envs->rewards);
        free(envs->dones);
        free(envs);
    }
}

/**
 * Resetea un entorno específico
 * Aplica Domain Randomization para nuevos parámetros físicos
 */
void reset_env(DroneEnvs* envs, int env_idx) {
    DroneState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(DroneState));

    // Posición inicial aleatoria (cerca del centro)
    state->x = (rand_uniform() - 0.5f) * 2.0f;
    state->y = (rand_uniform() - 0.5f) * 2.0f;
    state->z = 0.1f;  // Justo encima del suelo

    // Objetivo
    state->target_x = 0.0f;
    state->target_y = 0.0f;
    state->target_z = 1.0f;

    // Aplicar Domain Randomization (nuevos parámetros físicos para este episodio)
    apply_domain_randomization(state, &envs->dr_config);
}

/**
 * Resetea todos los entornos
 */
void reset_all(DroneEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        reset_env(envs, i);
    }
}

/**
 * Ejecuta un paso en todos los entornos
 *
 * actions: array de (num_envs * 4) floats [thrust, roll, pitch, yaw] por env
 */
void step_all(DroneEnvs* envs, float* actions) {
    for (int i = 0; i < envs->num_envs; i++) {
        DroneState* state = &envs->states[i];

        // Extraer acciones para este entorno
        float thrust = actions[i * 4 + 0];
        float roll_cmd = actions[i * 4 + 1];
        float pitch_cmd = actions[i * 4 + 2];
        float yaw_cmd = actions[i * 4 + 3];

        // Convertir acciones normalizadas [-1, 1] a comandos reales
        thrust = (thrust + 1.0f) * 0.5f * MAX_THRUST;  // [0, MAX_THRUST]
        roll_cmd *= MAX_TORQUE;
        pitch_cmd *= MAX_TORQUE;
        yaw_cmd *= MAX_TORQUE;

        // Ejecutar física
        physics_step(state, thrust, roll_cmd, pitch_cmd, yaw_cmd);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
        int done = 0;

        // Terminación por colisión
        if (state->collisions > 0 && state->steps > 10) {
            done = 1;
        }

        // Terminación por tiempo (500 pasos = 10 segundos)
        if (state->steps >= 500) {
            done = 1;
        }

        // Terminación por salir del área
        if (fabsf(state->x) >= WORLD_SIZE || fabsf(state->y) >= WORLD_SIZE) {
            done = 1;
        }

        envs->dones[i] = done;

        // Auto-reset si terminó
        if (done) {
            reset_env(envs, i);
        }
    }
}

/**
 * Obtiene las observaciones para todos los entornos
 * Formato: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, target_x, target_y, target_z]
 */
void get_observations(DroneEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        DroneState* state = &envs->states[i];
        float* obs = &envs->observations[i * 15];

        obs[0] = state->x / WORLD_SIZE;  // Normalizado
        obs[1] = state->y / WORLD_SIZE;
        obs[2] = state->z / MAX_HEIGHT;
        obs[3] = state->vx / 5.0f;  // Normalizado por velocidad máxima típica
        obs[4] = state->vy / 5.0f;
        obs[5] = state->vz / 5.0f;
        obs[6] = state->roll;
        obs[7] = state->pitch;
        obs[8] = state->yaw;
        obs[9] = state->roll_rate;
        obs[10] = state->pitch_rate;
        obs[11] = state->yaw_rate;
        obs[12] = (state->target_x - state->x) / WORLD_SIZE;  // Error relativo
        obs[13] = (state->target_y - state->y) / WORLD_SIZE;
        obs[14] = (state->target_z - state->z) / MAX_HEIGHT;
    }
}

/**
 * Obtiene punteros a los buffers (para ctypes)
 */
float* get_observations_ptr(DroneEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(DroneEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(DroneEnvs* envs) { return envs->dones; }
int get_num_envs(DroneEnvs* envs) { return envs->num_envs; }
