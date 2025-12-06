/**
 * Sprint 3: Drone Dynamics - High-Speed Physics Kernel in C
 *
 * Física simplificada de quadcopter para entrenamiento RL ultrarrápido.
 * Diseñado para ser llamado millones de veces por segundo.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Constantes físicas
#define GRAVITY 9.81f
#define MASS 1.0f
#define DT 0.02f  // 50 Hz simulation
#define MAX_THRUST 20.0f
#define MAX_TORQUE 5.0f

// Límites del mundo
#define WORLD_SIZE 10.0f
#define MAX_HEIGHT 5.0f
#define MIN_HEIGHT 0.0f

// Estado del dron (44 bytes)
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
} DroneState;

// Configuración del entorno
typedef struct {
    int num_envs;           // Número de entornos paralelos
    DroneState* states;     // Array de estados
    float* observations;    // Buffer de observaciones (num_envs * obs_size)
    float* rewards;         // Buffer de recompensas
    int* dones;             // Buffer de terminaciones
} DroneEnvs;

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
 */
void physics_step(DroneState* state, float thrust, float roll_cmd, float pitch_cmd, float yaw_cmd) {
    // Limitar comandos
    thrust = clamp(thrust, 0.0f, MAX_THRUST);
    roll_cmd = clamp(roll_cmd, -MAX_TORQUE, MAX_TORQUE);
    pitch_cmd = clamp(pitch_cmd, -MAX_TORQUE, MAX_TORQUE);
    yaw_cmd = clamp(yaw_cmd, -MAX_TORQUE, MAX_TORQUE);

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

    // Aceleración = Fuerza/Masa - Gravedad
    float ax = thrust_world_x / MASS;
    float ay = thrust_world_y / MASS;
    float az = (thrust_world_z / MASS) - GRAVITY;

    // Damping aerodinámico
    float air_damping = 0.1f;
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

    // Inicializar estados
    for (int i = 0; i < num_envs; i++) {
        envs->states[i].target_z = 1.0f;  // Objetivo: hovering a 1m
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
 */
void reset_env(DroneEnvs* envs, int env_idx) {
    DroneState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(DroneState));

    // Posición inicial aleatoria (cerca del centro)
    state->x = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    state->y = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    state->z = 0.1f;  // Justo encima del suelo

    // Objetivo
    state->target_x = 0.0f;
    state->target_y = 0.0f;
    state->target_z = 1.0f;
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
