/**
 * Universal Simulation Engine - Generated for: WarehouseRobot
 *
 * GENERADO AUTOMÁTICAMENTE - NO EDITAR MANUALMENTE
 *
 * Dominio: Robot móvil con tracción diferencial para navegación en almacén.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// CONSTANTES
// ============================================================

#define DT 0.02f
#define DEFAULT_GRAVITY 0.0f
#define WORLD_SIZE_X 20.0f
#define WORLD_SIZE_Y 20.0f
#define WORLD_SIZE_Z 0.0f
#define MAX_EPISODE_STEPS 1000
#define WHEEL_RADIUS 0.1f
#define WHEEL_BASE 0.5f
#define MAX_SPEED 2.0f

// ============================================================
// ESTRUCTURAS DE DATOS
// ============================================================

// Estado del sistema
typedef struct {
    // Posición X (metros)
    float x;
    // Posición Y (metros)
    float y;
    // Orientación (radianes)
    float theta;
    // Velocidad lineal (m/s)
    float v_linear;
    // Velocidad angular (rad/s)
    float v_angular;
    // Objetivo X
    float target_x;
    // Objetivo Y
    float target_y;
    // Distancia a obstáculo frontal
    float obstacle_front;
    // Distancia a obstáculo izquierdo
    float obstacle_left;
    // Distancia a obstáculo derecho
    float obstacle_right;

    // === Métricas estándar ===
    int steps;
    float total_reward;
} WarehouseRobotState;

// Configuración de Domain Randomization
// No hay parámetros de Domain Randomization

// Contenedor de entornos paralelos
typedef struct {
    int num_envs;
    WarehouseRobotState* states;
    float* observations;
    float* rewards;
    int* dones;
    int dr_enabled;  // Placeholder
} UniversalEnvs;

// ============================================================
// FUNCIONES AUXILIARES
// ============================================================

static inline float rand_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

static inline float rand_range(float min_val, float max_val) {
    return min_val + rand_uniform() * (max_val - min_val);
}

static inline float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// ============================================================
// DOMAIN RANDOMIZATION
// ============================================================

void apply_domain_randomization(void* state, void* config) {
    // No hay Domain Randomization configurada
}

void set_domain_randomization(UniversalEnvs* envs, int enabled) {
    envs->dr_enabled = enabled;
}

// ============================================================
// FUNCIÓN DE FÍSICA (GENERADA O PROPORCIONADA)
// ============================================================

void physics_step(WarehouseRobotState* state, float* actions) {
    // Velocidades de ruedas normalizadas a [-1, 1]
    float wheel_left = actions[0];
    float wheel_right = actions[1];

    // Parámetros del robot
    float wheel_radius = 0.1f;
    float wheel_base = 0.5f;
    float max_wheel_speed = 2.0f;

    // Convertir a velocidades reales
    float v_left = wheel_left * max_wheel_speed;
    float v_right = wheel_right * max_wheel_speed;

    // Cinemática diferencial
    state->v_linear = (v_left + v_right) * wheel_radius / 2.0f;
    state->v_angular = (v_right - v_left) * wheel_radius / wheel_base;

    // Integrar orientación
    state->theta += state->v_angular * DT;

    // Normalizar theta a [-pi, pi]
    while (state->theta > 3.14159f) state->theta -= 6.28318f;
    while (state->theta < -3.14159f) state->theta += 6.28318f;

    // Integrar posición
    state->x += state->v_linear * cosf(state->theta) * DT;
    state->y += state->v_linear * sinf(state->theta) * DT;

    // Límites del mundo
    if (state->x > WORLD_SIZE_X) state->x = WORLD_SIZE_X;
    if (state->x < -WORLD_SIZE_X) state->x = -WORLD_SIZE_X;
    if (state->y > WORLD_SIZE_Y) state->y = WORLD_SIZE_Y;
    if (state->y < -WORLD_SIZE_Y) state->y = -WORLD_SIZE_Y;

    // Simular sensores de obstáculos (simplificado)
    state->obstacle_front = 5.0f;  // Sin obstáculos por ahora
    state->obstacle_left = 5.0f;
    state->obstacle_right = 5.0f;

    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================

float calculate_reward(WarehouseRobotState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float distance = sqrtf(dx*dx + dy*dy);

    // Recompensa por cercanía
    float reward = 2.0f - distance * 0.1f;

    // Bonus por llegar
    if (distance < 0.3f) {
        reward += 10.0f;
    }

    // Premiar velocidad moderada (no muy lento, no muy rápido)
    float speed = fabsf(state->v_linear);
    if (speed > 0.3f && speed < 1.5f) {
        reward += 0.5f;
    } else if (speed > 2.0f) {
        reward -= (speed - 2.0f) * 0.5f;
    }

    // Penalizar giros bruscos
    reward -= fabsf(state->v_angular) * 0.2f;

    // Penalizar cercanía a obstáculos
    if (state->obstacle_front < 0.5f) {
        reward -= (0.5f - state->obstacle_front) * 2.0f;
    }

    return reward;
}

// ============================================================
// API PÚBLICA
// ============================================================

/**
 * Crea un conjunto de entornos paralelos
 */
UniversalEnvs* create_envs(int num_envs) {
    UniversalEnvs* envs = (UniversalEnvs*)malloc(sizeof(UniversalEnvs));
    envs->num_envs = num_envs;
    envs->states = (WarehouseRobotState*)calloc(num_envs, sizeof(WarehouseRobotState));
    envs->observations = (float*)calloc(num_envs * 10, sizeof(float));
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

    envs->dr_enabled = 0;

    // Inicializar estados
    for (int i = 0; i < num_envs; i++) {
                // Estado inicializado a cero
    }

    return envs;
}

/**
 * Libera memoria
 */
void destroy_envs(UniversalEnvs* envs) {
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
void reset_env(UniversalEnvs* envs, int env_idx) {
    WarehouseRobotState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(WarehouseRobotState));

    // Inicialización específica del dominio
        state->x = (rand_uniform() - 0.5f) * 2.0f;
    state->y = (rand_uniform() - 0.5f) * 2.0f;
    state->target_x = rand_range(-2.0f, 2.0f);
    state->target_y = rand_range(-2.0f, 2.0f);

    // Aplicar Domain Randomization
    // No DR
}

/**
 * Resetea todos los entornos
 */
void reset_all(UniversalEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        reset_env(envs, i);
    }
}

/**
 * Ejecuta un paso en todos los entornos
 */
void step_all(UniversalEnvs* envs, float* actions) {
    for (int i = 0; i < envs->num_envs; i++) {
        WarehouseRobotState* state = &envs->states[i];

        // Ejecutar física
        physics_step(state, &actions[i * 2]);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
                int done = 0;
        // Objetivo alcanzado
        if (sqrtf((state->x - state->target_x)*(state->x - state->target_x) + (state->y - state->target_y)*(state->y - state->target_y)) < 0.3f) done = 1;
        // Colisión con obstáculo
        if (state->obstacle_front < 0.1f) done = 1;
        // Tiempo agotado
        if (state->steps >= MAX_EPISODE_STEPS) done = 1;

        envs->dones[i] = done;

        // Auto-reset si terminó
        if (done) {
            reset_env(envs, i);
        }
    }
}

/**
 * Obtiene las observaciones para todos los entornos
 */
void get_observations(UniversalEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        WarehouseRobotState* state = &envs->states[i];
        float* obs = &envs->observations[i * 10];

        obs[0] = state->x / 20.0f;
        obs[1] = state->y / 20.0f;
        obs[2] = state->theta / 3.14159f;
        obs[3] = state->v_linear / 2.0f;
        obs[4] = state->v_angular / 2.0f;
        obs[5] = state->target_x / 20.0f;
        obs[6] = state->target_y / 20.0f;
        obs[7] = state->obstacle_front / 5.0f;
        obs[8] = state->obstacle_left / 5.0f;
        obs[9] = state->obstacle_right / 5.0f;
    }
}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(UniversalEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(UniversalEnvs* envs) { return envs->dones; }
int get_num_envs(UniversalEnvs* envs) { return envs->num_envs; }
int get_obs_size() { return 10; }
int get_action_size() { return 2; }
