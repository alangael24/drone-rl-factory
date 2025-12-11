/**
 * Universal Simulation Engine - Generated for: Drone
 *
 * GENERADO AUTOMÁTICAMENTE - NO EDITAR MANUALMENTE
 *
 * Dominio: Quadcopter volador con 4 motores. Control de empuje y torques.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// CONSTANTES
// ============================================================

#define DT 0.02f
#define DEFAULT_GRAVITY 9.81f
#define WORLD_SIZE_X 10.0f
#define WORLD_SIZE_Y 10.0f
#define WORLD_SIZE_Z 5.0f
#define MAX_EPISODE_STEPS 500
#define MAX_THRUST 20.0f
#define MAX_TORQUE 5.0f

// ============================================================
// ESTRUCTURAS DE DATOS
// ============================================================

// Estado del sistema
typedef struct {
    // Posición X (metros)
    float x;
    // Posición Y (metros)
    float y;
    // Posición Z/altura (metros)
    float z;
    // Velocidad en X (m/s)
    float vx;
    // Velocidad en Y (m/s)
    float vy;
    // Velocidad en Z (m/s)
    float vz;
    // Rotación roll (radianes)
    float roll;
    // Rotación pitch (radianes)
    float pitch;
    // Rotación yaw (radianes)
    float yaw;
    // Velocidad angular roll (rad/s)
    float roll_rate;
    // Velocidad angular pitch (rad/s)
    float pitch_rate;
    // Velocidad angular yaw (rad/s)
    float yaw_rate;
    // Objetivo X
    float target_x;
    // Objetivo Y
    float target_y;
    // Objetivo Z
    float target_z;
    // Número de colisiones
    int collisions;

    // === Métricas estándar ===
    int steps;
    float total_reward;

    // === Domain Randomization ===
    float mass;  // Masa del dron (kg)
    float drag_coeff;  // Coeficiente de arrastre
    float wind_x;  // Viento en X
    float wind_y;  // Viento en Y
    float wind_z;  // Viento en Z
    float gravity;  // Gravedad
    float motor_noise;  // Ruido de motores
} DroneState;

// Configuración de Domain Randomization
typedef struct {
    float mass_min;
    float mass_max;
    float drag_coeff_min;
    float drag_coeff_max;
    float wind_x_min;
    float wind_x_max;
    float wind_y_min;
    float wind_y_max;
    float wind_z_min;
    float wind_z_max;
    float gravity_min;
    float gravity_max;
    float motor_noise_min;
    float motor_noise_max;
    int enabled;
} DomainRandomConfig;

// Contenedor de entornos paralelos
typedef struct {
    int num_envs;
    DroneState* states;
    float* observations;
    float* rewards;
    int* dones;
    DomainRandomConfig dr_config;
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

void apply_domain_randomization(DroneState* state, DomainRandomConfig* config) {
    if (!config->enabled) {
        state->mass = 1.0f;
        state->drag_coeff = 0.1f;
        state->wind_x = 0.0f;
        state->wind_y = 0.0f;
        state->wind_z = 0.0f;
        state->gravity = 9.81f;
        state->motor_noise = 0.0f;
        return;
    }

    state->mass = rand_range(config->mass_min, config->mass_max);
    state->drag_coeff = rand_range(config->drag_coeff_min, config->drag_coeff_max);
    state->wind_x = rand_range(config->wind_x_min, config->wind_x_max);
    state->wind_y = rand_range(config->wind_y_min, config->wind_y_max);
    state->wind_z = rand_range(config->wind_z_min, config->wind_z_max);
    state->gravity = rand_range(config->gravity_min, config->gravity_max);
    state->motor_noise = rand_range(config->motor_noise_min, config->motor_noise_max);
}

void set_domain_randomization(UniversalEnvs* envs, float mass_min, float mass_max, float drag_coeff_min, float drag_coeff_max, float wind_x_min, float wind_x_max, float wind_y_min, float wind_y_max, float wind_z_min, float wind_z_max, float gravity_min, float gravity_max, float motor_noise_min, float motor_noise_max, int enabled) {
    envs->dr_config.mass_min = mass_min;
    envs->dr_config.mass_max = mass_max;
    envs->dr_config.drag_coeff_min = drag_coeff_min;
    envs->dr_config.drag_coeff_max = drag_coeff_max;
    envs->dr_config.wind_x_min = wind_x_min;
    envs->dr_config.wind_x_max = wind_x_max;
    envs->dr_config.wind_y_min = wind_y_min;
    envs->dr_config.wind_y_max = wind_y_max;
    envs->dr_config.wind_z_min = wind_z_min;
    envs->dr_config.wind_z_max = wind_z_max;
    envs->dr_config.gravity_min = gravity_min;
    envs->dr_config.gravity_max = gravity_max;
    envs->dr_config.motor_noise_min = motor_noise_min;
    envs->dr_config.motor_noise_max = motor_noise_max;
    envs->dr_config.enabled = enabled;
}

// ============================================================
// FUNCIÓN DE FÍSICA (GENERADA O PROPORCIONADA)
// ============================================================



// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================



// ============================================================
// FUNCIÓN DE VERIFICACIÓN SEMÁNTICA (GENERADA POR LLM)
// ============================================================

int verify_domain_physics(DroneState* state) {
    // No puede estar bajo tierra
    if (state->z < 0.0f) return 0;

    // Altura máxima razonable (100 metros)
    if (state->z > 100.0f) return 0;

    // Velocidades no pueden ser extremas
    if (fabsf(state->vx) > 50.0f) return 0;
    if (fabsf(state->vy) > 50.0f) return 0;
    if (fabsf(state->vz) > 50.0f) return 0;

    // Orientaciones en rango válido
    if (fabsf(state->roll) > 3.14159f) return 0;
    if (fabsf(state->pitch) > 3.14159f) return 0;

    return 1;
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
    envs->states = (DroneState*)calloc(num_envs, sizeof(DroneState));
    envs->observations = (float*)calloc(num_envs * 15, sizeof(float));
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

        envs->dr_config.mass_min = 0.8f;
    envs->dr_config.mass_max = 1.2f;
    envs->dr_config.drag_coeff_min = 0.05f;
    envs->dr_config.drag_coeff_max = 0.2f;
    envs->dr_config.wind_x_min = -1.0f;
    envs->dr_config.wind_x_max = 1.0f;
    envs->dr_config.wind_y_min = -1.0f;
    envs->dr_config.wind_y_max = 1.0f;
    envs->dr_config.wind_z_min = -0.5f;
    envs->dr_config.wind_z_max = 0.5f;
    envs->dr_config.gravity_min = 9.32f;
    envs->dr_config.gravity_max = 10.3f;
    envs->dr_config.motor_noise_min = 0.0f;
    envs->dr_config.motor_noise_max = 0.1f;
    envs->dr_config.enabled = 0;

    // Inicializar estados
    for (int i = 0; i < num_envs; i++) {
                apply_domain_randomization(&envs->states[i], &envs->dr_config);
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
    DroneState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(DroneState));

    // Inicialización específica del dominio
        state->x = (rand_uniform() - 0.5f) * 2.0f;
    state->y = (rand_uniform() - 0.5f) * 2.0f;
    state->z = 0.1f;
    state->target_x = rand_range(-2.0f, 2.0f);
    state->target_y = rand_range(-2.0f, 2.0f);
    state->target_z = 1.0f;

    // Aplicar Domain Randomization
    apply_domain_randomization(state, &envs->dr_config);
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
        DroneState* state = &envs->states[i];

        // Ejecutar física
        physics_step(state, &actions[i * 4]);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
                int done = 0;
        // Colisión con suelo
        if (state->collisions > 0 && state->steps > 10) done = 1;
        // Tiempo agotado
        if (state->steps >= MAX_EPISODE_STEPS) done = 1;
        // Fuera del área
        if (fabsf(state->x) >= WORLD_SIZE_X || fabsf(state->y) >= WORLD_SIZE_Y) done = 1;

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
        DroneState* state = &envs->states[i];
        float* obs = &envs->observations[i * 15];

        obs[0] = state->x / 10.0f;
        obs[1] = state->y / 10.0f;
        obs[2] = state->z / 5.0f;
        obs[3] = state->vx / 5.0f;
        obs[4] = state->vy / 5.0f;
        obs[5] = state->vz / 5.0f;
        obs[6] = state->roll;
        obs[7] = state->pitch;
        obs[8] = state->yaw;
        obs[9] = state->roll_rate;
        obs[10] = state->pitch_rate;
        obs[11] = state->yaw_rate;
        obs[12] = state->target_x / 10.0f;
        obs[13] = state->target_y / 10.0f;
        obs[14] = state->target_z / 5.0f;
    }
}

/**
 * Verifica si un estado es semánticamente válido
 */
int verify_state(DroneState* state) {
    if (!state) return 0;
    return verify_domain_physics(state);
}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(UniversalEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(UniversalEnvs* envs) { return envs->dones; }
int get_num_envs(UniversalEnvs* envs) { return envs->num_envs; }
int get_obs_size() { return 15; }
int get_action_size() { return 4; }
