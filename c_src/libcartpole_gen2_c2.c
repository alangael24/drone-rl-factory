/**
 * Universal Simulation Engine - Generated for: CartPole
 *
 * GENERADO AUTOMÁTICAMENTE - NO EDITAR MANUALMENTE
 *
 * Dominio: Carro con péndulo invertido. El objetivo es balancear el palo vertical.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// CONSTANTES
// ============================================================

#define DT 0.02f
#define DEFAULT_GRAVITY 9.81f
#define WORLD_SIZE_X 2.4f
#define WORLD_SIZE_Y 0.0f
#define WORLD_SIZE_Z 0.0f
#define MAX_EPISODE_STEPS 500
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define POLE_LENGTH 0.5f
#define FORCE_MAX 10.0f

// ============================================================
// ESTRUCTURAS DE DATOS
// ============================================================

// Estado del sistema
typedef struct {
    // Posición del carro
    float cart_position;
    // Velocidad del carro
    float cart_velocity;
    // Ángulo del palo (radianes)
    float pole_angle;
    // Velocidad angular del palo
    float pole_velocity;

    // === Métricas estándar ===
    int steps;
    float total_reward;

    // === Domain Randomization ===
    float cart_mass;  // Masa del carro
    float pole_mass;  // Masa del palo
    float pole_length;  // Longitud del palo
} CartPoleState;

// Configuración de Domain Randomization
typedef struct {
    float cart_mass_min;
    float cart_mass_max;
    float pole_mass_min;
    float pole_mass_max;
    float pole_length_min;
    float pole_length_max;
    int enabled;
} DomainRandomConfig;

// Contenedor de entornos paralelos
typedef struct {
    int num_envs;
    CartPoleState* states;
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

void apply_domain_randomization(CartPoleState* state, DomainRandomConfig* config) {
    if (!config->enabled) {
        state->cart_mass = 1.0f;
        state->pole_mass = 0.1f;
        state->pole_length = 0.5f;
        return;
    }

    state->cart_mass = rand_range(config->cart_mass_min, config->cart_mass_max);
    state->pole_mass = rand_range(config->pole_mass_min, config->pole_mass_max);
    state->pole_length = rand_range(config->pole_length_min, config->pole_length_max);
}

void set_domain_randomization(UniversalEnvs* envs, float cart_mass_min, float cart_mass_max, float pole_mass_min, float pole_mass_max, float pole_length_min, float pole_length_max, int enabled) {
    envs->dr_config.cart_mass_min = cart_mass_min;
    envs->dr_config.cart_mass_max = cart_mass_max;
    envs->dr_config.pole_mass_min = pole_mass_min;
    envs->dr_config.pole_mass_max = pole_mass_max;
    envs->dr_config.pole_length_min = pole_length_min;
    envs->dr_config.pole_length_max = pole_length_max;
    envs->dr_config.enabled = enabled;
}

// ============================================================
// FUNCIÓN DE FÍSICA (GENERADA O PROPORCIONADA)
// ============================================================

void physics_step(CartPoleState* state, float* actions) {
    // Parámetros físicos
    float force = actions[0] * 10.0f;  // Convertir de [-1,1] a [-10,10]

    float cart_mass = state->cart_mass > 0.0f ? state->cart_mass : 1.0f;
    float pole_mass = state->pole_mass > 0.0f ? state->pole_mass : 0.1f;
    float pole_length = state->pole_length > 0.0f ? state->pole_length : 0.5f;

    float total_mass = cart_mass + pole_mass;
    float pole_half = pole_length * 0.5f;

    float sin_theta = sinf(state->pole_angle);
    float cos_theta = cosf(state->pole_angle);

    // Ecuaciones del péndulo invertido
    float temp = (force + pole_mass * pole_half * state->pole_velocity * state->pole_velocity * sin_theta) / total_mass;

    float theta_acc = (DEFAULT_GRAVITY * sin_theta - cos_theta * temp) /
                      (pole_half * (4.0f/3.0f - pole_mass * cos_theta * cos_theta / total_mass));

    float x_acc = temp - pole_mass * pole_half * theta_acc * cos_theta / total_mass;

    // Integración de Euler
    state->cart_velocity += x_acc * DT;
    state->cart_position += state->cart_velocity * DT;

    state->pole_velocity += theta_acc * DT;
    state->pole_angle += state->pole_velocity * DT;

    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================

float calculate_reward(CartPoleState* state) {
    // Recompensa por mantener el palo vertical
    float angle_penalty = fabsf(state->pole_angle);

    // Recompensa por mantener el carro centrado
    float position_penalty = fabsf(state->cart_position) * 0.1f;

    // Recompensa base por sobrevivir
    float reward = 1.0f;

    // Penalizaciones
    reward -= angle_penalty * 2.0f;
    reward -= position_penalty;

    // Penalización por velocidades altas
    reward -= fabsf(state->pole_velocity) * 0.1f;
    reward -= fabsf(state->cart_velocity) * 0.05f;

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
    envs->states = (CartPoleState*)calloc(num_envs, sizeof(CartPoleState));
    envs->observations = (float*)calloc(num_envs * 4, sizeof(float));
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

        envs->dr_config.cart_mass_min = 0.8f;
    envs->dr_config.cart_mass_max = 1.2f;
    envs->dr_config.pole_mass_min = 0.08f;
    envs->dr_config.pole_mass_max = 0.12f;
    envs->dr_config.pole_length_min = 0.4f;
    envs->dr_config.pole_length_max = 0.6f;
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
    CartPoleState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(CartPoleState));

    // Inicialización específica del dominio
        // Reset por defecto

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
        CartPoleState* state = &envs->states[i];

        // Ejecutar física
        physics_step(state, &actions[i * 1]);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
                int done = 0;
        // Palo cayó (>24°)
        if (fabsf(state->pole_angle) > 0.418f) done = 1;
        // Carro fuera de límites
        if (fabsf(state->cart_position) > 2.4f) done = 1;
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
        CartPoleState* state = &envs->states[i];
        float* obs = &envs->observations[i * 4];

        obs[0] = state->cart_position / 2.4f;
        obs[1] = state->cart_velocity / 5.0f;
        obs[2] = state->pole_angle / 0.418f;
        obs[3] = state->pole_velocity / 5.0f;
    }
}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(UniversalEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(UniversalEnvs* envs) { return envs->dones; }
int get_num_envs(UniversalEnvs* envs) { return envs->num_envs; }
int get_obs_size() { return 4; }
int get_action_size() { return 1; }
