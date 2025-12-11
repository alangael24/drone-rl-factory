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
    const float gravity = DEFAULT_GRAVITY;
    const float dt = DT;
    const float cart_pos = state->cart_position;
    const float cart_vel = state->cart_velocity;
    const float pole_angle = state->pole_angle;
    const float pole_vel = state->pole_velocity;
    const float cart_mass = state->cart_mass;
    const float pole_mass = state->pole_mass;
    const float pole_length_full = state->pole_length;
    const float pole_length_com = pole_length_full * 0.5f;
    const float force = actions[0];
    const float sin_angle = sinf(pole_angle);
    const float cos_angle = cosf(pole_angle);
    const float total_mass = cart_mass + pole_mass;
    const float pole_mass_length_com = pole_mass * pole_length_com;
    const float temp_numerator = force + pole_mass_length_com * pole_vel * pole_vel * sin_angle;
    const float temp = temp_numerator / total_mass;
    const float angle_accel_numerator = gravity * sin_angle - cos_angle * temp;
    const float angle_accel_denominator = pole_length_com * (4.0f/3.0f - pole_mass * cos_angle * cos_angle / total_mass);
    const float pole_angle_accel = angle_accel_numerator / angle_accel_denominator;
    const float cart_accel = temp - pole_mass_length_com * pole_angle_accel * cos_angle / total_mass;
    state->cart_position += dt * cart_vel;
    state->cart_velocity += dt * cart_accel;
    state->pole_angle += dt * pole_vel;
    state->pole_velocity += dt * pole_angle_accel;
    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================

float calculate_reward(CartPoleState* state) {
    float reward = 0.0f;
    const float angle_threshold = 0.418f;
    const float position_threshold = WORLD_SIZE_X;
    if (fabsf(state->pole_angle) > angle_threshold || fabsf(state->cart_position) > position_threshold) {
        reward = -10.0f;
    } else {
        reward = 1.0f;
        float angle_deviation_norm = fabsf(state->pole_angle) / angle_threshold;
        reward -= 2.0f * angle_deviation_norm;
        float position_deviation_norm = fabsf(state->cart_position) / position_threshold;
        reward -= 1.0f * position_deviation_norm;
        const float max_cart_vel = 5.0f;
        const float max_pole_vel = 5.0f;
        float cart_vel_norm = fabsf(state->cart_velocity) / max_cart_vel;
        float pole_vel_norm = fabsf(state->pole_velocity) / max_pole_vel;
        reward -= 0.5f * cart_vel_norm;
        reward -= 0.5f * pole_vel_norm;
    }
    return reward;
}

// ============================================================
// FUNCIÓN DE VERIFICACIÓN SEMÁNTICA (GENERADA POR LLM)
// ============================================================

int verify_domain_physics(CartPoleState* state) {
    // Verificación genérica: aceptar siempre
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

/**
 * Verifica si un estado es semánticamente válido
 */
int verify_state(CartPoleState* state) {
    if (!state) return 0;
    return verify_domain_physics(state);
}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(UniversalEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(UniversalEnvs* envs) { return envs->dones; }
int get_num_envs(UniversalEnvs* envs) { return envs->num_envs; }
int get_obs_size() { return 4; }
int get_action_size() { return 1; }
