/**
 * Universal Simulation Engine - Generated for: RoboticArm2D
 *
 * GENERADO AUTOMÁTICAMENTE - NO EDITAR MANUALMENTE
 *
 * Dominio: Brazo robótico planar de 2 articulaciones para tareas de alcance.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// CONSTANTES
// ============================================================

#define DT 0.02f
#define DEFAULT_GRAVITY 0.0f
#define WORLD_SIZE_X 2.0f
#define WORLD_SIZE_Y 2.0f
#define WORLD_SIZE_Z 0.0f
#define MAX_EPISODE_STEPS 200
#define LINK1_LENGTH 1.0f
#define LINK2_LENGTH 1.0f
#define MAX_TORQUE 1.0f

// ============================================================
// ESTRUCTURAS DE DATOS
// ============================================================

// Estado del sistema
typedef struct {
    // Ángulo del primer joint (rad)
    float joint1_angle;
    // Ángulo del segundo joint (rad)
    float joint2_angle;
    // Velocidad angular joint 1 (rad/s)
    float joint1_velocity;
    // Velocidad angular joint 2 (rad/s)
    float joint2_velocity;
    // Posición X del gripper
    float end_effector_x;
    // Posición Y del gripper
    float end_effector_y;
    // Objetivo X
    float target_x;
    // Objetivo Y
    float target_y;

    // === Métricas estándar ===
    int steps;
    float total_reward;
} RoboticArm2DState;

// Configuración de Domain Randomization
// No hay parámetros de Domain Randomization

// Contenedor de entornos paralelos
typedef struct {
    int num_envs;
    RoboticArm2DState* states;
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

void physics_step(RoboticArm2DState* state, float* actions) {
    // Torques aplicados
    float torque1 = actions[0];
    float torque2 = actions[1];

    // Actualizar velocidades angulares con damping
    float damping = 0.95f;
    state->joint1_velocity = state->joint1_velocity * damping + torque1 * 0.5f;
    state->joint2_velocity = state->joint2_velocity * damping + torque2 * 0.5f;

    // Integrar ángulos
    state->joint1_angle += state->joint1_velocity * DT;
    state->joint2_angle += state->joint2_velocity * DT;

    // Limitar ángulos a [-pi, pi]
    while (state->joint1_angle > 3.14159f) state->joint1_angle -= 6.28318f;
    while (state->joint1_angle < -3.14159f) state->joint1_angle += 6.28318f;
    while (state->joint2_angle > 3.14159f) state->joint2_angle -= 6.28318f;
    while (state->joint2_angle < -3.14159f) state->joint2_angle += 6.28318f;

    // Cinemática directa para calcular posición del end effector
    float L1 = 1.0f;  // LINK1_LENGTH
    float L2 = 1.0f;  // LINK2_LENGTH

    state->end_effector_x = L1 * cosf(state->joint1_angle) + L2 * cosf(state->joint1_angle + state->joint2_angle);
    state->end_effector_y = L1 * sinf(state->joint1_angle) + L2 * sinf(state->joint1_angle + state->joint2_angle);

    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================

float calculate_reward(RoboticArm2DState* state) {
    // Distancia del end effector al objetivo
    float dx = state->target_x - state->end_effector_x;
    float dy = state->target_y - state->end_effector_y;
    float distance = sqrtf(dx*dx + dy*dy);

    // Recompensa por cercanía
    float reward = 1.0f / (1.0f + distance * 2.0f);

    // Bonus por alcanzar objetivo
    if (distance < 0.1f) {
        reward += 10.0f;
    } else if (distance < 0.3f) {
        reward += 3.0f;
    }

    // Penalización por velocidad angular alta (eficiencia energética)
    float vel_penalty = (fabsf(state->joint1_velocity) + fabsf(state->joint2_velocity)) * 0.1f;
    reward -= vel_penalty;

    return reward;
}

// ============================================================
// FUNCIÓN DE VERIFICACIÓN SEMÁNTICA (GENERADA POR LLM)
// ============================================================

int verify_domain_physics(RoboticArm2DState* state) {
    // Ángulos en rango válido
    if (state->joint1_angle < -3.14159f || state->joint1_angle > 3.14159f) return 0;
    if (state->joint2_angle < -3.14159f || state->joint2_angle > 3.14159f) return 0;

    // Velocidades articulares razonables
    if (fabsf(state->joint1_velocity) > 5.0f) return 0;
    if (fabsf(state->joint2_velocity) > 5.0f) return 0;

    // Posición del end-effector dentro de límites
    if (state->end_effector_x < -2.0f || state->end_effector_x > 2.0f) return 0;
    if (state->end_effector_y < -2.0f || state->end_effector_y > 2.0f) return 0;

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
    envs->states = (RoboticArm2DState*)calloc(num_envs, sizeof(RoboticArm2DState));
    envs->observations = (float*)calloc(num_envs * 8, sizeof(float));
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
    RoboticArm2DState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(RoboticArm2DState));

    // Inicialización específica del dominio
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
        RoboticArm2DState* state = &envs->states[i];

        // Ejecutar física
        physics_step(state, &actions[i * 2]);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
                int done = 0;
        // Objetivo alcanzado
        if (sqrtf((state->end_effector_x - state->target_x)*(state->end_effector_x - state->target_x) + (state->end_effector_y - state->target_y)*(state->end_effector_y - state->target_y)) < 0.05f) done = 1;
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
        RoboticArm2DState* state = &envs->states[i];
        float* obs = &envs->observations[i * 8];

        obs[0] = state->joint1_angle / 3.14159f;
        obs[1] = state->joint2_angle / 3.14159f;
        obs[2] = state->joint1_velocity / 5.0f;
        obs[3] = state->joint2_velocity / 5.0f;
        obs[4] = state->end_effector_x / 2.0f;
        obs[5] = state->end_effector_y / 2.0f;
        obs[6] = state->target_x / 2.0f;
        obs[7] = state->target_y / 2.0f;
    }
}

/**
 * Verifica si un estado es semánticamente válido
 */
int verify_state(RoboticArm2DState* state) {
    if (!state) return 0;
    return verify_domain_physics(state);
}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(UniversalEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(UniversalEnvs* envs) { return envs->dones; }
int get_num_envs(UniversalEnvs* envs) { return envs->num_envs; }
int get_obs_size() { return 8; }
int get_action_size() { return 2; }
