"""
Sprint 3: Compilador Dinámico
Toma código C del LLM, lo inserta en drone_dynamics.c, y compila.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class CompileResult(Enum):
    SUCCESS = "SUCCESS"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    LINK_ERROR = "LINK_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"


@dataclass
class CompileOutput:
    result: CompileResult
    message: str
    lib_path: str = ""
    stderr: str = ""


# Template del código C completo
DRONE_DYNAMICS_TEMPLATE = '''/**
 * Sprint 3: Drone Dynamics - Auto-generated with LLM reward function
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GRAVITY 9.81f
#define MASS 1.0f
#define DT 0.02f
#define MAX_THRUST 20.0f
#define MAX_TORQUE 5.0f
#define WORLD_SIZE 10.0f
#define MAX_HEIGHT 5.0f
#define MIN_HEIGHT 0.0f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float roll, pitch, yaw;
    float roll_rate, pitch_rate, yaw_rate;
    float target_x, target_y, target_z;
    int steps;
    int collisions;
    float total_reward;
} DroneState;

typedef struct {
    int num_envs;
    DroneState* states;
    float* observations;
    float* rewards;
    int* dones;
} DroneEnvs;

static inline float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

void physics_step(DroneState* state, float thrust, float roll_cmd, float pitch_cmd, float yaw_cmd) {
    thrust = clamp(thrust, 0.0f, MAX_THRUST);
    roll_cmd = clamp(roll_cmd, -MAX_TORQUE, MAX_TORQUE);
    pitch_cmd = clamp(pitch_cmd, -MAX_TORQUE, MAX_TORQUE);
    yaw_cmd = clamp(yaw_cmd, -MAX_TORQUE, MAX_TORQUE);

    float damping = 0.95f;
    state->roll_rate = state->roll_rate * damping + roll_cmd * 0.1f;
    state->pitch_rate = state->pitch_rate * damping + pitch_cmd * 0.1f;
    state->yaw_rate = state->yaw_rate * damping + yaw_cmd * 0.1f;

    state->roll += state->roll_rate * DT;
    state->pitch += state->pitch_rate * DT;
    state->yaw += state->yaw_rate * DT;

    state->roll = clamp(state->roll, -1.0f, 1.0f);
    state->pitch = clamp(state->pitch, -1.0f, 1.0f);

    float cos_roll = cosf(state->roll);
    float sin_roll = sinf(state->roll);
    float cos_pitch = cosf(state->pitch);
    float sin_pitch = sinf(state->pitch);

    float thrust_world_x = thrust * sin_pitch;
    float thrust_world_y = -thrust * sin_roll * cos_pitch;
    float thrust_world_z = thrust * cos_roll * cos_pitch;

    float ax = thrust_world_x / MASS;
    float ay = thrust_world_y / MASS;
    float az = (thrust_world_z / MASS) - GRAVITY;

    float air_damping = 0.1f;
    ax -= state->vx * air_damping;
    ay -= state->vy * air_damping;
    az -= state->vz * air_damping;

    state->vx += ax * DT;
    state->vy += ay * DT;
    state->vz += az * DT;

    state->x += state->vx * DT;
    state->y += state->vy * DT;
    state->z += state->vz * DT;

    if (state->z < MIN_HEIGHT) {
        state->z = MIN_HEIGHT;
        state->vz = 0.0f;
        state->collisions++;
    }

    if (state->z > MAX_HEIGHT) {
        state->z = MAX_HEIGHT;
        state->vz = 0.0f;
    }

    state->x = clamp(state->x, -WORLD_SIZE, WORLD_SIZE);
    state->y = clamp(state->y, -WORLD_SIZE, WORLD_SIZE);

    state->steps++;
}

// ============================================================
// FUNCIÓN DE RECOMPENSA GENERADA POR LLM
// ============================================================

{REWARD_FUNCTION}

// ============================================================
// API PÚBLICA
// ============================================================

DroneEnvs* create_envs(int num_envs) {
    DroneEnvs* envs = (DroneEnvs*)malloc(sizeof(DroneEnvs));
    envs->num_envs = num_envs;
    envs->states = (DroneState*)calloc(num_envs, sizeof(DroneState));
    envs->observations = (float*)calloc(num_envs * 15, sizeof(float));
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

    for (int i = 0; i < num_envs; i++) {
        envs->states[i].target_z = 1.0f;
    }

    return envs;
}

void destroy_envs(DroneEnvs* envs) {
    if (envs) {
        free(envs->states);
        free(envs->observations);
        free(envs->rewards);
        free(envs->dones);
        free(envs);
    }
}

void reset_env(DroneEnvs* envs, int env_idx) {
    DroneState* state = &envs->states[env_idx];
    memset(state, 0, sizeof(DroneState));
    state->x = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    state->y = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    state->z = 0.1f;
    state->target_x = 0.0f;
    state->target_y = 0.0f;
    state->target_z = 1.0f;
}

void reset_all(DroneEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        reset_env(envs, i);
    }
}

void step_all(DroneEnvs* envs, float* actions) {
    for (int i = 0; i < envs->num_envs; i++) {
        DroneState* state = &envs->states[i];

        float thrust = actions[i * 4 + 0];
        float roll_cmd = actions[i * 4 + 1];
        float pitch_cmd = actions[i * 4 + 2];
        float yaw_cmd = actions[i * 4 + 3];

        thrust = (thrust + 1.0f) * 0.5f * MAX_THRUST;
        roll_cmd *= MAX_TORQUE;
        pitch_cmd *= MAX_TORQUE;
        yaw_cmd *= MAX_TORQUE;

        physics_step(state, thrust, roll_cmd, pitch_cmd, yaw_cmd);

        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        int done = 0;
        if (state->collisions > 0 && state->steps > 10) done = 1;
        if (state->steps >= 500) done = 1;
        if (fabsf(state->x) >= WORLD_SIZE || fabsf(state->y) >= WORLD_SIZE) done = 1;

        envs->dones[i] = done;
        if (done) reset_env(envs, i);
    }
}

void get_observations(DroneEnvs* envs) {
    for (int i = 0; i < envs->num_envs; i++) {
        DroneState* state = &envs->states[i];
        float* obs = &envs->observations[i * 15];

        obs[0] = state->x / WORLD_SIZE;
        obs[1] = state->y / WORLD_SIZE;
        obs[2] = state->z / MAX_HEIGHT;
        obs[3] = state->vx / 5.0f;
        obs[4] = state->vy / 5.0f;
        obs[5] = state->vz / 5.0f;
        obs[6] = state->roll;
        obs[7] = state->pitch;
        obs[8] = state->yaw;
        obs[9] = state->roll_rate;
        obs[10] = state->pitch_rate;
        obs[11] = state->yaw_rate;
        obs[12] = (state->target_x - state->x) / WORLD_SIZE;
        obs[13] = (state->target_y - state->y) / WORLD_SIZE;
        obs[14] = (state->target_z - state->z) / MAX_HEIGHT;
    }
}

float* get_observations_ptr(DroneEnvs* envs) { return envs->observations; }
float* get_rewards_ptr(DroneEnvs* envs) { return envs->rewards; }
int* get_dones_ptr(DroneEnvs* envs) { return envs->dones; }
int get_num_envs(DroneEnvs* envs) { return envs->num_envs; }
'''


def compile_reward_function(
    reward_code: str,
    output_path: str = None,
    keep_source: bool = False
) -> CompileOutput:
    """
    Compila una función de recompensa C en una librería compartida.

    Args:
        reward_code: Código C de la función calculate_reward
        output_path: Path de salida para libdrone.so (default: c_src/libdrone.so)
        keep_source: Si True, guarda el archivo .c generado

    Returns:
        CompileOutput con el resultado
    """
    # Path por defecto
    if output_path is None:
        output_path = str(Path(__file__).parent / "c_src" / "libdrone.so")

    # Generar código completo
    full_code = DRONE_DYNAMICS_TEMPLATE.replace("{REWARD_FUNCTION}", reward_code)

    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.c',
        delete=not keep_source
    ) as f:
        f.write(full_code)
        f.flush()
        source_path = f.name

        # Compilar
        try:
            result = subprocess.run(
                [
                    "gcc",
                    "-O3",           # Optimización máxima
                    "-fPIC",         # Position independent code
                    "-shared",       # Crear librería compartida
                    "-o", output_path,
                    source_path,
                    "-lm"            # Link math library
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                # Error de compilación
                return CompileOutput(
                    result=CompileResult.SYNTAX_ERROR,
                    message="Error de compilación",
                    stderr=result.stderr
                )

            # Verificar que se creó el archivo
            if not Path(output_path).exists():
                return CompileOutput(
                    result=CompileResult.LINK_ERROR,
                    message="El archivo .so no se creó"
                )

            return CompileOutput(
                result=CompileResult.SUCCESS,
                message="Compilación exitosa",
                lib_path=output_path
            )

        except subprocess.TimeoutExpired:
            return CompileOutput(
                result=CompileResult.SYNTAX_ERROR,
                message="Timeout de compilación"
            )
        except Exception as e:
            return CompileOutput(
                result=CompileResult.RUNTIME_ERROR,
                message=str(e)
            )


def test_compiled_library(lib_path: str) -> tuple[bool, str]:
    """
    Prueba que la librería compilada funcione correctamente.

    Returns:
        (success, message)
    """
    import ctypes
    import numpy as np

    try:
        lib = ctypes.CDLL(lib_path)

        # Configurar funciones
        lib.create_envs.argtypes = [ctypes.c_int]
        lib.create_envs.restype = ctypes.c_void_p
        lib.destroy_envs.argtypes = [ctypes.c_void_p]
        lib.reset_all.argtypes = [ctypes.c_void_p]
        lib.step_all.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.get_rewards_ptr.argtypes = [ctypes.c_void_p]
        lib.get_rewards_ptr.restype = ctypes.POINTER(ctypes.c_float)

        # Crear entornos
        num_envs = 4
        envs = lib.create_envs(num_envs)
        lib.reset_all(envs)

        # Ejecutar pasos
        actions = np.zeros((num_envs, 4), dtype=np.float32)
        actions[:, 0] = 0.5  # Thrust medio

        for _ in range(10):
            actions_ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.step_all(envs, actions_ptr)

        # Obtener recompensas
        rewards_ptr = lib.get_rewards_ptr(envs)
        rewards = np.ctypeslib.as_array(rewards_ptr, shape=(num_envs,))

        # Verificar que las recompensas sean números válidos
        if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
            lib.destroy_envs(envs)
            return False, "Recompensas contienen NaN o Inf"

        avg_reward = np.mean(rewards)
        lib.destroy_envs(envs)

        return True, f"Test exitoso. Recompensa promedio: {avg_reward:.4f}"

    except Exception as e:
        return False, f"Error en test: {str(e)}"


if __name__ == "__main__":
    from architect import generate_reward_mock

    print("=== Compilador Dinámico ===\n")

    # Generar código de recompensa
    print("1. Generando código de recompensa (mock)...")
    reward_code = generate_reward_mock("hover")
    print(reward_code[:200] + "...\n")

    # Compilar
    print("2. Compilando...")
    output = compile_reward_function(reward_code)
    print(f"   Resultado: {output.result.value}")
    print(f"   Mensaje: {output.message}")

    if output.result == CompileResult.SUCCESS:
        print(f"   Librería: {output.lib_path}")

        # Probar
        print("\n3. Probando librería compilada...")
        success, msg = test_compiled_library(output.lib_path)
        print(f"   {'OK' if success else 'FAIL'}: {msg}")
    else:
        print(f"   Errores: {output.stderr}")
