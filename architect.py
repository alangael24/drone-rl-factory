"""
Sprint 3: El Arquitecto - Generador de funciones de recompensa en C
"""

import os
import re
from config import (
    GEMINI_API_KEY,
    API_PROVIDER,
    ARCHITECT_SYSTEM_PROMPT,
    TASK_HOVER,
    TASK_WAYPOINT,
    TASK_SMOOTH,
)


def get_llm_client():
    """Obtiene el cliente de API."""
    if API_PROVIDER == "gemini":
        import google.generativeai as genai
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada")
        genai.configure(api_key=api_key)
        return genai, "gemini-2.0-flash", "gemini"
    else:
        raise ValueError(f"API_PROVIDER no válido: {API_PROVIDER}")


def extract_c_code(response: str) -> str:
    """Extrae código C de la respuesta del LLM."""
    # Buscar bloques de código C
    if "```c" in response:
        code = response.split("```c")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response

    return code.strip()


def validate_c_syntax(code: str) -> tuple[bool, str]:
    """
    Validación básica de sintaxis C.
    Verifica que tenga la estructura esperada.
    """
    # Debe contener la función calculate_reward
    if "calculate_reward" not in code:
        return False, "No se encontró la función 'calculate_reward'"

    # Debe tener return
    if "return" not in code:
        return False, "La función no tiene 'return'"

    # No debe usar printf, malloc, etc.
    forbidden = ["printf", "malloc", "free", "scanf", "fopen", "fprintf"]
    for f in forbidden:
        if f in code:
            return False, f"Uso prohibido de '{f}'"

    # Debe usar state->
    if "state->" not in code:
        return False, "Debe usar 'state->' para acceder a los campos"

    return True, "Sintaxis válida"


def generate_reward_function(task: str = "hover", feedback: str = "") -> str:
    """
    Genera una función de recompensa en C usando el LLM.

    Args:
        task: Tipo de tarea ("hover", "waypoint", "smooth")
        feedback: Feedback de compilación/ejecución anterior

    Returns:
        Código C de la función
    """
    client, model, provider_type = get_llm_client()

    # Seleccionar prompt según tarea
    task_prompts = {
        "hover": TASK_HOVER,
        "waypoint": TASK_WAYPOINT,
        "smooth": TASK_SMOOTH,
    }
    user_prompt = task_prompts.get(task, TASK_HOVER)

    # Agregar feedback si existe
    if feedback:
        user_prompt += f"""

IMPORTANTE - FEEDBACK DEL INTENTO ANTERIOR:
{feedback}

Corrige los errores y genera código C válido."""

    # Generar
    full_prompt = f"{ARCHITECT_SYSTEM_PROMPT}\n\n{user_prompt}"
    model_instance = client.GenerativeModel(model)
    response = model_instance.generate_content(full_prompt)

    # Extraer código
    code = extract_c_code(response.text)

    return code


def generate_reward_mock(task: str = "hover") -> str:
    """Versión mock para pruebas sin API."""

    if task == "hover":
        return '''float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa base por cercanía
    float reward = 1.0f / (1.0f + distance);

    // Bonus por estar muy cerca
    if (distance < 0.1f) {
        reward += 5.0f;
    } else if (distance < 0.3f) {
        reward += 2.0f;
    }

    // Penalización por orientación
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 0.5f;
    reward -= orientation_penalty;

    // Penalización por velocidad angular alta
    float ang_vel = fabsf(state->roll_rate) + fabsf(state->pitch_rate) + fabsf(state->yaw_rate);
    reward -= ang_vel * 0.1f;

    // Penalización por velocidad lineal alta
    float speed = sqrtf(state->vx*state->vx + state->vy*state->vy + state->vz*state->vz);
    reward -= speed * 0.05f;

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 2.0f;
    }

    return reward;
}'''

    elif task == "waypoint":
        return '''float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa inversamente proporcional a distancia
    float reward = 2.0f / (1.0f + distance * distance);

    // Gran bonus por llegar al objetivo
    if (distance < 0.2f) {
        reward += 10.0f;
    } else if (distance < 0.5f) {
        reward += 3.0f;
    }

    // Penalización suave por orientación
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 0.3f;
    reward -= orientation_penalty;

    // Permitir algo de velocidad (queremos que se mueva)
    float speed = sqrtf(state->vx*state->vx + state->vy*state->vy + state->vz*state->vz);
    if (speed > 3.0f) {
        reward -= (speed - 3.0f) * 0.2f;  // Solo penalizar velocidad excesiva
    }

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 5.0f;
    }

    return reward;
}'''

    else:  # smooth
        return '''float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa base
    float reward = 1.0f / (1.0f + distance);

    // FUERTE penalización por orientación
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 2.0f;
    if (fabsf(state->roll) > 0.2f || fabsf(state->pitch) > 0.2f) {
        orientation_penalty += 3.0f;  // Penalización extra por inclinación excesiva
    }
    reward -= orientation_penalty;

    // FUERTE penalización por velocidad angular
    float ang_vel = fabsf(state->roll_rate) + fabsf(state->pitch_rate) + fabsf(state->yaw_rate);
    reward -= ang_vel * 0.5f;

    // Penalización por velocidad lineal alta
    float speed = sqrtf(state->vx*state->vx + state->vy*state->vy + state->vz*state->vz);
    reward -= speed * 0.2f;

    // Bonus por estabilidad perfecta
    if (fabsf(state->roll) < 0.05f && fabsf(state->pitch) < 0.05f && ang_vel < 0.1f) {
        reward += 1.0f;
    }

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 5.0f;
    }

    return reward;
}'''


if __name__ == "__main__":
    print("=== El Arquitecto de Recompensas C ===\n")

    for task in ["hover", "waypoint", "smooth"]:
        print(f"--- Tarea: {task} ---")
        code = generate_reward_mock(task)
        print(code[:300] + "...\n")

    # Probar validación
    print("--- Validación de sintaxis ---")
    code = generate_reward_mock("hover")
    valid, msg = validate_c_syntax(code)
    print(f"Resultado: {valid} - {msg}")
