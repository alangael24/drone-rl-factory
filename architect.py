"""
Sprint 3: El Arquitecto - Generador de funciones de recompensa en C

Implementa el bucle de reflexión de Eureka:
1. Genera función de recompensa inicial
2. Recibe métricas de entrenamiento
3. Refina la función basándose en el feedback
"""

import os
import re
from typing import Optional
from config import (
    GEMINI_API_KEY,
    API_PROVIDER,
    ARCHITECT_SYSTEM_PROMPT,
    TASK_HOVER,
    TASK_WAYPOINT,
    TASK_SMOOTH,
)


# Prompt para reflexión semántica (Eureka-style mejorado)
REFLECTION_PROMPT = """Eres un experto en Reinforcement Learning analizando el rendimiento de una función de recompensa.

CÓDIGO DE RECOMPENSA ANTERIOR:
```c
{previous_code}
```

MÉTRICAS DE ENTRENAMIENTO:
- Recompensa media final: {mean_reward:.4f}
- Recompensa máxima alcanzada: {max_reward:.4f}
- Recompensa mínima: {min_reward:.4f}
- Pasos promedio por episodio: {mean_steps:.1f}
- Colisiones promedio: {mean_collisions:.2f}
- Distancia final promedio al objetivo: {final_distance:.4f}
- Velocidad promedio: {mean_speed:.4f}
- Orientación promedio (roll+pitch): {mean_orientation:.4f}
- Velocidad angular promedio: {mean_angular_velocity:.4f}
- Tasa de éxito: {success_rate:.1%}

DIAGNÓSTICO DEL COMPORTAMIENTO:
{semantic_feedback}

TAREA ORIGINAL: {task}

Basándote en el diagnóstico anterior, genera UNA NUEVA versión MEJORADA del código C.
Haz cambios QUIRÚRGICOS y ESPECÍFICOS para resolver los problemas identificados.
Responde SOLO con el código C de la función, sin explicaciones."""


def generate_semantic_reflection(metrics: dict) -> str:
    """
    Genera texto semántico que describe el comportamiento del dron basándose en métricas.

    Esto permite al LLM entender QUÉ está mal, no solo ver números.
    Basado en el paper Eureka: la reflexión semántica mejora la calidad de las ediciones.

    Args:
        metrics: Dict con métricas del entrenamiento

    Returns:
        Texto descriptivo del comportamiento observado
    """
    feedback = []

    # Análisis de supervivencia/colisiones
    mean_steps = metrics.get('mean_steps', 0)
    mean_collisions = metrics.get('mean_collisions', 0)

    if mean_steps < 10:
        feedback.append(
            "🔴 CRÍTICO: El dron colisiona casi inmediatamente al inicio del episodio. "
            "La política no ha aprendido a generar empuje suficiente para despegar. "
            "SOLUCIÓN: Simplifica la recompensa inicial, prioriza fuertemente la supervivencia "
            "y reduce todas las penalizaciones excepto la de colisión."
        )
    elif mean_steps < 50:
        feedback.append(
            "🟠 PROBLEMA: El dron sobrevive poco tiempo (< 1 segundo). "
            "Probablemente está siendo demasiado agresivo con los controles. "
            "SOLUCIÓN: Aumenta significativamente la penalización por velocidad angular alta "
            "y por inclinaciones extremas."
        )
    elif mean_steps > 400:
        feedback.append(
            "🟢 BIEN: El dron sobrevive la mayoría del episodio. "
            "Ahora podemos enfocarnos en optimizar el comportamiento objetivo."
        )

    if mean_collisions > 5:
        feedback.append(
            "🔴 COLISIONES FRECUENTES: El dron está chocando repetidamente contra el suelo. "
            "SOLUCIÓN: Aumenta dramáticamente la penalización por colisión (multiplicar por 5-10x) "
            "y añade una penalización por estar cerca del suelo (z < 0.3m)."
        )
    elif mean_collisions > 1:
        feedback.append(
            "🟠 COLISIONES OCASIONALES: El dron toca el suelo de vez en cuando. "
            "SOLUCIÓN: Añade un término de penalización proporcional a la velocidad vertical "
            "cuando está cerca del suelo."
        )

    # Análisis de estabilidad
    mean_orientation = metrics.get('mean_orientation', 0)
    mean_angular_velocity = metrics.get('mean_angular_velocity', 0)

    if mean_angular_velocity > 3.0:
        feedback.append(
            "🔴 VUELO MUY INESTABLE: El dron vibra/oscila violentamente. "
            "Las velocidades angulares son extremadamente altas. "
            "SOLUCIÓN: Añade una penalización cuadrática por velocidad angular: "
            "`-ang_vel * ang_vel * 0.5f` en lugar de lineal."
        )
    elif mean_angular_velocity > 1.5:
        feedback.append(
            "🟠 OSCILACIONES: El dron tiene movimientos oscilatorios notables. "
            "SOLUCIÓN: Aumenta la penalización por velocidad angular (roll_rate, pitch_rate) "
            "y considera añadir un bonus por estabilidad cuando ang_vel < 0.1."
        )

    if mean_orientation > 0.5:
        feedback.append(
            "🟠 INCLINACIÓN EXCESIVA: El dron vuela muy inclinado (roll/pitch altos). "
            "SOLUCIÓN: Penaliza exponencialmente las inclinaciones mayores a 0.2 radianes."
        )

    # Análisis de objetivo
    final_distance = metrics.get('final_distance', 0)
    mean_speed = metrics.get('mean_speed', 0)

    if final_distance > 2.0:
        feedback.append(
            "🔴 NO ALCANZA OBJETIVO: El dron sobrevive pero no se acerca al objetivo. "
            "SOLUCIÓN: Aumenta significativamente la recompensa por cercanía al objetivo. "
            "Considera usar una recompensa exponencial: `exp(-distance)` en lugar de `1/(1+distance)`."
        )
    elif final_distance > 0.5:
        feedback.append(
            "🟠 LEJOS DEL OBJETIVO: El dron se acerca pero no lo suficiente. "
            "SOLUCIÓN: Añade un bonus escalonado por proximidad: +2 si <0.5m, +5 si <0.2m, +10 si <0.1m."
        )
    elif final_distance < 0.2:
        feedback.append(
            "🟢 CERCA DEL OBJETIVO: El dron alcanza el objetivo consistentemente. "
            "Ahora enfócate en la suavidad del vuelo."
        )

    if mean_speed > 3.0:
        feedback.append(
            "🟠 VELOCIDAD EXCESIVA: El dron se mueve demasiado rápido. "
            "SOLUCIÓN: Penaliza velocidades lineales mayores a 2 m/s."
        )
    elif mean_speed < 0.1 and final_distance > 0.5:
        feedback.append(
            "🔴 DRON ESTÁTICO: El dron no se mueve hacia el objetivo. "
            "Puede que la penalización por movimiento sea muy alta. "
            "SOLUCIÓN: Reduce las penalizaciones por velocidad y añade un pequeño bonus "
            "por moverse hacia el objetivo (dot product de velocidad con dirección al target)."
        )

    # Análisis de recompensa general
    mean_reward = metrics.get('mean_reward', 0)
    success_rate = metrics.get('success_rate', 0)

    if mean_reward < -5.0:
        feedback.append(
            "🔴 RECOMPENSA MUY NEGATIVA: La función de recompensa es demasiado punitiva. "
            "El agente no puede aprender porque todas las acciones resultan en castigo. "
            "SOLUCIÓN: Reduce TODAS las penalizaciones a la mitad y asegúrate de que "
            "hay una recompensa base positiva por simplemente sobrevivir."
        )
    elif mean_reward < 0:
        feedback.append(
            "🟠 RECOMPENSA NEGATIVA: El balance entre premios y castigos está desbalanceado. "
            "SOLUCIÓN: Añade una recompensa base pequeña (+0.1) por cada paso de supervivencia."
        )

    if success_rate == 0:
        feedback.append(
            "🔴 TASA DE ÉXITO 0%: La política falló completamente en todos los episodios. "
            "SOLUCIÓN DRÁSTICA: Reescribe la función de recompensa desde cero con una estructura "
            "más simple. Prioriza: 1) No colisionar, 2) Ganar altura, 3) Acercarse al objetivo."
        )
    elif success_rate < 0.1:
        feedback.append(
            "🟠 TASA DE ÉXITO BAJA (<10%): La política raramente tiene éxito. "
            "SOLUCIÓN: El criterio de éxito puede ser muy estricto o la recompensa muy escasa. "
            "Aumenta los bonus por acercarse al objetivo."
        )

    # Si no hay problemas detectados
    if not feedback:
        feedback.append(
            "🟢 COMPORTAMIENTO ACEPTABLE: No se detectaron problemas críticos. "
            "Considera ajustes finos para mejorar la suavidad o eficiencia del vuelo."
        )

    return "\n\n".join(feedback)


def get_llm_client():
    """Obtiene el cliente de API."""
    if API_PROVIDER == "gemini":
        import google.generativeai as genai
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada")
        genai.configure(api_key=api_key)
        # Usar Gemini 2.5 Flash para mejor razonamiento
        return genai, "gemini-2.5-flash-preview-05-20", "gemini"
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


def evolve_reward_function(
    task: str,
    previous_code: str,
    training_metrics: dict
) -> str:
    """
    Mejora una función de recompensa existente basada en métricas de entrenamiento.
    Implementa el bucle de reflexión semántica de Eureka.

    La reflexión semántica traduce métricas numéricas en descripciones de comportamiento
    que el LLM puede entender y usar para hacer ediciones "quirúrgicas" al código.

    Args:
        task: Tipo de tarea ("hover", "waypoint", "smooth")
        previous_code: Código C de la función anterior
        training_metrics: Dict con métricas del entrenamiento:
            - mean_reward: Recompensa promedio
            - max_reward: Recompensa máxima
            - min_reward: Recompensa mínima
            - mean_steps: Pasos promedio por episodio
            - mean_collisions: Colisiones promedio
            - final_distance: Distancia final al objetivo
            - mean_speed: Velocidad promedio
            - mean_orientation: Orientación promedio (|roll| + |pitch|)
            - mean_angular_velocity: Velocidad angular promedio
            - success_rate: Tasa de éxito (0-1)

    Returns:
        Código C mejorado de la función
    """
    client, model, provider_type = get_llm_client()

    # Generar reflexión semántica (el paso clave de Eureka)
    semantic_feedback = generate_semantic_reflection(training_metrics)

    # Construir prompt de reflexión con feedback semántico
    reflection_prompt = REFLECTION_PROMPT.format(
        previous_code=previous_code,
        task=task,
        mean_reward=training_metrics.get('mean_reward', 0),
        max_reward=training_metrics.get('max_reward', 0),
        min_reward=training_metrics.get('min_reward', 0),
        mean_steps=training_metrics.get('mean_steps', 0),
        mean_collisions=training_metrics.get('mean_collisions', 0),
        final_distance=training_metrics.get('final_distance', 0),
        mean_speed=training_metrics.get('mean_speed', 0),
        mean_orientation=training_metrics.get('mean_orientation', 0),
        mean_angular_velocity=training_metrics.get('mean_angular_velocity', 0),
        success_rate=training_metrics.get('success_rate', 0),
        semantic_feedback=semantic_feedback,
    )

    # Generar código mejorado usando Gemini 2.5
    model_instance = client.GenerativeModel(model)
    response = model_instance.generate_content(reflection_prompt)

    # Extraer y validar código
    code = extract_c_code(response.text)
    valid, msg = validate_c_syntax(code)

    if not valid:
        # Si el código no es válido, intentar una vez más con feedback
        retry_prompt = f"{reflection_prompt}\n\nERROR EN INTENTO ANTERIOR: {msg}\nGenera código válido."
        response = model_instance.generate_content(retry_prompt)
        code = extract_c_code(response.text)

    return code


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
