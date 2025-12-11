"""
Universal Architect - Genera física Y recompensa para CUALQUIER dominio

A diferencia del architect.py original que solo generaba recompensas para drones,
este arquitecto puede:
1. Generar el struct de estado
2. Generar la función de física (physics_step)
3. Generar la función de recompensa (calculate_reward)
"""

import os
import re
import json
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Intentar importar API de Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from domain_spec import DomainSpec, get_domain


@dataclass
class GeneratedCode:
    """Código generado por el arquitecto"""
    physics_code: str  # Función physics_step
    reward_code: str   # Función calculate_reward
    verify_code: str   # Función verify_domain_physics (semántica)
    success: bool
    error_message: str = ""
    raw_response: str = ""


# ============================================================
# PROMPTS UNIVERSALES
# ============================================================

UNIVERSAL_SYSTEM_PROMPT = """Eres un Ingeniero de Simulación Universal experto en:
- Física de sistemas robóticos
- Reinforcement Learning
- Programación en C de alto rendimiento

Tu trabajo es diseñar simulaciones para CUALQUIER sistema robótico.
Puedes generar código para drones, brazos robóticos, vehículos, péndulos, etc.

REGLAS ESTRICTAS:
1. SOLO usa math.h (fabsf, sqrtf, sinf, cosf, expf, etc.)
2. NO uses printf, malloc, free, ni funciones de sistema
3. El código debe ser EFICIENTE - se ejecutará millones de veces
4. Usa el operador -> para acceder a campos de structs
5. Todos los literales flotantes deben terminar en 'f' (ej: 0.5f, 1.0f)
6. Las recompensas deben estar en rango razonable (-10 a +10)

IMPORTANTE - FORMATO DE SALIDA:
- NO escribas "Aquí está el código" ni explicaciones
- NO uses Markdown (```c)
- Tu salida debe empezar DIRECTAMENTE con "void physics_step" o "float calculate_reward"
- Termina con la última llave "}"
- Cualquier texto fuera del código C puro romperá el sistema
"""

PHYSICS_GENERATION_PROMPT = """Genera la función physics_step para el siguiente sistema:

{domain_prompt}

La función debe tener esta firma:
```c
void physics_step({state_struct}* state, float* actions) {{
    // Tu código aquí
}}
```

Donde 'actions' es un array de {action_size} valores en rango [-1, 1]:
{action_descriptions}

REQUISITOS:
1. Implementa las ecuaciones de movimiento del sistema
2. Actualiza todos los campos de estado relevantes
3. USA INTEGRACIÓN RUNGE-KUTTA 4TO ORDEN (RK4):
   - k1 = f(y)
   - k2 = f(y + dt/2 * k1)
   - k3 = f(y + dt/2 * k2)
   - k4 = f(y + dt * k3)
   - y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
4. Aplica límites físicos razonables (clamp)
5. Incrementa state->steps al final

Responde SOLO con el código C de la función, sin explicaciones ni markdown.
"""

REWARD_GENERATION_PROMPT = """Genera la función calculate_reward para el siguiente sistema:

{domain_prompt}

TAREA ESPECÍFICA:
{task_description}

La función debe tener esta firma:
```c
float calculate_reward({state_struct}* state) {{
    // Tu código aquí
    return reward;
}}
```

REQUISITOS:
1. La recompensa debe guiar al agente hacia la tarea
2. Usa componentes aditivos (premios y penalizaciones)
3. Mantén la recompensa en rango [-10, +10]
4. Sé específico: no copies ejemplos genéricos

Responde SOLO con el código C de la función, sin explicaciones ni markdown.
"""

COMBINED_GENERATION_PROMPT = """Genera las funciones physics_step y calculate_reward para CartPole.

TAREA: {task_description}

NO INCLUYAS typedef del estado - ya existe. Solo genera estas DOS funciones:

void physics_step({state_struct}* state, float* actions) {{
    // USA RK4: calcular k1,k2,k3,k4 y aplicar formula RK4
    // Acciones: actions[0] en [-1,1] -> fuerza
    // Usar constantes: DT, DEFAULT_GRAVITY
    // Campos del estado: cart_position, cart_velocity, pole_angle, pole_velocity
    // Campos masa: cart_mass, pole_mass, pole_length
    state->steps++;
}}

float calculate_reward({state_struct}* state) {{
    // Recompensa por mantener el palo vertical y carro centrado
    return reward;
}}

Acciones: {action_descriptions}

RESPONDE SOLO CON CÓDIGO C. Empieza directamente con "void physics_step".
"""

VERIFY_GENERATION_PROMPT = """Genera una función de verificación semántica para validar que la física se comporta correctamente.

DOMINIO:
{domain_prompt}

FÍSICA GENERADA:
```c
{physics_code}
```

Necesito una función C con esta firma:
```c
int verify_domain_physics({state_struct}* state) {{
    // Retorna 1 si el estado es físicamente válido, 0 si no
}}
```

REQUISITOS:
1. Valida que el estado tenga valores SEMÁNTICAMENTE correctos
2. NO valides solo rangos numéricos, valida FÍSICA REAL
3. Ejemplos de checks:
   - Para drones: if (z < 0) return 0; // No puede estar bajo tierra
   - Para coches: if (wheel_angle > max_steering) return 0; // Giro excesivo
   - Para brazos: if (joint_angle < min_angle || joint_angle > max_angle) return 0;
   - Para cosas con velocidad: if (speed > max_speed) return 0; // Violación de límite
4. Usa lógica if/else simple, sin bucles
5. Retorna 0 si hay violación, 1 si todo está bien

Responde SOLO con el código C de la función, sin explicaciones ni markdown.
"""

REFLECTION_PROMPT = """Eres un experto analizando resultados de entrenamiento de RL.

CÓDIGO ANTERIOR:
```c
{previous_code}
```

MÉTRICAS DE ENTRENAMIENTO:
{metrics}

DIAGNÓSTICO:
{diagnosis}

Basándote en este análisis, genera una versión MEJORADA de la función de recompensa.

Cambios específicos a realizar:
{specific_changes}

Responde SOLO con el nuevo código C, sin explicaciones.
"""


class UniversalArchitect:
    """
    Arquitecto Universal que genera código para cualquier dominio robótico.

    Puede generar:
    - Función de física (physics_step)
    - Función de recompensa (calculate_reward)
    - Ambas juntas
    """

    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Inicializa el arquitecto.

        Args:
            api_key: API key de Gemini. Si no se proporciona, busca en env.
            use_mock: Si True, usa funciones mock en lugar de LLM.
        """
        self.use_mock = use_mock

        if not use_mock and GEMINI_AVAILABLE:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                genai.configure(api_key=key)
                # Usar Gemini 2.5 Flash
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                print("✅ Usando Gemini 2.5 Flash")
            else:
                print("Advertencia: No hay API key. Usando modo mock.")
                self.use_mock = True
        elif not GEMINI_AVAILABLE:
            self.use_mock = True

    def _call_llm(self, prompt: str) -> str:
        """Llama al LLM y retorna la respuesta"""
        if self.use_mock:
            return ""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 8192,  # Aumentado para funciones completas
                }
            )
            # Debug: verificar si se truncó
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish = str(candidate.finish_reason)
                    if 'MAX_TOKENS' in finish or 'LENGTH' in finish:
                        print(f"⚠️ Respuesta truncada por límite de tokens")
                    elif 'SAFETY' in finish:
                        print(f"⚠️ Respuesta bloqueada por filtros de seguridad")
            return response.text
        except Exception as e:
            print(f"Error llamando LLM: {e}")
            return ""

    def _extract_c_code(self, text: str) -> str:
        """Extrae código C de la respuesta (puede estar en markdown)"""
        # Buscar bloques de código
        code_pattern = r"```(?:c|C)?\s*\n(.*?)```"
        matches = re.findall(code_pattern, text, re.DOTALL)

        if matches:
            return "\n\n".join(matches).strip()

        # Si no hay bloques markdown, asumir que todo es código
        # Pero limpiar líneas que parecen explicaciones
        lines = []
        for line in text.split("\n"):
            # Ignorar líneas que parecen comentarios de explicación (sin //)
            if line.strip() and not line.strip().startswith(("Explicación", "Nota:", "Este código")):
                lines.append(line)

        return "\n".join(lines).strip()

    def _extract_function(self, code: str, func_signature: str) -> str:
        """Extrae una función C completa usando contador de llaves."""
        # Buscar inicio de la función con patrón más flexible
        # Permite cualquier caracter en los parámetros (incluyendo *, &, etc.)
        pattern = rf"({func_signature}\s*\(.*?\)\s*\{{)"
        match = re.search(pattern, code, re.DOTALL)
        if not match:
            # Intentar un enfoque más simple: buscar la posición del nombre de función
            func_pos = code.find(func_signature)
            if func_pos == -1:
                return ""
            # Buscar la primera llave después de la función
            brace_pos = code.find("{", func_pos)
            if brace_pos == -1:
                return ""
            start_pos = func_pos
        else:
            start_pos = match.start()

        brace_count = 0
        in_function = False
        end_pos = start_pos

        for i, char in enumerate(code[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end_pos = i + 1
                    break

        return code[start_pos:end_pos].strip()

    def _extract_physics_with_helpers(self, code: str) -> str:
        """
        Extrae todo el código de física incluyendo typedef, calc_derivs y physics_step.
        Captura desde el inicio hasta el final de physics_step.
        """
        # Primero intentar extraer solo physics_step
        physics_func = self._extract_function(code, "void physics_step")

        # Encontrar donde empieza calculate_reward para saber dónde cortar
        reward_pos = code.find("float calculate_reward")
        if reward_pos == -1:
            # Si no hay reward, usar solo physics_step si existe
            return physics_func if physics_func else code.strip()

        # Todo lo que está antes de calculate_reward es código de física
        physics_section = code[:reward_pos].strip()

        # Verificar que contiene physics_step
        if "void physics_step" in physics_section:
            return physics_section

        return physics_func if physics_func else ""

    def _validate_physics_code(self, code: str, domain: DomainSpec) -> Tuple[bool, str]:
        """Valida código de física"""
        if "void physics_step" not in code:
            return False, "Falta la función physics_step"

        if "state->" not in code:
            return False, "No usa state-> para acceder al estado"

        if "steps" not in code:
            return False, "No incrementa state->steps"

        # Verificar que no usa funciones prohibidas
        forbidden = ["printf", "malloc", "free", "scanf", "fopen"]
        for fn in forbidden:
            if fn in code:
                return False, f"Usa función prohibida: {fn}"

        return True, ""

    def _validate_reward_code(self, code: str, domain: DomainSpec) -> Tuple[bool, str]:
        """Valida código de recompensa"""
        if "float calculate_reward" not in code:
            return False, "Falta la función calculate_reward"

        if "return" not in code:
            return False, "No hay return en la función"

        if "state->" not in code:
            return False, "No usa state-> para acceder al estado"

        return True, ""

    def generate_physics(self, domain: DomainSpec) -> GeneratedCode:
        """
        Genera la función de física para un dominio.

        Args:
            domain: Especificación del dominio

        Returns:
            GeneratedCode con la función physics_step
        """
        if self.use_mock:
            return self._mock_physics(domain)

        # Preparar prompt
        action_descriptions = "\n".join(
            f"  [{i}] {a.name}: {a.description} (rango real: [{a.min_value}, {a.max_value}])"
            for i, a in enumerate(domain.action_fields)
        )

        prompt = UNIVERSAL_SYSTEM_PROMPT + "\n\n" + PHYSICS_GENERATION_PROMPT.format(
            domain_prompt=domain.to_architect_prompt(),
            state_struct=domain.state_struct_name,
            action_size=domain.action_size,
            action_descriptions=action_descriptions,
        )

        # Llamar LLM
        response = self._call_llm(prompt)
        if not response:
            return GeneratedCode("", "", "", False, "No hubo respuesta del LLM")

        # Extraer y validar código
        code = self._extract_c_code(response)
        valid, error = self._validate_physics_code(code, domain)

        return GeneratedCode(
            physics_code=code if valid else "",
            reward_code="",
            verify_code="",
            success=valid,
            error_message=error,
            raw_response=response,
        )

    def generate_reward(
        self,
        domain: DomainSpec,
        task_description: str,
        feedback: Optional[str] = None
    ) -> GeneratedCode:
        """
        Genera función de recompensa para una tarea.

        Args:
            domain: Especificación del dominio
            task_description: Descripción de la tarea
            feedback: Retroalimentación opcional de entrenamientos previos

        Returns:
            GeneratedCode con la función calculate_reward
        """
        if self.use_mock:
            return self._mock_reward(domain, task_description)

        prompt = UNIVERSAL_SYSTEM_PROMPT + "\n\n" + REWARD_GENERATION_PROMPT.format(
            domain_prompt=domain.to_architect_prompt(),
            state_struct=domain.state_struct_name,
            task_description=task_description,
        )

        if feedback:
            prompt += f"\n\nFEEDBACK DE ENTRENAMIENTO PREVIO:\n{feedback}"

        response = self._call_llm(prompt)
        if not response:
            return GeneratedCode("", "", "", False, "No hubo respuesta del LLM")

        code = self._extract_c_code(response)
        valid, error = self._validate_reward_code(code, domain)

        return GeneratedCode(
            physics_code="",
            reward_code=code if valid else "",
            verify_code="",
            success=valid,
            error_message=error,
            raw_response=response,
        )

    def generate_both(
        self,
        domain: DomainSpec,
        task_description: str
    ) -> GeneratedCode:
        """
        Genera TANTO física como recompensa en una sola llamada.

        Args:
            domain: Especificación del dominio
            task_description: Descripción de la tarea

        Returns:
            GeneratedCode con ambas funciones
        """
        if self.use_mock:
            physics = self._mock_physics(domain)
            reward = self._mock_reward(domain, task_description)
            return GeneratedCode(
            physics_code=physics.physics_code,
            reward_code=reward.reward_code,
            verify_code="",
            success=physics.success and reward.success,
            )

        action_descriptions = "\n".join(
            f"  [{i}] {a.name}: {a.description}"
            for i, a in enumerate(domain.action_fields)
        )

        prompt = UNIVERSAL_SYSTEM_PROMPT + "\n\n" + COMBINED_GENERATION_PROMPT.format(
            domain_prompt=domain.to_architect_prompt(),
            state_struct=domain.state_struct_name,
            task_description=task_description,
            action_size=domain.action_size,
            action_descriptions=action_descriptions,
        )

        response = self._call_llm(prompt)
        if not response:
            print("⚠️ LLM no respondió, usando fallback mock...")
            return self._fallback_to_mock(domain, task_description)

        # DEBUG
        print(f"🔍 Respuesta LLM ({len(response)} chars):")
        print(response[:800] if len(response) > 800 else response)
        print("-" * 50)

        # Separar las dos funciones usando el extractor
        code = self._extract_c_code(response)

        # Extraer physics_step con todo el código previo (typedef, calc_derivs, etc.)
        physics_code = self._extract_physics_with_helpers(code)
        reward_code = self._extract_function(code, "float calculate_reward")

        physics_valid, physics_error = self._validate_physics_code(physics_code, domain) if physics_code else (False, "No se encontró physics_step")
        reward_valid, reward_error = self._validate_reward_code(reward_code, domain) if reward_code else (False, "No se encontró calculate_reward")

        # Si alguna función falló, usar fallback mock
        if not physics_valid or not reward_valid:
            print(f"⚠️ LLM generó código incompleto (Physics: {physics_error}, Reward: {reward_error})")
            print("   Usando fallback mock...")
            return self._fallback_to_mock(domain, task_description)

        print("✅ Código generado por LLM validado correctamente")
        return GeneratedCode(
            physics_code=physics_code,
            reward_code=reward_code,
            verify_code="",
            success=True,
            raw_response=response,
        )

    def _fallback_to_mock(self, domain: DomainSpec, task_description: str) -> GeneratedCode:
        """Fallback al código mock cuando el LLM falla."""
        physics = self._mock_physics(domain)
        reward = self._mock_reward(domain, task_description)
        return GeneratedCode(
            physics_code=physics.physics_code,
            reward_code=reward.reward_code,
            verify_code="",
            success=physics.success and reward.success,
        )

    def evolve_reward(
        self,
        domain: DomainSpec,
        previous_code: str,
        metrics: Dict[str, Any],
        diagnosis: str
    ) -> GeneratedCode:
        """
        Evoluciona una función de recompensa basándose en métricas.

        Args:
            domain: Especificación del dominio
            previous_code: Código de recompensa anterior
            metrics: Métricas del entrenamiento
            diagnosis: Diagnóstico semántico de los problemas

        Returns:
            GeneratedCode con la función mejorada
        """
        if self.use_mock:
            # En modo mock, retornar el mismo código
            return GeneratedCode("", previous_code, True)

        # Generar cambios específicos basados en métricas
        specific_changes = self._generate_specific_changes(metrics)

        prompt = UNIVERSAL_SYSTEM_PROMPT + "\n\n" + REFLECTION_PROMPT.format(
            previous_code=previous_code,
            metrics=json.dumps(metrics, indent=2),
            diagnosis=diagnosis,
            specific_changes=specific_changes,
        )

        response = self._call_llm(prompt)
        if not response:
            return GeneratedCode("", "", "", False, "No hubo respuesta del LLM")

        code = self._extract_c_code(response)
        valid, error = self._validate_reward_code(code, domain)

        return GeneratedCode(
            physics_code="",
            reward_code=code if valid else previous_code,
            verify_code="",
            success=valid,
            error_message=error,
            raw_response=response,
        )

    def _generate_specific_changes(self, metrics: Dict[str, Any]) -> str:
        """Genera lista de cambios específicos basados en métricas"""
        changes = []

        mean_reward = metrics.get("mean_reward", 0)
        if mean_reward < -5:
            changes.append("- La recompensa es demasiado negativa. Reduce las penalizaciones a la mitad.")

        mean_steps = metrics.get("mean_steps", 0)
        if mean_steps < 20:
            changes.append("- Los episodios terminan muy rápido. Prioriza la supervivencia sobre el objetivo.")

        success_rate = metrics.get("success_rate", 0)
        if success_rate == 0:
            changes.append("- Tasa de éxito 0%. Simplifica la recompensa drásticamente.")

        final_distance = metrics.get("final_distance", float("inf"))
        if final_distance > 2.0:
            changes.append("- No alcanza el objetivo. Aumenta el gradiente de recompensa por cercanía.")

        if not changes:
            changes.append("- El entrenamiento va bien. Refina los detalles para mejorar la suavidad.")

        return "\n".join(changes)

    # ============================================================
    # FUNCIONES MOCK (para testing sin LLM)
    # ============================================================

    def _mock_physics(self, domain: DomainSpec) -> GeneratedCode:
        """Genera física mock basada en el dominio"""

        if domain.name == "Drone":
            code = self._mock_drone_physics()
        elif domain.name == "CartPole":
            code = self._mock_cartpole_physics()
        elif domain.name == "RoboticArm2D":
            code = self._mock_arm_physics()
        elif domain.name == "WarehouseRobot":
            code = self._mock_warehouse_physics()
        else:
            code = self._mock_generic_physics(domain)

        return GeneratedCode(physics_code=code, reward_code="", verify_code="", success=True)

    def _mock_reward(self, domain: DomainSpec, task: str) -> GeneratedCode:
        """Genera recompensa mock basada en el dominio y tarea"""

        task_lower = task.lower()

        if domain.name == "Drone":
            if "hover" in task_lower:
                code = self._mock_drone_hover_reward()
            elif "waypoint" in task_lower:
                code = self._mock_drone_waypoint_reward()
            else:
                code = self._mock_drone_hover_reward()

        elif domain.name == "CartPole":
            code = self._mock_cartpole_reward()

        elif domain.name == "RoboticArm2D":
            code = self._mock_arm_reward()

        elif domain.name == "WarehouseRobot":
            code = self._mock_warehouse_reward()

        else:
            code = self._mock_generic_reward(domain)

        return GeneratedCode(physics_code="", reward_code=code, verify_code="", success=True)

    def _mock_drone_physics(self) -> str:
        return """void physics_step(DroneState* state, float* actions) {
    // Convertir acciones normalizadas
    float thrust = (actions[0] + 1.0f) * 0.5f * 20.0f;
    float roll_cmd = actions[1] * 5.0f;
    float pitch_cmd = actions[2] * 5.0f;
    float yaw_cmd = actions[3] * 5.0f;

    // Actualizar velocidades angulares
    float damping = 0.95f;
    state->roll_rate = state->roll_rate * damping + roll_cmd * 0.1f;
    state->pitch_rate = state->pitch_rate * damping + pitch_cmd * 0.1f;
    state->yaw_rate = state->yaw_rate * damping + yaw_cmd * 0.1f;

    // Integrar orientación
    state->roll += state->roll_rate * DT;
    state->pitch += state->pitch_rate * DT;
    state->yaw += state->yaw_rate * DT;

    // Limitar orientación
    if (state->roll > 1.0f) state->roll = 1.0f;
    if (state->roll < -1.0f) state->roll = -1.0f;
    if (state->pitch > 1.0f) state->pitch = 1.0f;
    if (state->pitch < -1.0f) state->pitch = -1.0f;

    // Calcular fuerzas en frame mundo
    float cos_roll = cosf(state->roll);
    float sin_roll = sinf(state->roll);
    float cos_pitch = cosf(state->pitch);
    float sin_pitch = sinf(state->pitch);

    float thrust_x = thrust * sin_pitch;
    float thrust_y = -thrust * sin_roll * cos_pitch;
    float thrust_z = thrust * cos_roll * cos_pitch;

    // Aceleración
    float mass = state->mass > 0.0f ? state->mass : 1.0f;
    float gravity = state->gravity > 0.0f ? state->gravity : 9.81f;

    float ax = (thrust_x + state->wind_x) / mass;
    float ay = (thrust_y + state->wind_y) / mass;
    float az = ((thrust_z + state->wind_z) / mass) - gravity;

    // Arrastre
    float drag = state->drag_coeff > 0.0f ? state->drag_coeff : 0.1f;
    ax -= state->vx * drag;
    ay -= state->vy * drag;
    az -= state->vz * drag;

    // Integrar velocidad y posición
    state->vx += ax * DT;
    state->vy += ay * DT;
    state->vz += az * DT;

    state->x += state->vx * DT;
    state->y += state->vy * DT;
    state->z += state->vz * DT;

    // Colisión con suelo
    if (state->z < 0.0f) {
        state->z = 0.0f;
        state->vz = 0.0f;
        state->collisions++;
    }

    // Límites
    if (state->z > WORLD_SIZE_Z) state->z = WORLD_SIZE_Z;
    if (state->x > WORLD_SIZE_X) state->x = WORLD_SIZE_X;
    if (state->x < -WORLD_SIZE_X) state->x = -WORLD_SIZE_X;
    if (state->y > WORLD_SIZE_Y) state->y = WORLD_SIZE_Y;
    if (state->y < -WORLD_SIZE_Y) state->y = -WORLD_SIZE_Y;

    state->steps++;
}"""

    def _mock_cartpole_physics(self) -> str:
        return """void physics_step(CartPoleState* state, float* actions) {
    // Parámetros físicos
    float force = actions[0] * 10.0f;
    float cart_mass = state->cart_mass > 0.0f ? state->cart_mass : 1.0f;
    float pole_mass = state->pole_mass > 0.0f ? state->pole_mass : 0.1f;
    float pole_length = state->pole_length > 0.0f ? state->pole_length : 0.5f;
    float total_mass = cart_mass + pole_mass;
    float pole_half = pole_length * 0.5f;
    float dt = DT;
    float g = DEFAULT_GRAVITY;

    // Estado actual
    float x = state->cart_position;
    float x_dot = state->cart_velocity;
    float theta = state->pole_angle;
    float theta_dot = state->pole_velocity;

    // Función de derivadas del CartPole
    #define CARTPOLE_DERIVS(th, th_dot, x_d_out, th_d_out) { \\
        float st = sinf(th); \\
        float ct = cosf(th); \\
        float tmp = (force + pole_mass * pole_half * th_dot * th_dot * st) / total_mass; \\
        th_d_out = (g * st - ct * tmp) / (pole_half * (4.0f/3.0f - pole_mass * ct * ct / total_mass)); \\
        x_d_out = tmp - pole_mass * pole_half * th_d_out * ct / total_mass; \\
    }

    // RK4 - k1
    float k1_x_dot, k1_theta_dot;
    CARTPOLE_DERIVS(theta, theta_dot, k1_x_dot, k1_theta_dot);
    float k1_x = x_dot;
    float k1_theta = theta_dot;

    // RK4 - k2
    float k2_x_dot, k2_theta_dot;
    CARTPOLE_DERIVS(theta + 0.5f*dt*k1_theta, theta_dot + 0.5f*dt*k1_theta_dot, k2_x_dot, k2_theta_dot);
    float k2_x = x_dot + 0.5f*dt*k1_x_dot;
    float k2_theta = theta_dot + 0.5f*dt*k1_theta_dot;

    // RK4 - k3
    float k3_x_dot, k3_theta_dot;
    CARTPOLE_DERIVS(theta + 0.5f*dt*k2_theta, theta_dot + 0.5f*dt*k2_theta_dot, k3_x_dot, k3_theta_dot);
    float k3_x = x_dot + 0.5f*dt*k2_x_dot;
    float k3_theta = theta_dot + 0.5f*dt*k2_theta_dot;

    // RK4 - k4
    float k4_x_dot, k4_theta_dot;
    CARTPOLE_DERIVS(theta + dt*k3_theta, theta_dot + dt*k3_theta_dot, k4_x_dot, k4_theta_dot);
    float k4_x = x_dot + dt*k3_x_dot;
    float k4_theta = theta_dot + dt*k3_theta_dot;

    // Integración RK4
    state->cart_position = x + (dt/6.0f) * (k1_x + 2.0f*k2_x + 2.0f*k3_x + k4_x);
    state->cart_velocity = x_dot + (dt/6.0f) * (k1_x_dot + 2.0f*k2_x_dot + 2.0f*k3_x_dot + k4_x_dot);
    state->pole_angle = theta + (dt/6.0f) * (k1_theta + 2.0f*k2_theta + 2.0f*k3_theta + k4_theta);
    state->pole_velocity = theta_dot + (dt/6.0f) * (k1_theta_dot + 2.0f*k2_theta_dot + 2.0f*k3_theta_dot + k4_theta_dot);

    #undef CARTPOLE_DERIVS
    state->steps++;
}"""

    def _mock_arm_physics(self) -> str:
        return """void physics_step(RoboticArm2DState* state, float* actions) {
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
}"""

    def _mock_warehouse_physics(self) -> str:
        return """void physics_step(WarehouseRobotState* state, float* actions) {
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
}"""

    def _mock_generic_physics(self, domain: DomainSpec) -> str:
        """Genera física genérica para dominios desconocidos"""
        return f"""void physics_step({domain.state_struct_name}* state, float* actions) {{
    // Física genérica - actualizar según las acciones
    // TODO: Implementar física específica para {domain.name}

    state->steps++;
}}"""

    def _mock_drone_hover_reward(self) -> str:
        return """float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa inversamente proporcional a la distancia
    float reward = 1.0f / (1.0f + distance);

    // Bonus por estar cerca
    if (distance < 0.2f) {
        reward += 5.0f;
    }

    // Penalización por orientación extrema
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 0.5f;
    reward -= orientation_penalty;

    // Penalización por velocidad angular alta
    float angular_penalty = (fabsf(state->roll_rate) + fabsf(state->pitch_rate) + fabsf(state->yaw_rate)) * 0.1f;
    reward -= angular_penalty;

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 5.0f;
    }

    return reward;
}"""

    def _mock_drone_waypoint_reward(self) -> str:
        return """float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa por cercanía
    float reward = 2.0f - distance;

    // Bonus grande por llegar
    if (distance < 0.2f) {
        reward += 10.0f;
    } else if (distance < 0.5f) {
        reward += 3.0f;
    }

    // Penalización por velocidad excesiva
    float speed = sqrtf(state->vx*state->vx + state->vy*state->vy + state->vz*state->vz);
    if (speed > 3.0f) {
        reward -= (speed - 3.0f) * 0.5f;
    }

    // Penalización leve por orientación
    reward -= (fabsf(state->roll) + fabsf(state->pitch)) * 0.2f;

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 10.0f;
    }

    return reward;
}"""

    def _mock_cartpole_reward(self) -> str:
        return """float calculate_reward(CartPoleState* state) {
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
}"""

    def _mock_arm_reward(self) -> str:
        return """float calculate_reward(RoboticArm2DState* state) {
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
}"""

    def _mock_warehouse_reward(self) -> str:
        return """float calculate_reward(WarehouseRobotState* state) {
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
}"""

    def _mock_generic_reward(self, domain: DomainSpec) -> str:
        """Genera recompensa genérica"""
        return f"""float calculate_reward({domain.state_struct_name}* state) {{
    // Recompensa genérica
    float reward = 1.0f;  // Recompensa base por sobrevivir

    // TODO: Implementar recompensa específica para {domain.name}

    return reward;
}}"""

    def generate_verification(
        self,
        domain: DomainSpec,
        physics_code: str
    ) -> str:
        """
        Genera función de verificación semántica en C.

        Args:
            domain: Especificación del dominio
            physics_code: Código de física ya generado

        Returns:
            Código C de la función verify_domain_physics()
        """
        if self.use_mock:
            return self._mock_verification(domain)

        prompt = UNIVERSAL_SYSTEM_PROMPT + "\n\n" + VERIFY_GENERATION_PROMPT.format(
            domain_prompt=domain.to_architect_prompt(),
            state_struct=domain.state_struct_name,
            physics_code=physics_code,
        )

        response = self._call_llm(prompt)
        if not response:
            return self._mock_verification(domain)

        code = self._extract_c_code(response)
        valid, _ = self._validate_verify_code(code, domain)

        return code if valid else self._mock_verification(domain)

    def _validate_verify_code(self, code: str, domain: DomainSpec) -> Tuple[bool, str]:
        """Valida código de verificación"""
        if "int verify_domain_physics" not in code:
            return False, "Falta la función verify_domain_physics"

        if "return" not in code:
            return False, "No retorna valor"

        # Verificar que no usa funciones prohibidas
        forbidden = ["printf", "malloc", "free", "scanf", "fopen", "for (", "while ("]
        for fn in forbidden:
            if fn in code:
                return False, f"Usa elemento prohibido: {fn}"

        return True, ""

    def _mock_verification(self, domain: DomainSpec) -> str:
        """Genera verificación mock basada en el dominio"""
        if domain.name == "Drone":
            return self._mock_drone_verify()
        elif domain.name == "CartPole":
            return self._mock_cartpole_verify()
        elif domain.name == "RoboticArm2D":
            return self._mock_arm_verify()
        elif domain.name == "WarehouseRobot":
            return self._mock_warehouse_verify()
        else:
            return self._mock_generic_verify(domain)

    def _mock_drone_verify(self) -> str:
        """Verificación mock para drone"""
        return """int verify_domain_physics(DroneState* state) {
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
}"""

    def _mock_cartpole_verify(self) -> str:
        """Verificación mock para cartpole"""
        return """int verify_domain_physics(CartPoleState* state) {
    // Ángulo del palo en rango [-pi, pi]
    if (fabsf(state->pole_angle) > 3.14159f) return 0;

    // Posición del carro dentro de límites
    if (fabsf(state->cart_position) > 2.4f) return 0;

    // Velocidades razonables
    if (fabsf(state->cart_velocity) > 10.0f) return 0;
    if (fabsf(state->pole_velocity) > 10.0f) return 0;

    return 1;
}"""

    def _mock_arm_verify(self) -> str:
        """Verificación mock para brazo robótico"""
        return """int verify_domain_physics(RoboticArm2DState* state) {
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
}"""

    def _mock_warehouse_verify(self) -> str:
        """Verificación mock para warehouse robot"""
        return """int verify_domain_physics(WarehouseRobotState* state) {
    // Posición dentro del almacén
    if (state->x < 0.0f || state->x > 10.0f) return 0;
    if (state->y < 0.0f || state->y > 10.0f) return 0;

    // Orientación en rango válido
    if (fabsf(state->theta) > 3.14159f) return 0;

    // Velocidades razonables
    if (fabsf(state->v_linear) > 5.0f) return 0;
    if (fabsf(state->v_angular) > 3.14159f) return 0;

    // Distancia a obstáculos debe ser positiva
    if (state->obstacle_front < 0.0f) return 0;
    if (state->obstacle_left < 0.0f) return 0;
    if (state->obstacle_right < 0.0f) return 0;

    return 1;
}"""

    def _mock_generic_verify(self, domain: DomainSpec) -> str:
        """Verificación genérica"""
        return f"""int verify_domain_physics({domain.state_struct_name}* state) {{
    // Verificación genérica: solo chequear que el estado no es NaN
    // TODO: Implementar verificación específica para {domain.name}
    return 1;
}}"""


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def create_architect(use_mock: bool = False) -> UniversalArchitect:
    """Crea un arquitecto universal"""
    return UniversalArchitect(use_mock=use_mock)


def generate_for_domain(
    domain_name: str,
    task_description: str,
    use_mock: bool = False
) -> GeneratedCode:
    """
    Genera código para un dominio predefinido.

    Args:
        domain_name: Nombre del dominio (drone, cartpole, etc.)
        task_description: Descripción de la tarea
        use_mock: Si usar funciones mock

    Returns:
        GeneratedCode con física y recompensa
    """
    domain = get_domain(domain_name)
    architect = create_architect(use_mock=use_mock)
    return architect.generate_both(domain, task_description)


if __name__ == "__main__":
    # Demo
    print("=== Universal Architect Demo ===\n")

    # Crear arquitecto en modo mock
    architect = UniversalArchitect(use_mock=True)

    # Probar con diferentes dominios
    for domain_name in ["drone", "cartpole", "robotic_arm", "warehouse_robot"]:
        print(f"\n{'='*50}")
        print(f"Dominio: {domain_name}")
        print('='*50)

        domain = get_domain(domain_name)

        # Generar física
        physics = architect.generate_physics(domain)
        print(f"\nFísica generada: {'OK' if physics.success else 'FALLO'}")
        if physics.physics_code:
            print(physics.physics_code[:200] + "...")

        # Generar recompensa
        reward = architect.generate_reward(domain, "Alcanzar el objetivo de forma suave")
        print(f"\nRecompensa generada: {'OK' if reward.success else 'FALLO'}")
        if reward.reward_code:
            print(reward.reward_code[:200] + "...")
