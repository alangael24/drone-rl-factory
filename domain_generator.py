"""
Domain Generator - Genera DomainSpec desde lenguaje natural (GenSim-Lite)

Basado en los papers:
- GenSim (Wang et al., 2024): LLM genera entornos programáticamente
- DrEureka (Ma et al., 2024): Co-evolución de física y recompensa

Este módulo permite crear dominios robóticos completos desde una
instrucción en lenguaje natural, sin necesidad de definirlos manualmente.
"""

import os
import re
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from domain_spec import (
    DomainSpec, StateField, ActionField, PhysicsConstants,
    TerminationCondition, FieldType
)


@dataclass
class GeneratedDomain:
    """Resultado de la generación de dominio"""
    domain: Optional[DomainSpec]
    physics_code: str
    reward_code: str
    success: bool
    error_message: str = ""
    raw_response: str = ""


@dataclass
class PhysicsCritique:
    """Crítica del LLM sobre la física generada"""
    is_realistic: bool
    is_solvable: bool
    issues: List[str]
    suggestions: List[str]
    confidence: float  # 0-1


# ============================================================
# PROMPTS PARA GENERACIÓN DE DOMINIOS
# ============================================================

DOMAIN_GENERATION_PROMPT = """Eres un experto en simulación física y robótica.

Tu trabajo es diseñar un sistema robótico completo basado en una descripción en lenguaje natural.

INSTRUCCIÓN DEL USUARIO:
"{instruction}"

Debes generar un JSON con la siguiente estructura:

```json
{{
    "name": "NombreDelSistema",
    "description": "Descripción breve del sistema",

    "state_fields": [
        {{"name": "x", "type": "float", "description": "Posición X", "normalize_by": 10.0, "default": 0.0}},
        {{"name": "y", "type": "float", "description": "Posición Y", "normalize_by": 10.0, "default": 0.0}},
        // ... más campos según el sistema
    ],

    "action_fields": [
        {{"name": "throttle", "min": -1.0, "max": 1.0, "description": "Acelerador"}},
        // ... más acciones
    ],

    "physics_constants": {{
        "dt": 0.02,
        "gravity": 9.81,
        "world_bounds": [20.0, 20.0, 10.0],
        "max_episode_steps": 500,
        "custom": {{
            "MASS": 1000.0,
            "FRICTION": 0.1
            // ... constantes específicas del sistema
        }}
    }},

    "termination_conditions": [
        {{"name": "reached_goal", "condition": "distancia < 0.5f", "description": "Llegó al objetivo"}},
        {{"name": "timeout", "condition": "steps >= MAX_EPISODE_STEPS", "description": "Tiempo agotado"}}
    ],

    "physics_description": "Descripción detallada de la física para el generador de código C",

    "reward_hints": "Sugerencias para la función de recompensa"
}}
```

REGLAS:
1. Incluye TODAS las variables de estado necesarias para simular el sistema
2. Para vehículos con líquidos: incluye masa variable, centro de masa, fuerzas de sloshing
3. Para robots articulados: incluye ángulos y velocidades de cada joint
4. Las acciones deben mapear a actuadores reales (motores, válvulas, etc.)
5. Incluye condiciones de terminación realistas (colisión, éxito, timeout)
6. La física debe ser simulable en C con math.h

Responde SOLO con el JSON, sin explicaciones adicionales.
"""

PHYSICS_GENERATION_PROMPT = """Genera el código C para physics_step basado en este dominio:

DOMINIO: {domain_name}
DESCRIPCIÓN: {domain_description}

ESTADO (struct {state_struct}):
{state_fields}

ACCIONES: {action_fields}

CONSTANTES DISPONIBLES:
{constants}

FÍSICA REQUERIDA:
{physics_description}

Genera la función:
```c
void physics_step({state_struct}* state, float* actions) {{
    // Tu código aquí
    state->steps++;
}}
```

REGLAS:
1. Usa integración RK4 para ecuaciones diferenciales
2. Solo usa funciones de math.h (sinf, cosf, sqrtf, fabsf, expf)
3. Aplica límites físicos realistas (clamp)
4. NO uses printf, malloc, ni funciones de sistema
5. Incrementa state->steps al final

Responde SOLO con el código C.
"""

REWARD_GENERATION_PROMPT = """Genera una función de recompensa en C para entrenar un agente RL.

DOMINIO: {domain_name}
DESCRIPCIÓN: {domain_description}
INSTRUCCIÓN ORIGINAL: {instruction}

ESTADO (struct {state_struct}):
{state_fields}

HINTS DE RECOMPENSA:
{reward_hints}

CONSTANTES DISPONIBLES:
{constants}

Genera la función:
```c
float calculate_reward({state_struct}* state) {{
    // Tu código aquí
}}
```

REGLAS:
1. La recompensa debe guiar al agente hacia el objetivo
2. Incluye términos para:
   - Progreso hacia la meta (distancia)
   - Penalización por estados indeseados
   - Bonus por completar la tarea
   - Shaping para facilitar el aprendizaje
3. Normaliza los términos para que estén en rangos similares
4. Solo usa funciones de math.h (sqrtf, fabsf, expf, fminf, fmaxf)
5. NO uses printf ni funciones de sistema

Responde SOLO con el código C de la función.
"""

PHYSICS_CRITIQUE_PROMPT = """Eres un físico experto evaluando una simulación.

SISTEMA: {domain_name}
DESCRIPCIÓN ESPERADA: {description}

CÓDIGO DE FÍSICA:
```c
{physics_code}
```

TRAYECTORIA DE PRUEBA (últimos 50 pasos):
{trajectory_sample}

MÉTRICAS:
- Posición final: ({final_x:.2f}, {final_y:.2f})
- Velocidad promedio: {avg_speed:.2f}
- Rango de valores: {value_ranges}

EVALÚA:
1. ¿La física es REALISTA para "{domain_name}"?
2. ¿El agente PUEDE resolver la tarea con esta física?
3. ¿Hay comportamientos IMPOSIBLES (volar sin alas, atravesar paredes)?

Responde en JSON:
```json
{{
    "is_realistic": true/false,
    "is_solvable": true/false,
    "confidence": 0.0-1.0,
    "issues": ["lista de problemas detectados"],
    "suggestions": ["lista de mejoras sugeridas"]
}}
```
"""


class DomainGenerator:
    """
    Meta-Arquitecto que genera dominios completos desde lenguaje natural.

    Implementa el concepto de GenSim: dado un texto como "una pipa de agua
    que va de A a B", genera automáticamente:
    - DomainSpec con variables de estado apropiadas
    - Código de física realista
    - Sugerencias de recompensa
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            self.use_mock = False
            print("✅ DomainGenerator usando Gemini 2.5 Flash")
        else:
            self.use_mock = True
            print("⚠️ DomainGenerator en modo mock")

    def _call_llm(self, prompt: str) -> str:
        """Llama al LLM"""
        if self.use_mock:
            return ""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                }
            )
            return response.text
        except Exception as e:
            print(f"Error LLM: {e}")
            return ""

    def generate_domain(self, instruction: str) -> GeneratedDomain:
        """
        Genera un DomainSpec completo desde una instrucción en lenguaje natural.

        Args:
            instruction: Descripción del sistema deseado (ej: "pipa de agua de A a B")

        Returns:
            GeneratedDomain con el dominio, física y recompensa
        """
        print(f"\n🧠 Generando dominio para: '{instruction}'")

        if self.use_mock:
            return self._mock_domain(instruction)

        # 1. Generar especificación del dominio
        prompt = DOMAIN_GENERATION_PROMPT.format(instruction=instruction)
        response = self._call_llm(prompt)

        if not response:
            return GeneratedDomain(None, "", "", False, "No hubo respuesta del LLM")

        # 2. Parsear JSON
        try:
            domain_dict = self._extract_json(response)
            domain = self._dict_to_domain(domain_dict)
        except Exception as e:
            return GeneratedDomain(None, "", "", False, f"Error parseando dominio: {e}", response)

        # 3. Generar código de física
        physics_code = self._generate_physics(domain)

        # 4. Generar código de recompensa inicial
        reward_code = self._generate_initial_reward(domain, instruction)

        return GeneratedDomain(
            domain=domain,
            physics_code=physics_code,
            reward_code=reward_code,
            success=True,
            raw_response=response
        )

    def _extract_json(self, text: str) -> Dict:
        """Extrae JSON de la respuesta del LLM"""
        # Buscar bloque JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Intentar parsear directamente
        # Buscar desde la primera { hasta la última }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        raise ValueError("No se encontró JSON válido en la respuesta")

    def _dict_to_domain(self, d: Dict) -> DomainSpec:
        """Convierte diccionario a DomainSpec"""
        # Convertir state_fields
        state_fields = []
        for sf in d.get("state_fields", []):
            field_type = FieldType.FLOAT
            if sf.get("type") == "int":
                field_type = FieldType.INT

            state_fields.append(StateField(
                name=sf["name"],
                field_type=field_type,
                description=sf.get("description", ""),
                default_value=sf.get("default", 0.0),
                normalize_by=sf.get("normalize_by", 1.0),
            ))

        # Convertir action_fields
        action_fields = []
        for af in d.get("action_fields", []):
            action_fields.append(ActionField(
                name=af["name"],
                min_value=af.get("min", -1.0),
                max_value=af.get("max", 1.0),
                description=af.get("description", ""),
            ))

        # Convertir physics_constants
        pc = d.get("physics_constants", {})
        physics_constants = PhysicsConstants(
            dt=pc.get("dt", 0.02),
            gravity=pc.get("gravity", 9.81),
            world_bounds=tuple(pc.get("world_bounds", [20.0, 20.0, 10.0])),
            max_episode_steps=pc.get("max_episode_steps", 500),
            custom=pc.get("custom", {}),
        )

        # Convertir termination_conditions
        termination_conditions = []
        for tc in d.get("termination_conditions", []):
            termination_conditions.append(TerminationCondition(
                name=tc["name"],
                condition_code=tc.get("condition", "0"),
                description=tc.get("description", ""),
            ))

        return DomainSpec(
            name=d.get("name", "GeneratedDomain"),
            description=d.get("description", ""),
            state_fields=state_fields,
            action_fields=action_fields,
            physics_constants=physics_constants,
            termination_conditions=termination_conditions,
            physics_description=d.get("physics_description", ""),
            reward_hints=d.get("reward_hints", ""),
        )

    def _generate_physics(self, domain: DomainSpec) -> str:
        """Genera código C de física para el dominio"""
        state_fields_str = "\n".join(
            f"  - {f.name}: {f.description}" for f in domain.state_fields
        )
        action_fields_str = ", ".join(
            f"{a.name}[{a.min_value},{a.max_value}]" for a in domain.action_fields
        )
        constants_str = domain.physics_constants.to_c_defines()

        prompt = PHYSICS_GENERATION_PROMPT.format(
            domain_name=domain.name,
            domain_description=domain.description,
            state_struct=domain.state_struct_name,
            state_fields=state_fields_str,
            action_fields=action_fields_str,
            constants=constants_str,
            physics_description=domain.physics_description,
        )

        response = self._call_llm(prompt)
        if not response:
            return self._mock_physics(domain)

        # Extraer código C
        code_match = re.search(r'```c?\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return response.strip()

    def _generate_initial_reward(self, domain: DomainSpec, instruction: str) -> str:
        """Genera una función de recompensa usando el LLM"""
        if self.use_mock:
            return self._mock_reward(domain)

        state_fields_str = "\n".join(
            f"  - {f.name}: {f.description}" for f in domain.state_fields
        )
        constants_str = domain.physics_constants.to_c_defines()

        prompt = REWARD_GENERATION_PROMPT.format(
            domain_name=domain.name,
            domain_description=domain.description,
            instruction=instruction,
            state_struct=domain.state_struct_name,
            state_fields=state_fields_str,
            reward_hints=domain.reward_hints,
            constants=constants_str,
        )

        response = self._call_llm(prompt)
        if not response:
            return self._mock_reward(domain)

        # Extraer código C
        code_match = re.search(r'```c?\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return response.strip()

    def _mock_reward(self, domain: DomainSpec) -> str:
        """Genera recompensa mock cuando no hay LLM"""
        # Buscar campos de objetivo
        goal_field = None
        pos_field = None
        for f in domain.state_fields:
            if "goal" in f.name.lower() or "target" in f.name.lower():
                goal_field = f.name.replace("_x", "").replace("_y", "")
            if f.name in ["x", "y", "pos_x", "pos_y"]:
                pos_field = f.name.replace("_x", "").replace("_y", "")

        if goal_field and pos_field:
            return f"""float calculate_reward({domain.state_struct_name}* state) {{
    float dx = state->{goal_field}_x - state->{pos_field}_x;
    float dy = state->{goal_field}_y - state->{pos_field}_y;
    float dist = sqrtf(dx*dx + dy*dy);

    float reward = -dist * 0.1f;
    if (dist < 2.0f) reward += 10.0f;

    return reward;
}}"""
        else:
            return f"""float calculate_reward({domain.state_struct_name}* state) {{
    return 1.0f;  // Recompensa base - necesita evolución
}}"""

    def critique_physics(
        self,
        domain: DomainSpec,
        physics_code: str,
        trajectory: List[Dict[str, float]]
    ) -> PhysicsCritique:
        """
        El LLM actúa como crítico de la física generada.

        Analiza si la física es realista y si la tarea es solucionable.
        """
        if self.use_mock or len(trajectory) < 10:
            return PhysicsCritique(
                is_realistic=True,
                is_solvable=True,
                issues=[],
                suggestions=[],
                confidence=0.5
            )

        # Preparar muestra de trayectoria
        sample = trajectory[-50:] if len(trajectory) > 50 else trajectory
        trajectory_str = "\n".join(
            f"  step {i}: x={s.get('x', 0):.2f}, y={s.get('y', 0):.2f}, v={s.get('v_linear', 0):.2f}"
            for i, s in enumerate(sample[-20:])
        )

        # Calcular métricas
        final = trajectory[-1] if trajectory else {}
        speeds = [abs(s.get('v_linear', 0)) for s in trajectory]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        # Calcular rangos
        value_ranges = {}
        for key in ['x', 'y', 'v_linear']:
            values = [s.get(key, 0) for s in trajectory]
            if values:
                value_ranges[key] = f"[{min(values):.2f}, {max(values):.2f}]"

        prompt = PHYSICS_CRITIQUE_PROMPT.format(
            domain_name=domain.name,
            description=domain.description,
            physics_code=physics_code,
            trajectory_sample=trajectory_str,
            final_x=final.get('x', 0),
            final_y=final.get('y', 0),
            avg_speed=avg_speed,
            value_ranges=json.dumps(value_ranges),
        )

        response = self._call_llm(prompt)

        try:
            critique_dict = self._extract_json(response)
            return PhysicsCritique(
                is_realistic=critique_dict.get("is_realistic", True),
                is_solvable=critique_dict.get("is_solvable", True),
                issues=critique_dict.get("issues", []),
                suggestions=critique_dict.get("suggestions", []),
                confidence=critique_dict.get("confidence", 0.5),
            )
        except:
            return PhysicsCritique(True, True, [], [], 0.5)

    def refine_domain(
        self,
        domain: DomainSpec,
        critique: PhysicsCritique
    ) -> DomainSpec:
        """Refina el dominio basándose en la crítica"""
        # Por ahora, retornar el mismo dominio
        # En una implementación completa, el LLM modificaría el DomainSpec
        return domain

    def _mock_domain(self, instruction: str) -> GeneratedDomain:
        """Genera un dominio mock basado en palabras clave"""
        instruction_lower = instruction.lower()

        if "pipa" in instruction_lower or "tanque" in instruction_lower or "agua" in instruction_lower:
            return self._mock_water_truck()
        elif "drone" in instruction_lower or "cuadricóptero" in instruction_lower:
            return self._mock_drone()
        elif "brazo" in instruction_lower or "arm" in instruction_lower:
            return self._mock_arm()
        else:
            return self._mock_generic_vehicle()

    def _mock_water_truck(self) -> GeneratedDomain:
        """Mock para pipa de agua"""
        domain = DomainSpec(
            name="WaterTruck",
            description="Camión cisterna con agua que debe transportarse de A a B sin derramar",

            state_fields=[
                StateField("x", description="Posición X", normalize_by=50.0),
                StateField("y", description="Posición Y", normalize_by=50.0),
                StateField("theta", description="Orientación", normalize_by=3.14159),
                StateField("v_linear", description="Velocidad lineal", normalize_by=10.0),
                StateField("v_angular", description="Velocidad angular", normalize_by=2.0),
                StateField("water_mass", description="Masa de agua (kg)", normalize_by=5000.0, default_value=4000.0),
                StateField("slosh_offset", description="Desplazamiento del agua", normalize_by=1.0),
                StateField("slosh_velocity", description="Velocidad del sloshing", normalize_by=2.0),
                StateField("target_x", description="Destino X", normalize_by=50.0, default_value=40.0),
                StateField("target_y", description="Destino Y", normalize_by=50.0, default_value=40.0),
            ],

            action_fields=[
                ActionField("throttle", -1.0, 1.0, "Acelerador/Freno"),
                ActionField("steering", -1.0, 1.0, "Dirección"),
            ],

            physics_constants=PhysicsConstants(
                dt=0.02,
                gravity=9.81,
                world_bounds=(50.0, 50.0, 0.0),
                max_episode_steps=1000,
                custom={
                    "TRUCK_MASS": 3000.0,
                    "MAX_WATER": 5000.0,
                    "MAX_SPEED": 8.0,
                    "MAX_STEERING": 0.5,
                    "SLOSH_DAMPING": 0.3,
                    "SLOSH_STIFFNESS": 2.0,
                }
            ),

            termination_conditions=[
                TerminationCondition("reached_goal",
                    "sqrtf((state->x-state->target_x)*(state->x-state->target_x)+(state->y-state->target_y)*(state->y-state->target_y)) < 2.0f",
                    "Llegó al destino"),
                TerminationCondition("rollover", "fabsf(state->slosh_offset) > 0.8f", "Volcó por sloshing"),
                TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
            ],

            physics_description="""
Camión cisterna con dinámica de fluidos simplificada:
- Modelo de vehículo Ackermann (throttle + steering)
- El agua dentro del tanque tiene masa y puede desplazarse (sloshing)
- Aceleraciones laterales causan movimiento del agua
- El desplazamiento del agua afecta el centro de masa y la estabilidad
- Si el sloshing es muy alto, el camión puede volcar
- La velocidad máxima depende de la carga de agua
""",

            reward_hints="""
- Premiar acercarse al destino
- Penalizar fuertemente el sloshing excesivo
- Premiar conducción suave (baja aceleración lateral)
- Penalizar frenar bruscamente
- Bonus grande al llegar sin derramar
"""
        )

        physics_code = """void physics_step(WaterTruckState* state, float* actions) {
    float throttle = actions[0];
    float steering = actions[1];

    // Masa total = camión + agua
    float total_mass = TRUCK_MASS + state->water_mass;
    float mass_ratio = state->water_mass / MAX_WATER;

    // Velocidad máxima reducida con carga
    float effective_max_speed = MAX_SPEED * (1.0f - 0.3f * mass_ratio);

    // Actualizar velocidad lineal
    float target_speed = throttle * effective_max_speed;
    float acceleration = (target_speed - state->v_linear) * 0.1f;
    state->v_linear += acceleration;
    state->v_linear *= 0.98f;  // Fricción

    // Steering (reducido con velocidad)
    float speed_factor = 1.0f - 0.5f * fabsf(state->v_linear) / MAX_SPEED;
    state->v_angular = steering * MAX_STEERING * speed_factor;

    // Aceleración lateral (causa sloshing)
    float lateral_accel = state->v_linear * state->v_angular;

    // Dinámica del sloshing (oscilador amortiguado)
    float slosh_force = -SLOSH_STIFFNESS * state->slosh_offset - SLOSH_DAMPING * state->slosh_velocity;
    slosh_force += lateral_accel * mass_ratio * 0.5f;  // El agua reacciona a la aceleración

    state->slosh_velocity += slosh_force * DT;
    state->slosh_offset += state->slosh_velocity * DT;

    // Limitar sloshing
    if (state->slosh_offset > 1.0f) { state->slosh_offset = 1.0f; state->slosh_velocity *= -0.5f; }
    if (state->slosh_offset < -1.0f) { state->slosh_offset = -1.0f; state->slosh_velocity *= -0.5f; }

    // Integrar posición y orientación
    state->theta += state->v_angular * DT;
    while (state->theta > 3.14159f) state->theta -= 6.28318f;
    while (state->theta < -3.14159f) state->theta += 6.28318f;

    state->x += state->v_linear * cosf(state->theta) * DT;
    state->y += state->v_linear * sinf(state->theta) * DT;

    // Límites del mundo
    if (state->x < 0.0f) state->x = 0.0f;
    if (state->x > WORLD_SIZE_X) state->x = WORLD_SIZE_X;
    if (state->y < 0.0f) state->y = 0.0f;
    if (state->y > WORLD_SIZE_Y) state->y = WORLD_SIZE_Y;

    state->steps++;
}"""

        reward_code = """float calculate_reward(WaterTruckState* state) {
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dist = sqrtf(dx*dx + dy*dy);

    float reward = -dist * 0.05f;

    // Penalizar sloshing
    reward -= fabsf(state->slosh_offset) * 2.0f;

    // Bonus por llegar
    if (dist < 2.0f) reward += 10.0f;

    return reward;
}"""

        return GeneratedDomain(domain, physics_code, reward_code, True)

    def _mock_physics(self, domain: DomainSpec) -> str:
        """Genera física mock genérica"""
        return f"""void physics_step({domain.state_struct_name}* state, float* actions) {{
    // Física genérica
    state->steps++;
}}"""

    def _mock_drone(self) -> GeneratedDomain:
        """Mock para drone - usar el existente"""
        from domain_spec import create_drone_domain
        domain = create_drone_domain()
        return GeneratedDomain(domain, "", "", True)

    def _mock_arm(self) -> GeneratedDomain:
        """Mock para brazo robótico - usar el existente"""
        from domain_spec import create_robotic_arm_domain
        domain = create_robotic_arm_domain()
        return GeneratedDomain(domain, "", "", True)

    def _mock_generic_vehicle(self) -> GeneratedDomain:
        """Mock para vehículo genérico"""
        from domain_spec import create_warehouse_robot_domain
        domain = create_warehouse_robot_domain()
        return GeneratedDomain(domain, "", "", True)


# ============================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================

def generate_from_text(instruction: str) -> GeneratedDomain:
    """
    Genera un dominio completo desde una instrucción en lenguaje natural.

    Ejemplo:
        result = generate_from_text("una pipa de agua que va de A a B")
        if result.success:
            print(result.domain.name)
            print(result.physics_code)
    """
    generator = DomainGenerator()
    return generator.generate_domain(instruction)


if __name__ == "__main__":
    print("=== Domain Generator Demo ===\n")

    # Probar con pipa de agua (mock)
    result = generate_from_text("una pipa de agua que transporta líquido de A a B")

    if result.success:
        print(f"Dominio: {result.domain.name}")
        print(f"Descripción: {result.domain.description}")
        print(f"Estado: {[f.name for f in result.domain.state_fields]}")
        print(f"Acciones: {[a.name for a in result.domain.action_fields]}")
        print()
        print("Física generada:")
        print(result.physics_code[:500] + "...")
    else:
        print(f"Error: {result.error_message}")
