"""
Domain Specification - Definición abstracta de cualquier sistema robótico

Este módulo permite definir CUALQUIER dominio (dron, brazo robótico, carro, péndulo)
de forma declarativa, sin hardcodear física ni estado.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import json


class FieldType(Enum):
    """Tipos de datos soportados para campos del estado"""
    FLOAT = "float"
    INT = "int"
    BOOL = "int"  # En C, bool es int


class ActionSpace(Enum):
    """Tipos de espacios de acción"""
    CONTINUOUS = "continuous"  # Box(-1, 1, shape=(n,))
    DISCRETE = "discrete"      # Discrete(n)


@dataclass
class StateField:
    """Define un campo del estado del sistema"""
    name: str
    field_type: FieldType = FieldType.FLOAT
    description: str = ""
    default_value: float = 0.0
    normalize_by: float = 1.0  # Para normalizar observaciones
    is_observable: bool = True  # Si se incluye en observaciones
    is_action_target: bool = False  # Si es modificado por acciones

    def to_c_declaration(self) -> str:
        """Genera declaración C para este campo"""
        return f"    {self.field_type.value} {self.name};"

    def to_c_comment(self) -> str:
        """Genera comentario descriptivo"""
        return f"    // {self.name}: {self.description}" if self.description else ""


@dataclass
class ActionField:
    """Define un actuador/acción del sistema"""
    name: str
    min_value: float = -1.0
    max_value: float = 1.0
    description: str = ""

    def to_c_conversion(self, idx: int) -> str:
        """Genera código C para convertir acción normalizada"""
        if self.min_value == -1.0 and self.max_value == 1.0:
            return f"    float {self.name} = actions[i * {{action_size}} + {idx}];"
        else:
            # Convertir de [-1, 1] a [min, max]
            return f"    float {self.name} = (actions[i * {{action_size}} + {idx}] + 1.0f) * 0.5f * {self.max_value - self.min_value}f + {self.min_value}f;"


@dataclass
class PhysicsConstants:
    """Constantes físicas del dominio"""
    dt: float = 0.02  # Paso de tiempo (50 Hz)
    gravity: float = 9.81
    world_bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0)  # X, Y, Z
    max_episode_steps: int = 500

    # Parámetros adicionales personalizables
    custom: Dict[str, float] = field(default_factory=dict)

    def to_c_defines(self) -> str:
        """Genera #define para constantes"""
        lines = [
            f"#define DT {self.dt}f",
            f"#define DEFAULT_GRAVITY {self.gravity}f",
            f"#define WORLD_SIZE_X {self.world_bounds[0]}f",
            f"#define WORLD_SIZE_Y {self.world_bounds[1]}f",
            f"#define WORLD_SIZE_Z {self.world_bounds[2]}f",
            f"#define MAX_EPISODE_STEPS {self.max_episode_steps}",
        ]
        for name, value in self.custom.items():
            lines.append(f"#define {name.upper()} {value}f")
        return "\n".join(lines)


@dataclass
class DomainRandomizationParam:
    """Parámetro de Domain Randomization"""
    name: str
    base_value: float
    min_value: float
    max_value: float
    description: str = ""


@dataclass
class TerminationCondition:
    """Condición de terminación de episodio"""
    name: str
    condition_code: str  # Código C que evalúa a true/false
    description: str = ""


@dataclass
class DomainSpec:
    """
    Especificación completa de un dominio robótico.

    Esta clase es el corazón de la Plataforma Universal.
    Define TODO lo necesario para simular cualquier sistema.
    """

    # Identificación
    name: str
    description: str

    # Estado del sistema
    state_fields: List[StateField]

    # Acciones/actuadores
    action_fields: List[ActionField]
    action_space: ActionSpace = ActionSpace.CONTINUOUS

    # Física
    physics_constants: PhysicsConstants = field(default_factory=PhysicsConstants)

    # Domain Randomization
    dr_params: List[DomainRandomizationParam] = field(default_factory=list)

    # Condiciones de terminación
    termination_conditions: List[TerminationCondition] = field(default_factory=list)

    # Metadatos para el LLM
    physics_description: str = ""  # Descripción en lenguaje natural de la física
    reward_hints: str = ""  # Pistas para diseñar recompensas

    @property
    def obs_size(self) -> int:
        """Número de observaciones"""
        return sum(1 for f in self.state_fields if f.is_observable)

    @property
    def action_size(self) -> int:
        """Número de acciones"""
        return len(self.action_fields)

    @property
    def state_struct_name(self) -> str:
        """Nombre del struct de estado en C"""
        # Preservar el nombre original, solo remover espacios
        clean_name = self.name.replace(' ', '')
        return f"{clean_name}State"

    def generate_state_struct(self) -> str:
        """Genera el typedef struct para el estado"""
        lines = [f"typedef struct {{"]

        # Agrupar campos por categoría (basado en comentarios del campo)
        for field in self.state_fields:
            if field.description:
                lines.append(f"    // {field.description}")
            lines.append(f"    {field.field_type.value} {field.name};")

        # Añadir campos de métricas estándar
        lines.extend([
            "",
            "    // === Métricas estándar ===",
            "    int steps;",
            "    float total_reward;",
        ])

        # Añadir campos de DR si hay
        if self.dr_params:
            lines.extend([
                "",
                "    // === Domain Randomization ===",
            ])
            for dr in self.dr_params:
                lines.append(f"    float {dr.name};  // {dr.description}")

        lines.append(f"}} {self.state_struct_name};")
        return "\n".join(lines)

    def generate_observation_code(self) -> str:
        """Genera código C para extraer observaciones"""
        lines = []
        obs_idx = 0
        for field in self.state_fields:
            if field.is_observable:
                if field.normalize_by != 1.0:
                    lines.append(f"        obs[{obs_idx}] = state->{field.name} / {field.normalize_by}f;")
                else:
                    lines.append(f"        obs[{obs_idx}] = state->{field.name};")
                obs_idx += 1
        return "\n".join(lines)

    def generate_action_conversion(self) -> str:
        """Genera código C para convertir acciones normalizadas"""
        lines = []
        for idx, action in enumerate(self.action_fields):
            code = action.to_c_conversion(idx)
            lines.append(code.format(action_size=self.action_size))
        return "\n".join(lines)

    def generate_termination_code(self) -> str:
        """Genera código C para verificar terminación"""
        if not self.termination_conditions:
            return "        int done = (state->steps >= MAX_EPISODE_STEPS) ? 1 : 0;"

        lines = ["        int done = 0;"]
        for cond in self.termination_conditions:
            lines.append(f"        // {cond.description}")
            lines.append(f"        if ({cond.condition_code}) done = 1;")
        return "\n".join(lines)

    def generate_dr_struct(self) -> str:
        """Genera struct de configuración de Domain Randomization"""
        if not self.dr_params:
            return ""

        lines = ["typedef struct {"]
        for dr in self.dr_params:
            lines.append(f"    float {dr.name}_min;")
            lines.append(f"    float {dr.name}_max;")
        lines.append("    int enabled;")
        lines.append("} DomainRandomConfig;")
        return "\n".join(lines)

    def generate_dr_apply_code(self) -> str:
        """Genera código C para aplicar Domain Randomization"""
        if not self.dr_params:
            return ""

        lines = [
            "void apply_domain_randomization(" + self.state_struct_name + "* state, DomainRandomConfig* config) {",
            "    if (!config->enabled) {"
        ]

        # Valores por defecto
        for dr in self.dr_params:
            lines.append(f"        state->{dr.name} = {dr.base_value}f;")
        lines.append("        return;")
        lines.append("    }")
        lines.append("")

        # Valores aleatorizados
        for dr in self.dr_params:
            lines.append(f"    state->{dr.name} = rand_range(config->{dr.name}_min, config->{dr.name}_max);")

        lines.append("}")
        return "\n".join(lines)

    def to_architect_prompt(self) -> str:
        """Genera el system prompt para el Arquitecto Universal"""
        return f"""Eres un ingeniero de simulación robótica experto en Reinforcement Learning.
Tu trabajo es escribir código C de alto rendimiento para el sistema: {self.name}

DESCRIPCIÓN DEL SISTEMA:
{self.description}

ESTRUCTURA DE ESTADO DISPONIBLE:
```c
{self.generate_state_struct()}
```

DESCRIPCIÓN DE LA FÍSICA:
{self.physics_description}

CONSTANTES DISPONIBLES:
```c
{self.physics_constants.to_c_defines()}
```

ACCIONES DISPONIBLES ({self.action_size} actuadores):
{chr(10).join(f"- {a.name}: [{a.min_value}, {a.max_value}] - {a.description}" for a in self.action_fields)}

OBSERVACIONES ({self.obs_size} valores):
{chr(10).join(f"- {f.name} (normalizado por {f.normalize_by})" for f in self.state_fields if f.is_observable)}

PISTAS PARA DISEÑO DE RECOMPENSAS:
{self.reward_hints}

REGLAS IMPORTANTES:
1. SOLO usa math.h (fabsf, sqrtf, sinf, cosf, etc.)
2. NO uses printf, malloc, ni funciones de sistema
3. Retorna valores float en rango razonable (-10 a +10)
4. Usa state-> para acceder a los campos
5. El código se ejecutará MILLONES de veces - debe ser eficiente
"""

    def to_dict(self) -> dict:
        """Serializa a diccionario para guardar/cargar"""
        return {
            "name": self.name,
            "description": self.description,
            "state_fields": [
                {
                    "name": f.name,
                    "type": f.field_type.value,
                    "description": f.description,
                    "default": f.default_value,
                    "normalize_by": f.normalize_by,
                    "observable": f.is_observable,
                }
                for f in self.state_fields
            ],
            "action_fields": [
                {
                    "name": a.name,
                    "min": a.min_value,
                    "max": a.max_value,
                    "description": a.description,
                }
                for a in self.action_fields
            ],
            "physics": {
                "dt": self.physics_constants.dt,
                "gravity": self.physics_constants.gravity,
                "world_bounds": self.physics_constants.world_bounds,
                "max_steps": self.physics_constants.max_episode_steps,
                "custom": self.physics_constants.custom,
            },
            "physics_description": self.physics_description,
            "reward_hints": self.reward_hints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainSpec":
        """Deserializa desde diccionario"""
        state_fields = [
            StateField(
                name=f["name"],
                field_type=FieldType(f.get("type", "float")),
                description=f.get("description", ""),
                default_value=f.get("default", 0.0),
                normalize_by=f.get("normalize_by", 1.0),
                is_observable=f.get("observable", True),
            )
            for f in data["state_fields"]
        ]

        action_fields = [
            ActionField(
                name=a["name"],
                min_value=a.get("min", -1.0),
                max_value=a.get("max", 1.0),
                description=a.get("description", ""),
            )
            for a in data["action_fields"]
        ]

        physics = data.get("physics", {})
        physics_constants = PhysicsConstants(
            dt=physics.get("dt", 0.02),
            gravity=physics.get("gravity", 9.81),
            world_bounds=tuple(physics.get("world_bounds", [10.0, 10.0, 5.0])),
            max_episode_steps=physics.get("max_steps", 500),
            custom=physics.get("custom", {}),
        )

        return cls(
            name=data["name"],
            description=data["description"],
            state_fields=state_fields,
            action_fields=action_fields,
            physics_constants=physics_constants,
            physics_description=data.get("physics_description", ""),
            reward_hints=data.get("reward_hints", ""),
        )

    def save(self, filepath: str):
        """Guarda especificación a archivo JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DomainSpec":
        """Carga especificación desde archivo JSON"""
        with open(filepath) as f:
            return cls.from_dict(json.load(f))


# ============================================================
# DOMINIOS PREDEFINIDOS
# ============================================================

def create_drone_domain() -> DomainSpec:
    """Crea especificación para el dominio de drones (compatibilidad con sistema actual)"""
    return DomainSpec(
        name="Drone",
        description="Quadcopter volador con 4 motores. Control de empuje y torques.",

        state_fields=[
            # Posición
            StateField("x", description="Posición X (metros)", normalize_by=10.0),
            StateField("y", description="Posición Y (metros)", normalize_by=10.0),
            StateField("z", description="Posición Z/altura (metros)", normalize_by=5.0),
            # Velocidad
            StateField("vx", description="Velocidad en X (m/s)", normalize_by=5.0),
            StateField("vy", description="Velocidad en Y (m/s)", normalize_by=5.0),
            StateField("vz", description="Velocidad en Z (m/s)", normalize_by=5.0),
            # Orientación
            StateField("roll", description="Rotación roll (radianes)"),
            StateField("pitch", description="Rotación pitch (radianes)"),
            StateField("yaw", description="Rotación yaw (radianes)"),
            # Velocidades angulares
            StateField("roll_rate", description="Velocidad angular roll (rad/s)"),
            StateField("pitch_rate", description="Velocidad angular pitch (rad/s)"),
            StateField("yaw_rate", description="Velocidad angular yaw (rad/s)"),
            # Objetivo
            StateField("target_x", description="Objetivo X", normalize_by=10.0),
            StateField("target_y", description="Objetivo Y", normalize_by=10.0),
            StateField("target_z", description="Objetivo Z", normalize_by=5.0),
            # Métricas específicas
            StateField("collisions", FieldType.INT, "Número de colisiones", is_observable=False),
        ],

        action_fields=[
            ActionField("thrust", 0.0, 20.0, "Empuje vertical (Newtons)"),
            ActionField("roll_cmd", -5.0, 5.0, "Torque de roll (Nm)"),
            ActionField("pitch_cmd", -5.0, 5.0, "Torque de pitch (Nm)"),
            ActionField("yaw_cmd", -5.0, 5.0, "Torque de yaw (Nm)"),
        ],

        physics_constants=PhysicsConstants(
            dt=0.02,
            gravity=9.81,
            world_bounds=(10.0, 10.0, 5.0),
            max_episode_steps=500,
            custom={"MAX_THRUST": 20.0, "MAX_TORQUE": 5.0}
        ),

        dr_params=[
            DomainRandomizationParam("mass", 1.0, 0.8, 1.2, "Masa del dron (kg)"),
            DomainRandomizationParam("drag_coeff", 0.1, 0.05, 0.2, "Coeficiente de arrastre"),
            DomainRandomizationParam("wind_x", 0.0, -1.0, 1.0, "Viento en X"),
            DomainRandomizationParam("wind_y", 0.0, -1.0, 1.0, "Viento en Y"),
            DomainRandomizationParam("wind_z", 0.0, -0.5, 0.5, "Viento en Z"),
            DomainRandomizationParam("gravity", 9.81, 9.32, 10.30, "Gravedad"),
            DomainRandomizationParam("motor_noise", 0.0, 0.0, 0.1, "Ruido de motores"),
        ],

        termination_conditions=[
            TerminationCondition("collision", "state->collisions > 0 && state->steps > 10", "Colisión con suelo"),
            TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
            TerminationCondition("out_of_bounds", "fabsf(state->x) >= WORLD_SIZE_X || fabsf(state->y) >= WORLD_SIZE_Y", "Fuera del área"),
        ],

        physics_description="""
El dron es un quadcopter simplificado con las siguientes características:
- El empuje (thrust) se aplica en la dirección Z del dron, transformado al frame mundo según roll/pitch
- Los torques controlan directamente las velocidades angulares con damping
- La gravedad actúa hacia abajo (-Z)
- Hay arrastre aerodinámico proporcional a la velocidad
- Colisión con suelo cuando z < 0
""",

        reward_hints="""
Para tareas de vuelo típicas:
- Premiar cercanía al objetivo usando distancia inversa o exponencial
- Penalizar orientaciones extremas (roll, pitch > 0.5 rad)
- Penalizar velocidades angulares altas para vuelo suave
- Bonus grande (+5 a +10) al llegar muy cerca del objetivo (<0.2m)
- Penalizar colisiones fuertemente (-5 a -10)
"""
    )


def create_cartpole_domain() -> DomainSpec:
    """Crea especificación para CartPole (péndulo invertido clásico)"""
    return DomainSpec(
        name="CartPole",
        description="Carro con péndulo invertido. El objetivo es balancear el palo vertical.",

        state_fields=[
            StateField("cart_position", description="Posición del carro", normalize_by=2.4),
            StateField("cart_velocity", description="Velocidad del carro", normalize_by=5.0),
            StateField("pole_angle", description="Ángulo del palo (radianes)", normalize_by=0.418),
            StateField("pole_velocity", description="Velocidad angular del palo", normalize_by=5.0),
            # Objetivo implícito: mantener pole_angle cerca de 0
        ],

        action_fields=[
            ActionField("force", -10.0, 10.0, "Fuerza horizontal aplicada al carro"),
        ],

        physics_constants=PhysicsConstants(
            dt=0.02,
            gravity=9.81,
            world_bounds=(2.4, 0.0, 0.0),  # Solo límite en X
            max_episode_steps=500,
            custom={
                "CART_MASS": 1.0,
                "POLE_MASS": 0.1,
                "POLE_LENGTH": 0.5,
                "FORCE_MAX": 10.0,
            }
        ),

        dr_params=[
            DomainRandomizationParam("cart_mass", 1.0, 0.8, 1.2, "Masa del carro"),
            DomainRandomizationParam("pole_mass", 0.1, 0.08, 0.12, "Masa del palo"),
            DomainRandomizationParam("pole_length", 0.5, 0.4, 0.6, "Longitud del palo"),
        ],

        termination_conditions=[
            TerminationCondition("pole_fell", "fabsf(state->pole_angle) > 0.418f", "Palo cayó (>24°)"),
            TerminationCondition("out_of_bounds", "fabsf(state->cart_position) > 2.4f", "Carro fuera de límites"),
            TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
        ],

        physics_description="""
Sistema de péndulo invertido sobre carro:
- El carro se mueve horizontalmente en un riel
- El palo está articulado en el centro del carro
- Se aplica fuerza horizontal al carro
- Ecuaciones de movimiento del péndulo invertido con fricción
- El palo empieza cerca de vertical (ángulo pequeño aleatorio)
""",

        reward_hints="""
Para balanceo del palo:
- Recompensa +1 por cada paso que sobrevive
- O recompensa basada en cercanía del ángulo a 0
- Penalizar velocidades extremas del carro
- Penalizar estar cerca de los bordes
"""
    )


def create_robotic_arm_domain() -> DomainSpec:
    """Crea especificación para brazo robótico de 2 joints"""
    return DomainSpec(
        name="RoboticArm2D",
        description="Brazo robótico planar de 2 articulaciones para tareas de alcance.",

        state_fields=[
            # Ángulos de joints
            StateField("joint1_angle", description="Ángulo del primer joint (rad)", normalize_by=3.14159),
            StateField("joint2_angle", description="Ángulo del segundo joint (rad)", normalize_by=3.14159),
            # Velocidades de joints
            StateField("joint1_velocity", description="Velocidad angular joint 1 (rad/s)", normalize_by=5.0),
            StateField("joint2_velocity", description="Velocidad angular joint 2 (rad/s)", normalize_by=5.0),
            # Posición del end effector (calculada)
            StateField("end_effector_x", description="Posición X del gripper", normalize_by=2.0),
            StateField("end_effector_y", description="Posición Y del gripper", normalize_by=2.0),
            # Objetivo
            StateField("target_x", description="Objetivo X", normalize_by=2.0),
            StateField("target_y", description="Objetivo Y", normalize_by=2.0),
        ],

        action_fields=[
            ActionField("torque1", -1.0, 1.0, "Torque aplicado al joint 1"),
            ActionField("torque2", -1.0, 1.0, "Torque aplicado al joint 2"),
        ],

        physics_constants=PhysicsConstants(
            dt=0.02,
            gravity=0.0,  # Brazo en plano horizontal, sin gravedad
            world_bounds=(2.0, 2.0, 0.0),
            max_episode_steps=200,
            custom={
                "LINK1_LENGTH": 1.0,
                "LINK2_LENGTH": 1.0,
                "MAX_TORQUE": 1.0,
            }
        ),

        termination_conditions=[
            TerminationCondition("reached_target",
                "sqrtf((state->end_effector_x - state->target_x)*(state->end_effector_x - state->target_x) + (state->end_effector_y - state->target_y)*(state->end_effector_y - state->target_y)) < 0.05f",
                "Objetivo alcanzado"),
            TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
        ],

        physics_description="""
Brazo robótico planar de 2 eslabones:
- Dos joints rotacionales en serie
- Longitud de cada eslabón: 1.0 unidad
- Torques aplicados directamente a cada joint
- Cinemática directa:
  end_x = L1*cos(θ1) + L2*cos(θ1+θ2)
  end_y = L1*sin(θ1) + L2*sin(θ1+θ2)
- Sin gravedad (brazo horizontal) o con compensación
""",

        reward_hints="""
Para tareas de alcance:
- Premiar cercanía del end effector al objetivo
- Penalizar uso excesivo de torque (eficiencia energética)
- Penalizar velocidades altas de joints (movimiento suave)
- Bonus grande al alcanzar objetivo
"""
    )


def create_warehouse_robot_domain() -> DomainSpec:
    """Crea especificación para robot de warehouse (2D con ruedas diferenciales)"""
    return DomainSpec(
        name="WarehouseRobot",
        description="Robot móvil con tracción diferencial para navegación en almacén.",

        state_fields=[
            # Posición
            StateField("x", description="Posición X (metros)", normalize_by=20.0),
            StateField("y", description="Posición Y (metros)", normalize_by=20.0),
            StateField("theta", description="Orientación (radianes)", normalize_by=3.14159),
            # Velocidad
            StateField("v_linear", description="Velocidad lineal (m/s)", normalize_by=2.0),
            StateField("v_angular", description="Velocidad angular (rad/s)", normalize_by=2.0),
            # Objetivo
            StateField("target_x", description="Objetivo X", normalize_by=20.0),
            StateField("target_y", description="Objetivo Y", normalize_by=20.0),
            # Sensores de obstáculos (simplificados)
            StateField("obstacle_front", description="Distancia a obstáculo frontal", normalize_by=5.0),
            StateField("obstacle_left", description="Distancia a obstáculo izquierdo", normalize_by=5.0),
            StateField("obstacle_right", description="Distancia a obstáculo derecho", normalize_by=5.0),
        ],

        action_fields=[
            ActionField("wheel_left", -1.0, 1.0, "Velocidad rueda izquierda"),
            ActionField("wheel_right", -1.0, 1.0, "Velocidad rueda derecha"),
        ],

        physics_constants=PhysicsConstants(
            dt=0.02,
            gravity=0.0,  # 2D top-down
            world_bounds=(20.0, 20.0, 0.0),
            max_episode_steps=1000,
            custom={
                "WHEEL_RADIUS": 0.1,
                "WHEEL_BASE": 0.5,
                "MAX_SPEED": 2.0,
            }
        ),

        termination_conditions=[
            TerminationCondition("reached_target",
                "sqrtf((state->x - state->target_x)*(state->x - state->target_x) + (state->y - state->target_y)*(state->y - state->target_y)) < 0.3f",
                "Objetivo alcanzado"),
            TerminationCondition("collision", "state->obstacle_front < 0.1f", "Colisión con obstáculo"),
            TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
        ],

        physics_description="""
Robot móvil con tracción diferencial:
- Dos ruedas independientes controlan velocidad lineal y angular
- v_linear = (v_left + v_right) * wheel_radius / 2
- v_angular = (v_right - v_left) * wheel_radius / wheel_base
- Movimiento en 2D con orientación theta
- Sensores de proximidad frontales y laterales
""",

        reward_hints="""
Para navegación en warehouse:
- Premiar acercarse al objetivo
- Premiar mantener velocidad moderada (no muy rápido, no muy lento)
- Penalizar acercarse a obstáculos
- Penalizar giros bruscos
- Bonus grande al llegar al destino
- Penalizar fuertemente colisiones
"""
    )


def create_pallet_jack_domain() -> DomainSpec:
    """Crea especificación para Electric Pallet Jack en almacén"""
    return DomainSpec(
        name="PalletJack",
        description="Transpaleta eléctrica para mover pallets del punto A al punto B en un almacén.",

        state_fields=[
            # Posición del pallet jack
            StateField("x", description="Posición X (metros)", normalize_by=20.0),
            StateField("y", description="Posición Y (metros)", normalize_by=20.0),
            StateField("theta", description="Orientación (radianes)", normalize_by=3.14159),
            # Velocidades
            StateField("v_linear", description="Velocidad lineal (m/s)", normalize_by=1.5),
            StateField("v_angular", description="Velocidad angular (rad/s)", normalize_by=1.0),
            # Punto de destino (B)
            StateField("target_x", description="Destino X", normalize_by=20.0, default_value=15.0),
            StateField("target_y", description="Destino Y", normalize_by=20.0, default_value=15.0),
            # Estado de carga
            StateField("has_pallet", FieldType.INT, description="1 si lleva pallet, 0 si no", default_value=1.0),
            # Sensores de obstáculos (estanterías)
            StateField("obstacle_front", description="Distancia obstáculo frontal", normalize_by=5.0, default_value=5.0),
            StateField("obstacle_left", description="Distancia obstáculo izq", normalize_by=5.0, default_value=5.0),
            StateField("obstacle_right", description="Distancia obstáculo der", normalize_by=5.0, default_value=5.0),
            # Batería
            StateField("battery", description="Nivel de batería (0-1)", normalize_by=1.0, default_value=1.0),
        ],

        action_fields=[
            ActionField("throttle", -1.0, 1.0, "Acelerador: -1=reversa, +1=adelante"),
            ActionField("steering", -1.0, 1.0, "Dirección: -1=izquierda, +1=derecha"),
        ],

        physics_constants=PhysicsConstants(
            dt=0.05,  # 20 Hz
            gravity=0.0,  # 2D top-down
            world_bounds=(20.0, 20.0, 0.0),
            max_episode_steps=500,
            custom={
                "MAX_SPEED": 1.5,
                "MAX_TURN_RATE": 0.8,
                "ACCELERATION": 0.5,
                "FRICTION": 0.1,
                "PALLET_WEIGHT_FACTOR": 0.7,
            }
        ),

        termination_conditions=[
            TerminationCondition("reached_target",
                "sqrtf((state->x - state->target_x)*(state->x - state->target_x) + (state->y - state->target_y)*(state->y - state->target_y)) < 0.5f",
                "Destino alcanzado"),
            TerminationCondition("collision", "state->obstacle_front < 0.2f", "Colisión"),
            TerminationCondition("timeout", "state->steps >= MAX_EPISODE_STEPS", "Tiempo agotado"),
            TerminationCondition("out_of_bounds", "state->x < 0 || state->x > WORLD_SIZE_X || state->y < 0 || state->y > WORLD_SIZE_Y", "Fuera de límites"),
        ],

        physics_description="""
Electric Pallet Jack (Transpaleta Eléctrica):
- Vehículo de almacén para transportar pallets
- Control: throttle (acelerar/frenar) y steering (girar)
- Modelo Ackermann simplificado para giros
- Velocidad máxima reducida cuando lleva carga (has_pallet=1)
- Inercia y fricción realistas
- Sensores de proximidad para detectar estanterías
- Batería que se consume con el uso
""",

        reward_hints="""
Para navegación de punto A a punto B:
- Premiar acercarse al destino (target_x, target_y)
- Penalizar colisiones con estanterías (obstacle_front < 0.3)
- Premiar velocidad moderada (no demasiado lento)
- Penalizar giros bruscos (suavidad)
- Bonus grande al llegar al destino
- Penalizar consumo excesivo de batería
"""
    )


# Catálogo de dominios predefinidos
DOMAIN_CATALOG = {
    "drone": create_drone_domain,
    "cartpole": create_cartpole_domain,
    "robotic_arm": create_robotic_arm_domain,
    "warehouse_robot": create_warehouse_robot_domain,
    "pallet_jack": create_pallet_jack_domain,
}


def get_domain(name: str) -> DomainSpec:
    """Obtiene un dominio predefinido por nombre"""
    if name not in DOMAIN_CATALOG:
        raise ValueError(f"Dominio '{name}' no encontrado. Disponibles: {list(DOMAIN_CATALOG.keys())}")
    return DOMAIN_CATALOG[name]()


def list_domains() -> List[str]:
    """Lista todos los dominios disponibles"""
    return list(DOMAIN_CATALOG.keys())
