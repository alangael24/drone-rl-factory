# Universal Robotics Platform

Plataforma de Inteligencia General para entrenar CUALQUIER sistema robótico usando RL + LLM.

## Arquitectura

```
Usuario: "Robot de warehouse que navega"
              │
              ▼
┌─────────────────────────────────┐
│  DomainSpec                     │  Define estado, acciones, física
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  UniversalArchitect (LLM)       │  Genera physics_step() + calculate_reward()
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  UniversalCompiler (GCC)        │  Compila C → .so (~2M steps/sec)
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  UniversalEnv (ctypes)          │  Python ↔ C zero-copy
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  PPO Training                   │  Entrenamiento vectorizado
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  UniversalJudge                 │  Métricas agnósticas de aprendizaje
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  UniversalEvolution (Eureka)    │  Evolución de recompensas
└─────────────────────────────────┘
```

## Dominios Disponibles

| Dominio | Observaciones | Acciones | Descripción |
|---------|---------------|----------|-------------|
| `drone` | 15 | 4 | Quadcopter volador |
| `cartpole` | 4 | 1 | Péndulo invertido |
| `robotic_arm` | 8 | 2 | Brazo de 2 joints |
| `warehouse_robot` | 10 | 2 | Robot diferencial |

## Uso

```bash
# Entrenamiento rápido
python universal_platform.py --domain cartpole --task "balancear palo"

# Con evolución Eureka
python universal_platform.py --domain drone --task "hover a 1m" --evolve

# Modo interactivo
python universal_platform.py --interactive

# Ver dominios
python universal_platform.py --list-domains
```

## Archivos

- `domain_spec.py` - Definición declarativa de dominios robóticos
- `universal_architect.py` - Generación de física y recompensa con LLM
- `universal_compiler.py` - Compilación dinámica de código C
- `universal_env.py` - Wrapper de entornos para cualquier dominio
- `universal_judge.py` - Evaluación agnóstica de aprendizaje
- `universal_evolution.py` - Bucle evolutivo Eureka
- `universal_platform.py` - CLI unificada

## Añadir Nuevo Dominio

```python
# En domain_spec.py
def create_my_robot_domain() -> DomainSpec:
    return DomainSpec(
        name="MyRobot",
        description="Mi robot personalizado",
        state_fields=[
            StateField("x", description="Posición X"),
            StateField("velocity", description="Velocidad"),
        ],
        action_fields=[
            ActionField("motor", -1.0, 1.0, "Control de motor"),
        ],
        physics_description="Descripción para el LLM",
        reward_hints="Pistas para diseñar recompensas",
    )

# Registrar en catálogo
DOMAIN_CATALOG["my_robot"] = create_my_robot_domain
```

## Requisitos

```bash
pip install torch numpy gymnasium
```

## Licencia

MIT
