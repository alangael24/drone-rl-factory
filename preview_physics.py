#!/usr/bin/env python3
"""
Preview Physics - Candado Visual

Antes de entrenar por horas, ejecuta 5 segundos de simulación
para que el ojo humano detecte alucinaciones (robots volando, teletransporte, etc).

El 99% de los errores de física son obvios en los primeros 5 segundos.
"""

import numpy as np
import sys
from typing import Optional, Tuple

from domain_spec import DomainSpec, get_domain
from universal_architect import UniversalArchitect
from universal_compiler import UniversalCompiler, CompileResult
from universal_env import UniversalVecEnv


def preview_physics(
    domain: DomainSpec,
    physics_code: str,
    reward_code: str,
    verify_code: str = "",
    duration_seconds: float = 5.0,
    fps: int = 10
) -> Tuple[bool, str]:
    """
    Ejecuta una vista previa visual de 5 segundos.

    Simula el robot con acciones aleatorias y muestra una visualización
    simple para que el humano detecte errores obvios.

    Args:
        domain: Especificación del dominio
        physics_code: Código de física
        reward_code: Código de recompensa
        verify_code: Código de verificación
        duration_seconds: Duración de la vista previa
        fps: Fotogramas por segundo (para mostrar progreso)

    Returns:
        (aprueba, diagnosis)
        - aprueba: True si se ve coherente
        - diagnosis: Mensaje explicativo
    """

    print(f"\n{'='*60}")
    print(f"VISTA PREVIA DE FÍSICA - {domain.name}")
    print(f"{'='*60}")
    print(f"\nEjecutando {duration_seconds}s de simulación...")
    print("(Se mostrarán los primeros 100 pasos de 10 en 10)")
    print()

    # Compilar
    print("Compilando...")
    compiler = UniversalCompiler()
    output = compiler.compile(
        domain, physics_code, reward_code, verify_code,
        output_name=f"lib{domain.name.lower()}_preview"
    )

    if output.result != CompileResult.SUCCESS:
        return False, f"Compilación fallida: {output.error_message[:200]}"

    # Crear entorno pequeño
    print("Creando entorno...")
    env = UniversalVecEnv(1, domain, output.library_path)

    # Simular
    steps = int(duration_seconds * fps)
    obs = env.reset()

    print(f"\nSimulando {steps} pasos (≈{duration_seconds}s)...\n")

    state_history = []
    reward_history = []
    error_detected = False
    error_message = ""

    for step in range(steps):
        # Acciones aleatorias
        actions = np.random.uniform(-1, 1, (1, domain.action_size)).astype(np.float32)

        # Ejecutar paso
        obs, rewards, dones, _, _ = env.step(actions)

        state_history.append(obs[0].copy())
        reward_history.append(rewards[0])

        # Mostrar progreso cada 10 pasos
        if step % 10 == 0:
            _print_frame(domain, obs[0], rewards[0], step)

        # Detectar anomalías
        if _has_anomaly(domain, obs[0]):
            error_detected = True
            error_message = f"Anomalía detectada en paso {step}: valores incoherentes"
            break

    env.close()

    # Analizar
    print(f"\n{'='*60}")
    print("ANÁLISIS DE LA SIMULACIÓN")
    print(f"{'='*60}\n")

    state_history = np.array(state_history)
    reward_history = np.array(reward_history)

    # Chequeos simples
    issues = []

    # 1. ¿Hay NaN o infinitos?
    if np.any(np.isnan(state_history)) or np.any(np.isinf(state_history)):
        issues.append("❌ NaN o infinitos detectados en el estado")

    # 2. ¿Explotan las recompensas?
    if np.any(np.isnan(reward_history)) or np.any(np.isinf(reward_history)):
        issues.append("❌ Recompensa explota (NaN/Inf)")

    if np.max(np.abs(reward_history)) > 1000:
        issues.append(f"⚠️  Recompensa muy grande (max={np.max(np.abs(reward_history)):.1f})")

    # 3. ¿El robot se mueve?
    movement = np.max(np.abs(state_history[-1] - state_history[0]))
    if movement < 0.01:
        issues.append(f"⚠️  Robot no se mueve (movimiento total: {movement:.6f})")

    # 4. Chequeos específicos por dominio
    domain_issues = _check_domain_specific(domain, state_history)
    issues.extend(domain_issues)

    # Resultado
    print(f"Pasos ejecutados: {len(state_history)}")
    print(f"Movimiento total: {movement:.4f} unidades")
    print(f"Recompensa media: {np.mean(reward_history):.4f}")
    print(f"Recompensa desv.est: {np.std(reward_history):.4f}")

    if issues:
        print(f"\n⚠️  PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   {issue}")
        return False, "Anomalías detectadas en vista previa"
    else:
        print(f"\n✅ LA FÍSICA SE VE COHERENTE")
        print(f"   - Robot se mueve correctamente")
        print(f"   - Recompensas dentro de rango razonable")
        print(f"   - No hay NaN/infinitos")
        return True, "Vista previa OK"


def _print_frame(domain: DomainSpec, obs: np.ndarray, reward: float, step: int):
    """Muestra un frame simple de la simulación"""
    obs_str = ", ".join(f"{x:.2f}" for x in obs[:min(4, len(obs))])
    if len(obs) > 4:
        obs_str += ", ..."

    print(f"  Paso {step:3d} | Obs: [{obs_str}] | Reward: {reward:+.3f}")


def _has_anomaly(domain: DomainSpec, obs: np.ndarray) -> bool:
    """Detecta anomalías obvias"""
    # NaN o infinitos
    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        return True

    # Valores extremadamente grandes
    if np.any(np.abs(obs) > 1e6):
        return True

    return False


def _check_domain_specific(domain: DomainSpec, state_history: np.ndarray) -> list:
    """Chequeos específicos por dominio"""
    issues = []

    if domain.name == "Drone":
        # Z debe ser positivo (no bajo tierra)
        z_min = np.min(state_history[:, 2])  # z es índice 2
        if z_min < -0.1:
            issues.append(f"❌ Drone vuela bajo tierra (z_min={z_min:.3f})")

        # Z debe ser razonable
        z_max = np.max(state_history[:, 2])
        if z_max > 200:
            issues.append(f"⚠️  Drone vuela muy alto (z_max={z_max:.1f}m)")

    elif domain.name == "CartPole":
        # Ángulo debe estar en [-pi, pi]
        angle = state_history[:, 2]  # pole_angle
        if np.any(np.abs(angle) > 4):
            issues.append(f"⚠️  Ángulo del palo fuera de rango (max={np.max(np.abs(angle)):.2f})")

    elif domain.name == "WarehouseRobot":
        # Posición dentro del espacio
        x = state_history[:, 0]
        y = state_history[:, 1]
        if np.any((x < -1) | (x > 11)) or np.any((y < -1) | (y > 11)):
            issues.append(f"⚠️  Robot salió de los límites del almacén")

    return issues


def interactive_preview():
    """Modo interactivo de vista previa"""
    from domain_spec import list_domains

    print("\n" + "="*60)
    print("VISTA PREVIA DE FÍSICA - MODO INTERACTIVO")
    print("="*60)

    print("\nDominios disponibles:")
    for i, name in enumerate(list_domains(), 1):
        print(f"  {i}. {name}")

    choice = input("\nSelecciona dominio (número): ").strip()
    domain_names = list_domains()

    try:
        domain = get_domain(domain_names[int(choice) - 1])
    except (ValueError, IndexError):
        print("Opción inválida")
        return

    # Generar código mock
    architect = UniversalArchitect(use_mock=True)
    physics = architect.generate_physics(domain)
    reward = architect.generate_reward(domain, "objetivo test")
    verify = architect.generate_verification(domain, physics.physics_code)

    # Vista previa
    approved, diagnosis = preview_physics(
        domain,
        physics.physics_code,
        reward.reward_code,
        verify
    )

    print(f"\nDiagnóstico: {diagnosis}")

    if approved:
        confirm = input("\n¿Aprobar para entrenar? (S/N): ").strip().lower()
        if confirm == 's':
            print("✅ Aprobado para entrenar")
        else:
            print("❌ Cancelado")
    else:
        print("❌ No aprobado para entrenar")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_preview()
    else:
        print(__doc__)
