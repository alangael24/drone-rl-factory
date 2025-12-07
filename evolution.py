#!/usr/bin/env python3
"""
Sprint 3: Búsqueda Evolutiva de Funciones de Recompensa

Implementa el algoritmo evolutivo de Eureka/RoboGen:
1. Genera una población de funciones de recompensa candidatas
2. Evalúa cada candidato con entrenamiento rápido
3. Selecciona los mejores y muta/refina con LLM
4. Repite hasta convergencia

Basado en:
- Eureka: https://arxiv.org/abs/2310.12931
- RoboGen: https://arxiv.org/abs/2311.01455
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch

from architect import (
    generate_reward_function,
    generate_reward_mock,
    evolve_reward_function,
    validate_c_syntax,
)
from compiler import compile_reward_function, test_compiled_library, CompileResult
from drone_env import DroneVecEnv
from train import DronePolicy, PPOTrainer


@dataclass
class EvaluationResult:
    """Resultado de evaluar una función de recompensa."""
    code: str
    mean_reward: float
    max_reward: float
    min_reward: float
    mean_steps: float
    mean_collisions: float
    final_distance: float
    mean_speed: float
    mean_orientation: float
    compile_success: bool
    training_time: float
    generation: int


def evaluate_reward_function(
    reward_code: str,
    num_envs: int = 32,
    eval_steps: int = 100_000,
    generation: int = 0
) -> Optional[EvaluationResult]:
    """
    Evalúa una función de recompensa compilándola y entrenando brevemente.

    Args:
        reward_code: Código C de la función de recompensa
        num_envs: Número de entornos paralelos
        eval_steps: Pasos de entrenamiento para evaluación rápida
        generation: Número de generación (para tracking)

    Returns:
        EvaluationResult con métricas o None si falla
    """
    start_time = time.time()

    # 1. Validar sintaxis
    valid, msg = validate_c_syntax(reward_code)
    if not valid:
        print(f"   [FAIL] Sintaxis inválida: {msg}")
        return None

    # 2. Compilar
    output = compile_reward_function(reward_code)
    if output.result != CompileResult.SUCCESS:
        print(f"   [FAIL] Compilación: {output.message}")
        return None

    # 3. Verificar librería
    success, msg = test_compiled_library(output.lib_path)
    if not success:
        print(f"   [FAIL] Test librería: {msg}")
        return None

    # 4. Entrenar brevemente
    env = DroneVecEnv(num_envs=num_envs)
    policy = DronePolicy(obs_size=15, action_size=4, hidden_size=64)

    trainer = PPOTrainer(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        n_steps=64,  # Rollouts más cortos para evaluación rápida
        n_epochs=2,
        batch_size=128
    )

    # Entrenar
    metrics_history = []
    trainer.policy.eval()

    while trainer.total_steps < eval_steps:
        next_value = trainer.collect_rollout()
        advantages, returns = trainer.compute_gae(next_value)

        for _ in range(trainer.n_epochs):
            trainer.train_epoch(advantages, returns)

        # Recolectar métricas
        with torch.no_grad():
            obs = trainer.obs
            rewards = trainer.rewards_buffer.mean().item()
            metrics_history.append(rewards)

    # 5. Evaluar política final
    eval_rewards = []
    eval_steps_list = []
    eval_collisions = []
    eval_distances = []
    eval_speeds = []
    eval_orientations = []

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    for _ in range(10):  # 10 episodios de evaluación
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < 500:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(obs_tensor)
                action_np = action.cpu().numpy()
                action_np = np.clip(action_np, -1, 1)

            obs, rewards, dones, _, _ = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            episode_reward += rewards.mean()
            episode_steps += 1

            # Calcular métricas del estado
            # obs format: [x, y, z, vx, vy, vz, roll, pitch, yaw, ...]
            speeds = np.sqrt(obs[:, 3]**2 + obs[:, 4]**2 + obs[:, 5]**2) * 5.0
            orientations = np.abs(obs[:, 6]) + np.abs(obs[:, 7])
            distances = np.sqrt(obs[:, 12]**2 + obs[:, 13]**2 + obs[:, 14]**2)

            eval_speeds.append(speeds.mean())
            eval_orientations.append(orientations.mean())
            eval_distances.append(distances.mean())

            if dones.any():
                done = True

        eval_rewards.append(episode_reward)
        eval_steps_list.append(episode_steps)

    env.close()
    training_time = time.time() - start_time

    return EvaluationResult(
        code=reward_code,
        mean_reward=np.mean(eval_rewards),
        max_reward=np.max(eval_rewards),
        min_reward=np.min(eval_rewards),
        mean_steps=np.mean(eval_steps_list),
        mean_collisions=0.0,  # TODO: agregar tracking de colisiones
        final_distance=np.mean(eval_distances) if eval_distances else 1.0,
        mean_speed=np.mean(eval_speeds) if eval_speeds else 0.0,
        mean_orientation=np.mean(eval_orientations) if eval_orientations else 0.0,
        compile_success=True,
        training_time=training_time,
        generation=generation
    )


def run_evolutionary_search(
    task: str = "hover",
    generations: int = 5,
    population_size: int = 4,
    eval_steps: int = 100_000,
    use_api: bool = False,
    verbose: bool = True
) -> tuple[str, EvaluationResult]:
    """
    Ejecuta el bucle evolutivo para encontrar la mejor función de recompensa.

    Args:
        task: Tipo de tarea ("hover", "waypoint", "smooth")
        generations: Número de generaciones evolutivas
        population_size: Tamaño de la población por generación
        eval_steps: Pasos de entrenamiento para evaluación
        use_api: Si usar API de LLM (True) o mocks (False)
        verbose: Si imprimir progreso detallado

    Returns:
        (mejor_código, mejor_resultado)
    """
    print("=" * 60)
    print("   BÚSQUEDA EVOLUTIVA DE RECOMPENSAS (Eureka-style)")
    print(f"   Tarea: {task.upper()}")
    print(f"   Generaciones: {generations} | Población: {population_size}")
    print("=" * 60)

    best_code = None
    best_result = None
    best_reward = float('-inf')

    # Historial para análisis
    history = []

    # 1. Generación inicial
    print(f"\n--- Generación 0: Inicialización ---")
    candidates = []

    for i in range(population_size):
        if use_api:
            code = generate_reward_function(task)
        else:
            code = generate_reward_mock(task)
        candidates.append(code)
        if verbose:
            print(f"   Candidato {i+1}: {len(code)} caracteres")

    # 2. Bucle evolutivo
    for gen in range(generations):
        print(f"\n--- Generación {gen + 1}/{generations} ---")
        results = []

        # Evaluar cada candidato
        for i, code in enumerate(candidates):
            print(f"\n   Evaluando candidato {i+1}/{len(candidates)}...")
            result = evaluate_reward_function(
                code,
                num_envs=32,
                eval_steps=eval_steps,
                generation=gen
            )

            if result:
                results.append(result)
                if verbose:
                    print(f"   [OK] Reward: {result.mean_reward:.3f} | "
                          f"Steps: {result.mean_steps:.1f} | "
                          f"Time: {result.training_time:.1f}s")
            else:
                print(f"   [SKIP] Candidato inválido")

        if not results:
            print("   [WARN] Ningún candidato válido en esta generación")
            # Regenerar candidatos
            candidates = []
            for _ in range(population_size):
                if use_api:
                    candidates.append(generate_reward_function(task))
                else:
                    candidates.append(generate_reward_mock(task))
            continue

        # 3. Selección - ordenar por recompensa
        results.sort(key=lambda x: x.mean_reward, reverse=True)

        # Actualizar mejor global
        if results[0].mean_reward > best_reward:
            best_reward = results[0].mean_reward
            best_result = results[0]
            best_code = results[0].code
            print(f"\n   ★ NUEVO MEJOR: {best_reward:.3f}")

        history.append({
            'generation': gen,
            'best_reward': results[0].mean_reward,
            'mean_reward': np.mean([r.mean_reward for r in results]),
            'population_size': len(results)
        })

        # 4. Elitismo + Mutación
        print(f"\n   Generando nueva población...")
        new_candidates = [results[0].code]  # Mantener el mejor (elitismo)

        # Convertir métricas a dict para evolve_reward_function
        best_metrics = {
            'mean_reward': results[0].mean_reward,
            'max_reward': results[0].max_reward,
            'min_reward': results[0].min_reward,
            'mean_steps': results[0].mean_steps,
            'mean_collisions': results[0].mean_collisions,
            'final_distance': results[0].final_distance,
            'mean_speed': results[0].mean_speed,
            'mean_orientation': results[0].mean_orientation,
        }

        # Generar mutaciones del mejor
        for i in range(population_size - 1):
            if use_api:
                mutated = evolve_reward_function(task, results[0].code, best_metrics)
            else:
                # Para mock, simplemente regenerar
                mutated = generate_reward_mock(task)
            new_candidates.append(mutated)

        candidates = new_candidates
        print(f"   Nueva población: {len(candidates)} candidatos")

    # 5. Resumen final
    print("\n" + "=" * 60)
    print("   BÚSQUEDA EVOLUTIVA COMPLETADA")
    print("=" * 60)

    if best_result:
        print(f"\n   Mejor recompensa: {best_result.mean_reward:.3f}")
        print(f"   Pasos promedio: {best_result.mean_steps:.1f}")
        print(f"   Generación: {best_result.generation}")
        print(f"\n   Código ({len(best_code)} caracteres):")
        print("   " + "-" * 40)
        for line in best_code.split('\n')[:10]:
            print(f"   {line}")
        print("   ...")

    return best_code, best_result


def train_with_best_reward(
    best_code: str,
    task: str = "hover",
    total_steps: int = 500_000,
    num_envs: int = 64
):
    """
    Entrena una política completa con la mejor función de recompensa encontrada.
    """
    print("\n" + "=" * 60)
    print("   ENTRENAMIENTO FINAL CON MEJOR RECOMPENSA")
    print("=" * 60)

    # Compilar
    output = compile_reward_function(best_code)
    if output.result != CompileResult.SUCCESS:
        print(f"[ERROR] No se pudo compilar: {output.message}")
        return None

    # Crear entorno y política
    env = DroneVecEnv(num_envs=num_envs)
    policy = DronePolicy(obs_size=15, action_size=4, hidden_size=64)

    trainer = PPOTrainer(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        n_steps=128,
        n_epochs=4,
        batch_size=256
    )

    # Entrenar
    trained_policy = trainer.train(total_steps=total_steps, log_interval=50000)

    # Guardar
    model_path = Path(__file__).parent / "models"
    model_path.mkdir(exist_ok=True)
    save_path = model_path / f"drone_policy_{task}_evolved.pt"
    torch.save(trained_policy.state_dict(), save_path)
    print(f"\nModelo guardado en: {save_path}")

    # Guardar código de recompensa
    reward_path = model_path / f"reward_{task}_evolved.c"
    with open(reward_path, 'w') as f:
        f.write(best_code)
    print(f"Recompensa guardada en: {reward_path}")

    env.close()
    return trained_policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Búsqueda evolutiva de recompensas")
    parser.add_argument("--task", type=str, default="hover",
                        choices=["hover", "waypoint", "smooth"])
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=50000)
    parser.add_argument("--api", action="store_true", help="Usar API de LLM")
    parser.add_argument("--train-final", action="store_true",
                        help="Entrenar política final con mejor recompensa")
    parser.add_argument("--final-steps", type=int, default=500000)

    args = parser.parse_args()

    # Ejecutar búsqueda evolutiva
    best_code, best_result = run_evolutionary_search(
        task=args.task,
        generations=args.generations,
        population_size=args.population,
        eval_steps=args.eval_steps,
        use_api=args.api,
        verbose=True
    )

    # Opcionalmente entrenar política final
    if args.train_final and best_code:
        train_with_best_reward(
            best_code,
            task=args.task,
            total_steps=args.final_steps
        )
