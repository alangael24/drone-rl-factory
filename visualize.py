#!/usr/bin/env python3
"""
Sprint 4: Visualizador de Políticas Entrenadas
Muestra la trayectoria 3D del dron controlado por la red neuronal.
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para WSL2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

from train import DronePolicy
from drone_env import DroneGymEnv


def load_policy(model_path: str) -> DronePolicy:
    """Carga una política entrenada."""
    policy = DronePolicy(obs_size=15, action_size=4, hidden_size=64)
    policy.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    policy.eval()
    return policy


def run_episode(policy: DronePolicy, max_steps: int = 500) -> dict:
    """
    Ejecuta un episodio completo y registra la trayectoria.

    Returns:
        dict con positions, velocities, orientations, rewards, actions
    """
    env = DroneGymEnv()

    # Registros
    positions = []
    velocities = []
    orientations = []
    rewards = []
    actions_log = []

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    total_reward = 0

    for step in range(max_steps):
        # Obtener acción de la política
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
            action = action.squeeze(0).numpy()
            action = np.clip(action, -1, 1)

        # Registrar estado (desnormalizar desde observaciones)
        pos = np.array([
            obs[0] * 10.0,  # x
            obs[1] * 10.0,  # y
            obs[2] * 5.0,   # z
        ])
        vel = np.array([obs[3], obs[4], obs[5]]) * 5.0
        orient = np.array([obs[6], obs[7], obs[8]])

        positions.append(pos.copy())
        velocities.append(vel.copy())
        orientations.append(orient.copy())
        actions_log.append(action.copy())

        # Ejecutar paso
        obs, reward, done, truncated, info = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        rewards.append(reward)
        total_reward += reward

        if done or truncated:
            break

    env.close()

    return {
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'orientations': np.array(orientations),
        'rewards': np.array(rewards),
        'actions': np.array(actions_log),
        'total_reward': total_reward,
        'steps': len(positions)
    }


def plot_trajectory_3d(trajectory: dict, title: str = "Trayectoria del Dron", save_path: str = None):
    """
    Crea una visualización 3D de la trayectoria.
    """
    positions = trajectory['positions']
    rewards = trajectory['rewards']

    fig = plt.figure(figsize=(14, 10))

    # === Plot 3D principal ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Trayectoria coloreada por tiempo
    n_points = len(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    for i in range(n_points - 1):
        ax1.plot3D(
            positions[i:i+2, 0],
            positions[i:i+2, 1],
            positions[i:i+2, 2],
            color=colors[i],
            linewidth=2
        )

    # Punto inicial (verde)
    ax1.scatter(*positions[0], color='green', s=100, marker='o', label='Inicio')

    # Punto final (rojo)
    ax1.scatter(*positions[-1], color='red', s=100, marker='s', label='Final')

    # Objetivo (estrella dorada)
    ax1.scatter(0, 0, 1, color='gold', s=200, marker='*', label='Objetivo (0,0,1)')

    # Suelo
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    ax1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='gray')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title}\nRecompensa Total: {trajectory["total_reward"]:.1f}')
    ax1.legend()
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([0, 3])

    # === Plot de altura vs tiempo ===
    ax2 = fig.add_subplot(2, 2, 2)
    time = np.arange(len(positions)) * 0.02  # dt = 0.02s

    ax2.plot(time, positions[:, 2], 'b-', linewidth=2, label='Altura')
    ax2.axhline(y=1.0, color='gold', linestyle='--', linewidth=2, label='Objetivo (1m)')
    ax2.fill_between(time, 0, positions[:, 2], alpha=0.3)

    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Altura (m)')
    ax2.set_title('Perfil de Altura')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 2])

    # === Plot de recompensa acumulada ===
    ax3 = fig.add_subplot(2, 2, 3)
    cumulative_reward = np.cumsum(rewards)

    ax3.plot(time[:len(rewards)], cumulative_reward, 'g-', linewidth=2)
    ax3.fill_between(time[:len(rewards)], 0, cumulative_reward, alpha=0.3, color='green')

    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Recompensa Acumulada')
    ax3.set_title('Curva de Recompensa')
    ax3.grid(True, alpha=0.3)

    # === Plot de orientación ===
    ax4 = fig.add_subplot(2, 2, 4)
    orientations = trajectory['orientations']

    ax4.plot(time, np.degrees(orientations[:, 0]), 'r-', label='Roll', linewidth=1.5)
    ax4.plot(time, np.degrees(orientations[:, 1]), 'g-', label='Pitch', linewidth=1.5)
    ax4.plot(time, np.degrees(orientations[:, 2]), 'b-', label='Yaw', linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Tiempo (s)')
    ax4.set_ylabel('Ángulo (grados)')
    ax4.set_title('Orientación del Dron')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-45, 45])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

    plt.show()


def plot_multiple_trajectories(trajectories: list, title: str = "Múltiples Trayectorias", save_path: str = None):
    """
    Visualiza múltiples trayectorias en el mismo gráfico 3D.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

    for i, traj in enumerate(trajectories):
        positions = traj['positions']
        ax.plot3D(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f'Ep {i+1} (R={traj["total_reward"]:.1f})'
        )
        ax.scatter(*positions[0], color=colors[i], s=50, marker='o')
        ax.scatter(*positions[-1], color=colors[i], s=50, marker='s')

    # Objetivo
    ax.scatter(0, 0, 1, color='gold', s=300, marker='*', label='Objetivo')

    # Suelo
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='gray')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

    plt.show()


def create_summary_report(trajectories: list, policy_name: str):
    """
    Crea un resumen estadístico de las evaluaciones.
    """
    rewards = [t['total_reward'] for t in trajectories]
    steps = [t['steps'] for t in trajectories]

    # Calcular distancia final al objetivo para cada trayectoria
    final_distances = []
    for t in trajectories:
        final_pos = t['positions'][-1]
        target = np.array([0, 0, 1])
        dist = np.linalg.norm(final_pos - target)
        final_distances.append(dist)

    print("\n" + "=" * 60)
    print(f"   REPORTE DE EVALUACIÓN: {policy_name}")
    print("=" * 60)
    print(f"\nEpisodios evaluados: {len(trajectories)}")
    print(f"\nRecompensa:")
    print(f"  Media:    {np.mean(rewards):>8.2f}")
    print(f"  Std:      {np.std(rewards):>8.2f}")
    print(f"  Min:      {np.min(rewards):>8.2f}")
    print(f"  Max:      {np.max(rewards):>8.2f}")
    print(f"\nDuración (pasos):")
    print(f"  Media:    {np.mean(steps):>8.1f}")
    print(f"  Min:      {np.min(steps):>8d}")
    print(f"  Max:      {np.max(steps):>8d}")
    print(f"\nDistancia final al objetivo:")
    print(f"  Media:    {np.mean(final_distances):>8.3f} m")
    print(f"  Min:      {np.min(final_distances):>8.3f} m")
    print(f"  Max:      {np.max(final_distances):>8.3f} m")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualizador de Políticas de Drones')
    parser.add_argument('--model', type=str, default='models/drone_policy_hover.pt',
                        help='Path al modelo .pt')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Número de episodios a visualizar')
    parser.add_argument('--save', type=str, default=None,
                        help='Path para guardar el gráfico')
    parser.add_argument('--single', action='store_true',
                        help='Mostrar solo un episodio con detalles')

    args = parser.parse_args()

    # Verificar modelo
    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        print("\nModelos disponibles:")
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            for m in models_dir.glob("*.pt"):
                print(f"  - models/{m.name}")
        sys.exit(1)

    print("=" * 60)
    print("   SPRINT 4: VISUALIZADOR DE POLÍTICAS")
    print("=" * 60)
    print(f"\nCargando modelo: {model_path}")

    # Cargar política
    policy = load_policy(str(model_path))
    print("Modelo cargado exitosamente")

    # Ejecutar episodios
    print(f"\nEjecutando {args.episodes} episodio(s)...")
    trajectories = []

    for i in range(args.episodes):
        traj = run_episode(policy)
        trajectories.append(traj)
        print(f"  Episodio {i+1}: {traj['steps']} pasos, recompensa = {traj['total_reward']:.2f}")

    # Generar reporte
    create_summary_report(trajectories, args.model)

    # Visualizar
    policy_name = Path(args.model).stem

    if args.single or args.episodes == 1:
        # Visualización detallada de un episodio
        save_path = args.save or f"trajectory_{policy_name}.png"
        plot_trajectory_3d(
            trajectories[0],
            title=f"Trayectoria - {policy_name}",
            save_path=save_path
        )
    else:
        # Visualización de múltiples trayectorias
        save_path = args.save or f"trajectories_{policy_name}.png"
        plot_multiple_trajectories(
            trajectories,
            title=f"Trayectorias - {policy_name}",
            save_path=save_path
        )


if __name__ == "__main__":
    main()
