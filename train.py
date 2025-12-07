#!/usr/bin/env python3
"""
Sprint 3: Entrenamiento RL con PPO
Usa PufferLib para entrenar una política neuronal de control de drones.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Importaciones locales
from drone_env import DroneVecEnv
from architect import generate_reward_function, generate_reward_mock
from compiler import compile_reward_function, test_compiled_library, CompileResult


class DronePolicy(nn.Module):
    """
    Red neuronal simple para control de drones.
    Arquitectura: MLP con 2 capas ocultas.
    """

    def __init__(self, obs_size: int = 15, action_size: int = 4, hidden_size: int = 64):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size

        # Red compartida (feature extractor)
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Cabeza del actor (política)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_logstd = nn.Parameter(torch.zeros(action_size))

        # Cabeza del crítico (valor)
        self.critic = nn.Linear(hidden_size, 1)

        # Inicialización
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Escalar salidas del actor
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, obs):
        features = self.shared(obs)
        return features

    def get_action_and_value(self, obs, action=None):
        features = self.forward(obs)

        # Actor
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)

        # Distribución normal
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Crítico
        value = self.critic(features)

        return action, log_prob, entropy, value

    def get_value(self, obs):
        features = self.forward(obs)
        return self.critic(features)


class PPOTrainer:
    """
    Entrenador PPO simplificado.
    """

    def __init__(
        self,
        env: DroneVecEnv,
        policy: DronePolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.num_envs = env.num_envs

        # Hiperparámetros
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Optimizador
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        # Buffers de rollout
        self.obs_buffer = torch.zeros((n_steps, self.num_envs, 15), device=device)
        self.actions_buffer = torch.zeros((n_steps, self.num_envs, 4), device=device)
        self.rewards_buffer = torch.zeros((n_steps, self.num_envs), device=device)
        self.dones_buffer = torch.zeros((n_steps, self.num_envs), device=device)
        self.values_buffer = torch.zeros((n_steps, self.num_envs), device=device)
        self.log_probs_buffer = torch.zeros((n_steps, self.num_envs), device=device)

        # Estado actual
        obs, _ = env.reset()
        self.obs = torch.tensor(obs, dtype=torch.float32, device=device)

        # Métricas
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollout(self):
        """Recolecta experiencias del entorno."""
        self.policy.eval()

        with torch.no_grad():
            for step in range(self.n_steps):
                # Guardar observación
                self.obs_buffer[step] = self.obs

                # Obtener acción
                action, log_prob, _, value = self.policy.get_action_and_value(self.obs)
                self.actions_buffer[step] = action
                self.log_probs_buffer[step] = log_prob
                self.values_buffer[step] = value.squeeze(-1)

                # Ejecutar en entorno
                action_np = action.cpu().numpy()
                action_np = np.clip(action_np, -1, 1)  # Asegurar rango

                obs, rewards, dones, truncated, infos = self.env.step(action_np)

                self.rewards_buffer[step] = torch.tensor(rewards, device=self.device)
                self.dones_buffer[step] = torch.tensor(dones, device=self.device)

                self.obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                self.total_steps += self.num_envs

            # Valor del siguiente estado (para GAE)
            next_value = self.policy.get_value(self.obs).squeeze(-1)

        return next_value

    def compute_gae(self, next_value):
        """Calcula Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards_buffer)
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones_buffer[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones_buffer[t]
                next_values = self.values_buffer[t + 1]

            delta = (
                self.rewards_buffer[t]
                + self.gamma * next_values * next_non_terminal
                - self.values_buffer[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values_buffer
        return advantages, returns

    def train_epoch(self, advantages, returns):
        """Entrena por una época."""
        self.policy.train()

        # Flatten buffers
        b_obs = self.obs_buffer.reshape(-1, 15)
        b_actions = self.actions_buffer.reshape(-1, 4)
        b_log_probs = self.log_probs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values_buffer.reshape(-1)

        # Normalizar ventajas
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Indices para minibatches
        batch_size = self.n_steps * self.num_envs
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        total_pg_loss = 0
        total_vf_loss = 0
        total_entropy = 0
        n_batches = 0

        for start in range(0, batch_size, self.batch_size):
            end = start + self.batch_size
            mb_idx = indices[start:end]

            _, new_log_prob, entropy, new_value = self.policy.get_action_and_value(
                b_obs[mb_idx], b_actions[mb_idx]
            )

            # Ratio de probabilidades
            log_ratio = new_log_prob - b_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            # Pérdida de política (clipped)
            pg_loss1 = -b_advantages[mb_idx] * ratio
            pg_loss2 = -b_advantages[mb_idx] * torch.clamp(
                ratio, 1 - self.clip_range, 1 + self.clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Pérdida de valor
            new_value = new_value.squeeze(-1)
            vf_loss = ((new_value - b_returns[mb_idx]) ** 2).mean()

            # Pérdida total
            loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy.mean()

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()
            n_batches += 1

        return {
            "pg_loss": total_pg_loss / n_batches,
            "vf_loss": total_vf_loss / n_batches,
            "entropy": total_entropy / n_batches,
        }

    def train(self, total_steps: int, log_interval: int = 10000, return_metrics: bool = False):
        """
        Loop principal de entrenamiento.

        Args:
            total_steps: Número total de pasos de entrenamiento
            log_interval: Intervalo para logging
            return_metrics: Si True, devuelve (policy, metrics_dict)

        Returns:
            policy o (policy, metrics_dict) si return_metrics=True
        """
        start_time = time.time()
        iteration = 0

        # Historial de métricas para análisis
        reward_history = []
        loss_history = []

        print(f"\nIniciando entrenamiento por {total_steps:,} pasos...")
        print(f"Entornos paralelos: {self.num_envs}")
        print(f"Pasos por rollout: {self.n_steps}")
        print("-" * 50)

        while self.total_steps < total_steps:
            # Recolectar experiencias
            next_value = self.collect_rollout()

            # Calcular ventajas
            advantages, returns = self.compute_gae(next_value)

            # Entrenar
            for epoch in range(self.n_epochs):
                metrics = self.train_epoch(advantages, returns)

            iteration += 1

            # Guardar métricas
            avg_reward = self.rewards_buffer.mean().item()
            reward_history.append(avg_reward)
            loss_history.append(metrics)

            # Logging
            if self.total_steps % log_interval < self.n_steps * self.num_envs:
                elapsed = time.time() - start_time
                sps = self.total_steps / elapsed

                print(
                    f"Pasos: {self.total_steps:>10,} | "
                    f"SPS: {sps:>8,.0f} | "
                    f"Reward: {avg_reward:>7.3f} | "
                    f"PG Loss: {metrics['pg_loss']:>7.4f} | "
                    f"VF Loss: {metrics['vf_loss']:>7.4f} | "
                    f"Entropy: {metrics['entropy']:>6.3f}"
                )

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Entrenamiento completado en {total_time:.1f}s")
        print(f"Velocidad promedio: {total_steps / total_time:,.0f} pasos/segundo")

        if return_metrics:
            # Calcular métricas finales detalladas
            final_metrics = {
                'mean_reward': np.mean(reward_history[-100:]) if reward_history else 0,
                'max_reward': np.max(reward_history) if reward_history else 0,
                'min_reward': np.min(reward_history) if reward_history else 0,
                'final_reward': reward_history[-1] if reward_history else 0,
                'reward_history': reward_history,
                'total_steps': self.total_steps,
                'training_time': total_time,
                'steps_per_second': total_steps / total_time,
                'final_pg_loss': loss_history[-1]['pg_loss'] if loss_history else 0,
                'final_vf_loss': loss_history[-1]['vf_loss'] if loss_history else 0,
                'final_entropy': loss_history[-1]['entropy'] if loss_history else 0,
            }
            return self.policy, final_metrics

        return self.policy


def train_with_generated_reward(
    task: str = "hover",
    total_steps: int = 500_000,
    num_envs: int = 64,
    use_mock: bool = True
):
    """
    Pipeline completo:
    1. Generar función de recompensa con LLM
    2. Compilar a C
    3. Entrenar política con PPO
    """
    print("=" * 60)
    print("   SPRINT 3: FÁBRICA DE CEREBROS")
    print(f"   Tarea: {task.upper()} | Pasos: {total_steps:,}")
    print("=" * 60)

    # 1. Generar recompensa
    print("\n[1/4] Generando función de recompensa...")
    if use_mock:
        print("   (Usando mock)")
        reward_code = generate_reward_mock(task)
    else:
        print("   (Llamando a LLM)")
        reward_code = generate_reward_function(task)

    print(f"   Código generado: {len(reward_code)} caracteres")

    # 2. Compilar
    print("\n[2/4] Compilando a C...")
    output = compile_reward_function(reward_code)

    if output.result != CompileResult.SUCCESS:
        print(f"   ERROR: {output.message}")
        print(f"   {output.stderr}")
        return None

    print(f"   Compilación exitosa: {output.lib_path}")

    # 3. Verificar
    print("\n[3/4] Verificando librería...")
    success, msg = test_compiled_library(output.lib_path)
    if not success:
        print(f"   ERROR: {msg}")
        return None
    print(f"   {msg}")

    # 4. Entrenar
    print("\n[4/4] Entrenando política neuronal...")

    # Crear entorno
    env = DroneVecEnv(num_envs=num_envs)

    # Crear política
    policy = DronePolicy(obs_size=15, action_size=4, hidden_size=64)

    # Crear entrenador
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

    # Guardar modelo
    model_path = Path(__file__).parent / "models"
    model_path.mkdir(exist_ok=True)
    save_path = model_path / f"drone_policy_{task}.pt"
    torch.save(trained_policy.state_dict(), save_path)
    print(f"\nModelo guardado en: {save_path}")

    env.close()
    return trained_policy


def evaluate_policy(policy: DronePolicy, num_episodes: int = 10):
    """Evalúa una política entrenada."""
    from drone_env import DroneGymEnv

    env = DroneGymEnv()
    policy.eval()

    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        episode_reward = 0
        steps = 0

        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(obs)
                action = action.squeeze(0).numpy()
                action = np.clip(action, -1, 1)

            obs, reward, done, truncated, info = env.step(action)
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            episode_reward += reward
            steps += 1

            if done or truncated:
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

    env.close()

    print(f"\nEvaluación ({num_episodes} episodios):")
    print(f"  Recompensa promedio: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Longitud promedio: {np.mean(episode_lengths):.1f} pasos")

    return total_rewards, episode_lengths


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "hover"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    use_api = "--api" in sys.argv

    policy = train_with_generated_reward(
        task=task,
        total_steps=steps,
        num_envs=64,
        use_mock=not use_api
    )

    if policy:
        evaluate_policy(policy, num_episodes=10)
