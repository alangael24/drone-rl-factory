"""
Simple PPO - Entrenador PPO simplificado para evaluación rápida

Diseñado para evaluar candidatos de recompensa en el bucle evolutivo.
No tiene todas las características del PPO completo pero es suficiente
para determinar si una función de recompensa permite aprendizaje.
"""

import numpy as np
from universal_judge import MetricsCollector, TrainingMetrics


class SimplePPO:
    """
    PPO simplificado para evaluación rápida de candidatos.

    No tiene todas las características del PPO completo,
    pero es suficiente para evaluar si una recompensa funciona.
    """

    def __init__(
        self,
        env,
        domain,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_steps: int = 128
    ):
        import torch
        import torch.nn as nn

        self.env = env
        self.domain = domain
        self.num_envs = env.num_envs
        self.obs_size = domain.obs_size
        self.action_size = domain.action_size

        self.lr = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_steps = n_steps

        # Red neuronal simple
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        ).to(self.device)

        self.actor_mean = nn.Linear(64, self.action_size).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(self.action_size, device=self.device))
        self.critic = nn.Linear(64, 1).to(self.device)

        # Optimizador
        params = list(self.policy.parameters()) + [self.actor_logstd] + \
                 list(self.actor_mean.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        # Métricas
        self.collector = MetricsCollector()

    def train(self, total_steps: int) -> TrainingMetrics:
        """Entrena por el número especificado de pasos"""
        import torch
        from torch.distributions import Normal

        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        steps_done = 0
        episode_rewards = []
        current_ep_reward = np.zeros(self.num_envs)

        while steps_done < total_steps:
            # Buffers para rollout
            obs_buffer = []
            actions_buffer = []
            rewards_buffer = []
            dones_buffer = []
            values_buffer = []
            log_probs_buffer = []

            # Recolectar rollout
            for _ in range(self.n_steps):
                with torch.no_grad():
                    features = self.policy(obs)
                    mean = self.actor_mean(features)
                    std = torch.exp(self.actor_logstd)
                    dist = Normal(mean, std)

                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    value = self.critic(features).squeeze(-1)

                # Ejecutar en entorno
                action_np = action.cpu().numpy()
                next_obs, rewards, dones, _, _ = self.env.step(action_np)

                # Guardar en buffers
                obs_buffer.append(obs)
                actions_buffer.append(action)
                rewards_buffer.append(torch.tensor(rewards, device=self.device))
                dones_buffer.append(torch.tensor(dones, device=self.device))
                values_buffer.append(value)
                log_probs_buffer.append(log_prob)

                # Actualizar métricas
                current_ep_reward += rewards
                for i, done in enumerate(dones):
                    if done:
                        episode_rewards.append(current_ep_reward[i])
                        self.collector.on_step(current_ep_reward[i], True)
                        current_ep_reward[i] = 0

                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                steps_done += self.num_envs

            # Calcular ventajas (GAE)
            with torch.no_grad():
                features = self.policy(obs)
                next_value = self.critic(features).squeeze(-1)

            advantages, returns = self._compute_gae(
                rewards_buffer, dones_buffer, values_buffer, next_value
            )

            # Optimizar
            self._optimize(
                obs_buffer, actions_buffer, log_probs_buffer,
                advantages, returns
            )

            # Registrar progreso
            if episode_rewards:
                mean_reward = np.mean(episode_rewards[-100:])
                self.collector.on_rollout_end(mean_reward)

        return self.collector.get_metrics()

    def _compute_gae(self, rewards, dones, values, next_value):
        """Calcula Generalized Advantage Estimation"""
        import torch

        advantages = []
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t].float()) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(values)

        return advantages, returns

    def _optimize(self, obs_buffer, actions_buffer, old_log_probs, advantages, returns):
        """Optimiza la política con PPO"""
        import torch
        from torch.distributions import Normal

        obs = torch.stack(obs_buffer)
        actions = torch.stack(actions_buffer)
        old_log_probs = torch.stack(old_log_probs)

        # Flatten
        obs = obs.view(-1, self.obs_size)
        actions = actions.view(-1, self.action_size)
        old_log_probs = old_log_probs.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # Normalizar ventajas
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimizar
        features = self.policy(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        new_log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().mean()
        values = self.critic(features).squeeze(-1)

        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Pérdida de política (clipped)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Pérdida de valor
        value_loss = 0.5 * (returns - values).pow(2).mean()

        # Pérdida total
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + [self.actor_logstd] +
            list(self.actor_mean.parameters()) + list(self.critic.parameters()),
            0.5
        )
        self.optimizer.step()

        self.collector.on_train_step(
            policy_loss.item(),
            value_loss.item(),
            entropy.item()
        )
