"""
Universal Evolution - Bucle evolutivo para CUALQUIER dominio robótico

Combina todos los componentes universales:
- DomainSpec: Define el sistema
- UniversalArchitect: Genera código
- UniversalCompiler: Compila a C
- UniversalEnv: Ejecuta simulación
- UniversalJudge: Evalúa aprendizaje

Implementa el algoritmo Eureka de forma genérica.
"""

import os
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from domain_spec import DomainSpec, get_domain
from universal_architect import UniversalArchitect, GeneratedCode
from universal_compiler import UniversalCompiler, CompileOutput, CompileResult
from universal_env import UniversalVecEnv
from universal_judge import (
    UniversalJudge, TrainingMetrics, JudgmentResult,
    MetricsCollector, generate_semantic_reflection, LearningQuality
)


@dataclass
class EvolutionCandidate:
    """Un candidato en la población evolutiva"""
    generation: int
    candidate_id: int

    # Código generado
    physics_code: str
    reward_code: str

    # Resultados de compilación
    compiled: bool = False
    library_path: str = ""

    # Métricas de entrenamiento
    judgment: Optional[JudgmentResult] = None
    training_time: float = 0.0

    # Metadatos
    parent_id: Optional[int] = None
    mutations: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Score del candidato (0-100)"""
        return self.judgment.score if self.judgment else 0

    @property
    def quality(self) -> str:
        """Calidad del aprendizaje"""
        return self.judgment.quality.value if self.judgment else "unknown"


@dataclass
class EvolutionResult:
    """Resultado de la evolución completa"""
    best_candidate: EvolutionCandidate
    all_candidates: List[EvolutionCandidate]
    total_time: float
    generations: int

    # Estadísticas por generación
    generation_stats: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Genera resumen de la evolución"""
        lines = [
            "=== Resumen de Evolución ===",
            f"Generaciones: {self.generations}",
            f"Tiempo total: {self.total_time:.1f}s",
            f"Mejor score: {self.best_candidate.score:.1f}",
            f"Mejor calidad: {self.best_candidate.quality}",
            "",
            "Progreso por generación:",
        ]

        for i, stats in enumerate(self.generation_stats):
            lines.append(
                f"  Gen {i}: mejor={stats['best_score']:.1f}, "
                f"promedio={stats['mean_score']:.1f}"
            )

        return "\n".join(lines)


class UniversalEvolution:
    """
    Motor de evolución universal que funciona con cualquier dominio.

    Algoritmo:
    1. Generar población inicial de funciones de recompensa
    2. Para cada generación:
       a. Compilar y entrenar cada candidato
       b. Evaluar con el juez universal
       c. Seleccionar los mejores
       d. Mutar/evolucionar para siguiente generación
    3. Retornar el mejor candidato
    """

    def __init__(
        self,
        domain: DomainSpec,
        task_description: str,
        population_size: int = 4,
        generations: int = 5,
        eval_steps: int = 100_000,
        num_envs: int = 32,
        use_mock: bool = False,
        output_dir: str = "c_src"
    ):
        """
        Inicializa el motor de evolución.

        Args:
            domain: Especificación del dominio
            task_description: Descripción de la tarea
            population_size: Tamaño de la población
            generations: Número de generaciones
            eval_steps: Pasos de entrenamiento por evaluación
            num_envs: Número de entornos paralelos
            use_mock: Si usar código mock
            output_dir: Directorio para archivos generados
        """
        self.domain = domain
        self.task_description = task_description
        self.population_size = population_size
        self.generations = generations
        self.eval_steps = eval_steps
        self.num_envs = num_envs
        self.use_mock = use_mock
        self.output_dir = output_dir

        # Componentes
        self.architect = UniversalArchitect(use_mock=use_mock)
        self.compiler = UniversalCompiler(output_dir)
        self.judge = UniversalJudge()

        # Estado
        self.current_generation = 0
        self.all_candidates: List[EvolutionCandidate] = []
        self.best_candidate: Optional[EvolutionCandidate] = None

    def run(self, verbose: bool = True) -> EvolutionResult:
        """
        Ejecuta la evolución completa.

        Args:
            verbose: Si imprimir progreso

        Returns:
            EvolutionResult con el mejor candidato y estadísticas
        """
        start_time = time.time()
        generation_stats = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"EVOLUCIÓN UNIVERSAL - {self.domain.name}")
            print(f"Tarea: {self.task_description}")
            print(f"Población: {self.population_size}, Generaciones: {self.generations}")
            print(f"{'='*60}\n")

        # Generar física una vez (compartida por todos los candidatos)
        if verbose:
            print("Generando función de física...")

        physics_result = self.architect.generate_physics(self.domain)
        if not physics_result.success:
            raise RuntimeError(f"Error generando física: {physics_result.error_message}")

        physics_code = physics_result.physics_code

        # Generación 0: Población inicial
        if verbose:
            print(f"\n--- Generación 0: Población Inicial ---")

        population = self._generate_initial_population(physics_code)

        for gen in range(self.generations):
            self.current_generation = gen

            if verbose:
                print(f"\n--- Generación {gen} ---")

            # Evaluar población
            evaluated = self._evaluate_population(population, physics_code, verbose)

            # Estadísticas de generación
            scores = [c.score for c in evaluated]
            stats = {
                "generation": gen,
                "best_score": max(scores),
                "mean_score": np.mean(scores),
                "worst_score": min(scores),
                "best_quality": evaluated[0].quality,
            }
            generation_stats.append(stats)

            if verbose:
                print(f"  Mejor: {stats['best_score']:.1f} ({stats['best_quality']})")
                print(f"  Promedio: {stats['mean_score']:.1f}")

            # Actualizar mejor global
            if self.best_candidate is None or evaluated[0].score > self.best_candidate.score:
                self.best_candidate = evaluated[0]

            # Si estamos en la última generación, terminar
            if gen == self.generations - 1:
                break

            # Evolucionar para siguiente generación
            population = self._evolve_population(evaluated, physics_code, verbose)

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"EVOLUCIÓN COMPLETADA")
            print(f"Tiempo: {total_time:.1f}s")
            print(f"Mejor score: {self.best_candidate.score:.1f}")
            print(f"Mejor calidad: {self.best_candidate.quality}")
            print(f"{'='*60}\n")

        return EvolutionResult(
            best_candidate=self.best_candidate,
            all_candidates=self.all_candidates,
            total_time=total_time,
            generations=self.generations,
            generation_stats=generation_stats,
        )

    def _generate_initial_population(
        self,
        physics_code: str
    ) -> List[EvolutionCandidate]:
        """Genera la población inicial de candidatos"""
        population = []

        for i in range(self.population_size):
            # Generar recompensa
            result = self.architect.generate_reward(self.domain, self.task_description)

            if not result.success:
                # Usar recompensa por defecto si falla
                result.reward_code = self._default_reward_code()

            candidate = EvolutionCandidate(
                generation=0,
                candidate_id=len(self.all_candidates),
                physics_code=physics_code,
                reward_code=result.reward_code,
            )

            population.append(candidate)
            self.all_candidates.append(candidate)

        return population

    def _evaluate_population(
        self,
        population: List[EvolutionCandidate],
        physics_code: str,
        verbose: bool = True
    ) -> List[EvolutionCandidate]:
        """
        Evalúa todos los candidatos de la población.

        Retorna la población ordenada por score (mejor primero).
        """
        for i, candidate in enumerate(population):
            if verbose:
                print(f"  Evaluando candidato {i+1}/{len(population)}...", end=" ")

            try:
                # Compilar
                output = self.compiler.compile(
                    self.domain,
                    physics_code,
                    candidate.reward_code,
                    output_name=f"lib{self.domain.name.lower()}_gen{self.current_generation}_c{i}"
                )

                if output.result != CompileResult.SUCCESS:
                    candidate.compiled = False
                    candidate.judgment = JudgmentResult(
                        quality=LearningQuality.FAILED,
                        score=0,
                        learning_gain=0,
                        learning_speed=0,
                        stability=0,
                        monotonicity=0,
                        final_performance=0,
                        diagnosis=f"Error de compilación: {output.error_message}",
                        recommendations=["Corregir errores de sintaxis C"],
                    )
                    if verbose:
                        print("FALLO (compilación)")
                    continue

                candidate.compiled = True
                candidate.library_path = output.library_path

                # Entrenar y evaluar
                start = time.time()
                metrics = self._train_and_evaluate(candidate)
                candidate.training_time = time.time() - start

                # Juzgar
                candidate.judgment = self.judge.judge(metrics)

                if verbose:
                    print(f"score={candidate.score:.1f} ({candidate.quality})")

            except Exception as e:
                candidate.judgment = JudgmentResult(
                    quality=LearningQuality.FAILED,
                    score=0,
                    learning_gain=0,
                    learning_speed=0,
                    stability=0,
                    monotonicity=0,
                    final_performance=0,
                    diagnosis=f"Error: {str(e)}",
                    recommendations=["Revisar código generado"],
                )
                if verbose:
                    print(f"ERROR: {e}")

        # Ordenar por score
        population.sort(key=lambda c: c.score, reverse=True)

        return population

    def _train_and_evaluate(
        self,
        candidate: EvolutionCandidate
    ) -> TrainingMetrics:
        """
        Entrena un candidato y retorna métricas.

        Usa PPO simplificado para evaluación rápida.
        """
        from simple_ppo import SimplePPO

        # Crear entorno
        env = UniversalVecEnv(
            self.num_envs,
            self.domain,
            candidate.library_path
        )

        try:
            # Entrenar
            ppo = SimplePPO(env, self.domain)
            metrics = ppo.train(self.eval_steps)

            return metrics

        finally:
            env.close()

    def _evolve_population(
        self,
        evaluated: List[EvolutionCandidate],
        physics_code: str,
        verbose: bool = True
    ) -> List[EvolutionCandidate]:
        """
        Crea la siguiente generación mediante selección, crossover y mutación.

        Estrategia (Paper R* + Eureka):
        - Mantener el mejor (elitismo)
        - Crossover entre los 2 mejores (si hay suficientes candidatos buenos)
        - Mutar los demás basándose en el mejor con reflexión semántica
        """
        next_population = []

        # Elitismo: mantener el mejor
        best = evaluated[0]
        elite = EvolutionCandidate(
            generation=self.current_generation + 1,
            candidate_id=len(self.all_candidates),
            physics_code=physics_code,
            reward_code=best.reward_code,
            parent_id=best.candidate_id,
            mutations=["elite (sin cambios)"],
        )
        next_population.append(elite)
        self.all_candidates.append(elite)

        # Detectar fortalezas de cada candidato para crossover
        def get_strengths(candidate: EvolutionCandidate) -> str:
            if not candidate.judgment:
                return "desconocidas"
            strengths = []
            if candidate.judgment.stability > 0.6:
                strengths.append("estabilidad")
            if candidate.judgment.learning_speed > 0.5:
                strengths.append("velocidad de convergencia")
            if candidate.judgment.monotonicity > 0.6:
                strengths.append("aprendizaje monotónico")
            if candidate.judgment.final_performance > 0.5:
                strengths.append("rendimiento final")
            return ", ".join(strengths) if strengths else "supervivencia básica"

        # Crossover: combinar los 2 mejores (si hay al menos 2 candidatos con score > 20)
        good_candidates = [c for c in evaluated if c.score > 20]

        if len(good_candidates) >= 2 and self.population_size > 2:
            if verbose:
                print(f"  Crossover entre candidatos con scores {good_candidates[0].score:.1f} y {good_candidates[1].score:.1f}")

            parent_a = good_candidates[0]
            parent_b = good_candidates[1]

            # Preparar datos para crossover
            candidate_a = {
                "code": parent_a.reward_code,
                "score": parent_a.score,
                "strengths": get_strengths(parent_a),
            }
            candidate_b = {
                "code": parent_b.reward_code,
                "score": parent_b.score,
                "strengths": get_strengths(parent_b),
            }

            # Realizar crossover
            crossover_result = self.architect.crossover_rewards(
                self.domain, candidate_a, candidate_b
            )

            crossover_child = EvolutionCandidate(
                generation=self.current_generation + 1,
                candidate_id=len(self.all_candidates),
                physics_code=physics_code,
                reward_code=crossover_result.reward_code if crossover_result.success else best.reward_code,
                parent_id=best.candidate_id,
                mutations=[f"crossover({parent_a.candidate_id}, {parent_b.candidate_id})"],
            )
            next_population.append(crossover_child)
            self.all_candidates.append(crossover_child)
            crossover_slots = 1
        else:
            crossover_slots = 0

        # Generar mutaciones del mejor con reflexión semántica (Eureka Strategy)
        mutations_needed = self.population_size - 1 - crossover_slots

        for i in range(mutations_needed):
            # Generar reflexión semántica mejorada
            reflection = generate_semantic_reflection(best.judgment)

            # Preparar métricas extendidas para evolución
            extended_metrics = best.judgment.raw_metrics.copy() if best.judgment else {}
            if best.judgment:
                extended_metrics["score"] = best.score
                extended_metrics["learning_gain"] = best.judgment.learning_gain
                extended_metrics["stability"] = best.judgment.stability

            # Evolucionar recompensa
            evolved = self.architect.evolve_reward(
                self.domain,
                best.reward_code,
                extended_metrics,
                reflection
            )

            mutant = EvolutionCandidate(
                generation=self.current_generation + 1,
                candidate_id=len(self.all_candidates),
                physics_code=physics_code,
                reward_code=evolved.reward_code if evolved.success else best.reward_code,
                parent_id=best.candidate_id,
                mutations=best.judgment.recommendations[:2] if best.judgment else [],
            )

            next_population.append(mutant)
            self.all_candidates.append(mutant)

        return next_population

    def _default_reward_code(self) -> str:
        """Código de recompensa por defecto"""
        return f"""float calculate_reward({self.domain.state_struct_name}* state) {{
    // Recompensa por defecto: sobrevivir
    return 1.0f;
}}"""

    def save_best(self, filepath: str):
        """Guarda el mejor candidato a un archivo"""
        if not self.best_candidate:
            raise ValueError("No hay candidato para guardar")

        data = {
            "domain": self.domain.name,
            "task": self.task_description,
            "score": self.best_candidate.score,
            "quality": self.best_candidate.quality,
            "physics_code": self.best_candidate.physics_code,
            "reward_code": self.best_candidate.reward_code,
            "diagnosis": self.best_candidate.judgment.diagnosis if self.best_candidate.judgment else "",
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================
# PPO SIMPLIFICADO PARA EVALUACIÓN RÁPIDA
# ============================================================

class SimplePPO:
    """
    PPO simplificado para evaluación rápida de candidatos.

    No tiene todas las características del PPO completo,
    pero es suficiente para evaluar si una recompensa funciona.
    """

    def __init__(
        self,
        env: UniversalVecEnv,
        domain: DomainSpec,
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


# ============================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================

def evolve_domain(
    domain_name: str,
    task: str,
    generations: int = 3,
    population_size: int = 3,
    eval_steps: int = 50_000,
    use_mock: bool = False,
    verbose: bool = True
) -> EvolutionResult:
    """
    Evoluciona una función de recompensa para un dominio predefinido.

    Args:
        domain_name: Nombre del dominio (drone, cartpole, etc.)
        task: Descripción de la tarea
        generations: Número de generaciones
        population_size: Tamaño de la población
        eval_steps: Pasos de evaluación por candidato
        use_mock: Si usar código mock
        verbose: Si imprimir progreso

    Returns:
        EvolutionResult
    """
    domain = get_domain(domain_name)

    evolution = UniversalEvolution(
        domain=domain,
        task_description=task,
        population_size=population_size,
        generations=generations,
        eval_steps=eval_steps,
        use_mock=use_mock,
    )

    return evolution.run(verbose=verbose)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=== Universal Evolution Demo ===\n")

    # Verificar que torch está disponible
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch no disponible. Instalando...")
        import subprocess
        subprocess.run(["pip", "install", "torch"])

    # Ejecutar evolución en modo mock (rápido)
    print("\nEjecutando evolución para CartPole...")

    try:
        result = evolve_domain(
            domain_name="cartpole",
            task="Mantener el palo vertical el mayor tiempo posible",
            generations=2,
            population_size=2,
            eval_steps=10_000,
            use_mock=True,
            verbose=True
        )

        print("\n" + result.summary())

        # Mostrar mejor código
        print("\n=== Mejor Función de Recompensa ===")
        print(result.best_candidate.reward_code[:500] + "...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
