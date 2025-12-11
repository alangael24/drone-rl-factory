"""
Co-Evolution Engine - Evoluciona DOMINIO + FÍSICA + RECOMPENSA

Combina los conceptos de:
- GenSim: Genera entornos desde lenguaje natural
- Eureka: Evoluciona funciones de recompensa
- DrEureka: Co-evoluciona física y recompensa

El bucle:
1. Genera dominio desde instrucción (Meta-Architect)
2. Genera física inicial
3. Valida física con crítico
4. Si física es mala -> refina física
5. Si física es buena -> evoluciona recompensa (Eureka)
6. Retorna el mejor agente
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from domain_spec import DomainSpec
from domain_generator import DomainGenerator, GeneratedDomain, PhysicsCritique
from universal_architect import UniversalArchitect
from universal_compiler import UniversalCompiler, CompileResult
from universal_env import UniversalVecEnv
from universal_judge import UniversalJudge, TrainingMetrics, JudgmentResult, LearningQuality, generate_semantic_reflection
from simple_ppo import SimplePPO


@dataclass
class CoEvolutionConfig:
    """Configuración del bucle de co-evolución"""
    # Generación de dominio
    max_domain_iterations: int = 3  # Intentos de refinar el dominio

    # Evolución de recompensa
    reward_generations: int = 5
    reward_population: int = 3

    # Entrenamiento
    eval_steps: int = 50000
    num_envs: int = 32

    # Criterios de éxito
    min_acceptable_score: float = 40.0  # Score mínimo para aceptar solución
    physics_confidence_threshold: float = 0.6


@dataclass
class CoEvolutionResult:
    """Resultado del proceso de co-evolución"""
    success: bool
    domain: Optional[DomainSpec]
    physics_code: str
    reward_code: str
    best_score: float
    iterations: int
    history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Co-Evolution Result ===",
            f"Success: {self.success}",
            f"Domain: {self.domain.name if self.domain else 'None'}",
            f"Best Score: {self.best_score:.1f}",
            f"Iterations: {self.iterations}",
        ]
        return "\n".join(lines)


class CoEvolutionEngine:
    """
    Motor de Co-Evolución que genera y optimiza entornos completos.

    Dado: "Una pipa de agua que va de A a B"
    Produce: Dominio + Física + Recompensa optimizados
    """

    def __init__(self, config: Optional[CoEvolutionConfig] = None):
        self.config = config or CoEvolutionConfig()

        # Componentes
        self.domain_generator = DomainGenerator()
        self.architect = UniversalArchitect()
        self.compiler = UniversalCompiler("c_src")
        self.judge = UniversalJudge()

    def evolve_from_instruction(
        self,
        instruction: str,
        verbose: bool = True
    ) -> CoEvolutionResult:
        """
        Punto de entrada principal: genera y optimiza un entorno desde texto.

        Args:
            instruction: Descripción en lenguaje natural (ej: "pipa de agua A->B")
            verbose: Si imprimir progreso

        Returns:
            CoEvolutionResult con el mejor dominio/física/recompensa encontrado
        """
        if verbose:
            print("=" * 70)
            print("CO-EVOLUCIÓN: Generando mundo desde lenguaje natural")
            print("=" * 70)
            print(f"Instrucción: '{instruction}'")
            print()

        history = []
        start_time = time.time()

        # ========================================
        # FASE 1: Generar Dominio Inicial
        # ========================================
        if verbose:
            print("🧠 FASE 1: Generando dominio desde instrucción...")

        generated = self.domain_generator.generate_domain(instruction)

        if not generated.success or not generated.domain:
            return CoEvolutionResult(
                success=False,
                domain=None,
                physics_code="",
                reward_code="",
                best_score=0,
                iterations=0,
                history=[{"phase": "domain_generation", "error": generated.error_message}]
            )

        domain = generated.domain
        physics_code = generated.physics_code
        reward_code = generated.reward_code

        if verbose:
            print(f"   ✅ Dominio: {domain.name}")
            print(f"   ✅ Estado: {len(domain.state_fields)} campos")
            print(f"   ✅ Acciones: {[a.name for a in domain.action_fields]}")

        history.append({
            "phase": "domain_generation",
            "domain": domain.name,
            "state_fields": [f.name for f in domain.state_fields],
        })

        # ========================================
        # FASE 2: Validar y Refinar Física
        # ========================================
        if verbose:
            print("\n🔬 FASE 2: Validando física...")

        for physics_iter in range(self.config.max_domain_iterations):
            # Compilar para probar
            compile_result = self.compiler.compile(
                domain, physics_code, reward_code,
                output_name=f"lib{domain.name.lower()}_coevo_p{physics_iter}"
            )

            if compile_result.result != CompileResult.SUCCESS:
                if verbose:
                    print(f"   ❌ Error compilación: {compile_result.error_message[:50]}")

                # Intentar regenerar física
                if not self.architect.use_mock:
                    physics_result = self.architect.generate_physics(domain)
                    if physics_result.success:
                        physics_code = physics_result.physics_code
                        continue

                history.append({"phase": "physics_validation", "error": "compile_failed"})
                continue

            # Ejecutar dry-run para obtener trayectoria
            trajectory = self._run_dry_simulation(domain, compile_result.library_path)

            # Crítico evalúa la física
            critique = self.domain_generator.critique_physics(domain, physics_code, trajectory)

            if verbose:
                status = "✅" if critique.is_realistic and critique.is_solvable else "⚠️"
                print(f"   {status} Física iter {physics_iter}: realistic={critique.is_realistic}, solvable={critique.is_solvable}")
                if critique.issues:
                    for issue in critique.issues[:2]:
                        print(f"      - {issue}")

            history.append({
                "phase": "physics_validation",
                "iteration": physics_iter,
                "is_realistic": critique.is_realistic,
                "is_solvable": critique.is_solvable,
                "confidence": critique.confidence,
            })

            if critique.is_realistic and critique.is_solvable:
                if verbose:
                    print("   ✅ Física validada!")
                break

            # Refinar si no es buena
            if critique.suggestions and not self.architect.use_mock:
                # El LLM intentaría refinar aquí
                pass

        # ========================================
        # FASE 3: Evolucionar Recompensa (Eureka)
        # ========================================
        if verbose:
            print(f"\n🚀 FASE 3: Evolucionando recompensa ({self.config.reward_generations} generaciones)...")

        best_score = 0.0
        best_reward_code = reward_code

        for gen in range(self.config.reward_generations):
            if verbose:
                print(f"\n   --- Generación {gen}/{self.config.reward_generations} ---")

            # Compilar con la recompensa actual
            compile_result = self.compiler.compile(
                domain, physics_code, reward_code,
                output_name=f"lib{domain.name.lower()}_coevo_g{gen}"
            )

            if compile_result.result != CompileResult.SUCCESS:
                if verbose:
                    print(f"   ❌ Compilación falló")
                continue

            # Entrenar y evaluar
            judgment, metrics = self._train_and_judge(
                domain, compile_result.library_path, verbose
            )

            if judgment:
                score = judgment.score

                if verbose:
                    quality_emoji = "🏆" if judgment.quality == LearningQuality.EXCELLENT else \
                                   "✅" if judgment.quality == LearningQuality.GOOD else \
                                   "⚠️" if judgment.quality == LearningQuality.MEDIOCRE else "❌"
                    print(f"   {quality_emoji} Score: {score:.1f} ({judgment.quality.value})")

                history.append({
                    "phase": "reward_evolution",
                    "generation": gen,
                    "score": score,
                    "quality": judgment.quality.value,
                })

                if score > best_score:
                    best_score = score
                    best_reward_code = reward_code

                # Evolucionar recompensa para siguiente generación
                if gen < self.config.reward_generations - 1:
                    reflection = generate_semantic_reflection(judgment)

                    metrics_dict = {
                        "score": score,
                        "mean_reward": judgment.raw_metrics.get("mean_reward", 0),
                        "rewards": judgment.raw_metrics.get("rewards", [])[-50:],
                        "mean_episode_length": judgment.raw_metrics.get("mean_episode_length", 100),
                        "learning_gain": judgment.learning_gain,
                        "stability": judgment.stability,
                    }

                    evolved = self.architect.evolve_reward(
                        domain, reward_code, metrics_dict, reflection
                    )

                    if evolved.success and "return" in evolved.reward_code:
                        reward_code = evolved.reward_code
                    else:
                        if verbose:
                            print(f"   ⚠️ Evolución falló, manteniendo código anterior")

        # ========================================
        # RESULTADO FINAL
        # ========================================
        total_time = time.time() - start_time
        success = best_score >= self.config.min_acceptable_score

        if verbose:
            print("\n" + "=" * 70)
            print("RESULTADO DE CO-EVOLUCIÓN")
            print("=" * 70)
            print(f"Dominio: {domain.name}")
            print(f"Mejor Score: {best_score:.1f}")
            print(f"Éxito: {'✅ SÍ' if success else '❌ NO'}")
            print(f"Tiempo: {total_time:.1f}s")

        return CoEvolutionResult(
            success=success,
            domain=domain,
            physics_code=physics_code,
            reward_code=best_reward_code,
            best_score=best_score,
            iterations=len(history),
            history=history,
        )

    def _run_dry_simulation(
        self,
        domain: DomainSpec,
        library_path: str,
        steps: int = 200
    ) -> List[Dict[str, float]]:
        """Ejecuta simulación sin entrenar para obtener trayectoria"""
        import numpy as np

        trajectory = []

        try:
            env = UniversalVecEnv(1, domain, library_path)
            obs = env.reset()

            for _ in range(steps):
                # Acciones aleatorias
                actions = np.random.uniform(-1, 1, (1, domain.action_size)).astype(np.float32)
                obs, rewards, dones, truncs, infos = env.step(actions)

                # Guardar estado (asumiendo que obs contiene posiciones)
                state = {}
                if domain.obs_size >= 2:
                    state["x"] = float(obs[0, 0])
                    state["y"] = float(obs[0, 1])
                if domain.obs_size >= 4:
                    state["v_linear"] = float(obs[0, 3]) if domain.obs_size > 3 else 0

                trajectory.append(state)

                if dones[0]:
                    obs = env.reset()

            env.close()
        except Exception as e:
            print(f"Error en dry-run: {e}")

        return trajectory

    def _train_and_judge(
        self,
        domain: DomainSpec,
        library_path: str,
        verbose: bool = True
    ) -> Tuple[Optional[JudgmentResult], Optional[TrainingMetrics]]:
        """Entrena y evalúa con el Juez Universal"""
        try:
            env = UniversalVecEnv(self.config.num_envs, domain, library_path)

            ppo = SimplePPO(env, domain)
            metrics = ppo.train(self.config.eval_steps)

            judgment = self.judge.judge(metrics)

            env.close()
            return judgment, metrics

        except Exception as e:
            if verbose:
                print(f"   Error entrenamiento: {e}")
            return None, None


# ============================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================

def create_world_from_text(
    instruction: str,
    generations: int = 5,
    verbose: bool = True
) -> CoEvolutionResult:
    """
    Crea un mundo robótico completo desde una descripción en texto.

    Ejemplo:
        result = create_world_from_text("una pipa de agua que va de A a B sin derramar")
        if result.success:
            print(f"Score: {result.best_score}")
            print(result.reward_code)
    """
    config = CoEvolutionConfig(
        reward_generations=generations,
        eval_steps=40000,
    )

    engine = CoEvolutionEngine(config)
    return engine.evolve_from_instruction(instruction, verbose)


if __name__ == "__main__":
    print("=== Co-Evolution Demo ===\n")

    # Probar con pipa de agua
    result = create_world_from_text(
        "una pipa de agua que transporta líquido de punto A a punto B",
        generations=3,
        verbose=True
    )

    if result.success:
        print("\n" + "=" * 50)
        print("CÓDIGO DE RECOMPENSA FINAL:")
        print("=" * 50)
        print(result.reward_code)
