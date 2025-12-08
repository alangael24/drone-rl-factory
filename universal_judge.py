"""
Universal Judge - Métricas de aprendizaje agnósticas del dominio

A diferencia de las métricas específicas de drones (distancia, orientación, etc.),
este juez evalúa la CALIDAD DEL APRENDIZAJE sin importar el dominio.

Basado en principios de:
- Eureka (aprendizaje progresivo)
- Curriculum Learning (dificultad apropiada)
- Meta-Learning (transferibilidad)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class LearningQuality(Enum):
    """Clasificación de la calidad del aprendizaje"""
    EXCELLENT = "excellent"      # Aprendizaje rápido y estable
    GOOD = "good"                # Aprendizaje progresivo
    MEDIOCRE = "mediocre"        # Aprendizaje lento o inestable
    POOR = "poor"                # Casi no hay aprendizaje
    FAILED = "failed"            # Sin aprendizaje o divergencia


@dataclass
class TrainingMetrics:
    """Métricas recolectadas durante el entrenamiento"""
    # Recompensas por época/rollout
    rewards: List[float] = field(default_factory=list)

    # Longitud de episodios
    episode_lengths: List[float] = field(default_factory=list)

    # Tasa de éxito (si hay criterio de éxito definido)
    success_rates: List[float] = field(default_factory=list)

    # Entropía de la política
    entropies: List[float] = field(default_factory=list)

    # Pérdida del crítico (value function)
    value_losses: List[float] = field(default_factory=list)

    # Pérdida de política
    policy_losses: List[float] = field(default_factory=list)

    # Métricas adicionales específicas del dominio
    domain_metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_reward(self, reward: float):
        self.rewards.append(reward)

    def add_episode_length(self, length: float):
        self.episode_lengths.append(length)

    def add_success_rate(self, rate: float):
        self.success_rates.append(rate)

    def add_entropy(self, entropy: float):
        self.entropies.append(entropy)


@dataclass
class JudgmentResult:
    """Resultado del juicio sobre el entrenamiento"""
    quality: LearningQuality
    score: float  # 0 a 100

    # Métricas calculadas
    learning_gain: float        # Mejora total de recompensa
    learning_speed: float       # Velocidad de aprendizaje
    stability: float            # Estabilidad (inversa de varianza)
    monotonicity: float         # Qué tan monotónica es la curva
    final_performance: float    # Rendimiento final

    # Diagnóstico
    diagnosis: str
    recommendations: List[str]

    # Datos para evolución
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


class UniversalJudge:
    """
    Juez universal que evalúa la calidad del aprendizaje.

    No importa si es un dron, un brazo robótico, o un CartPole:
    evalúa si el agente APRENDIÓ algo útil.
    """

    def __init__(
        self,
        min_learning_gain: float = 0.1,
        stability_threshold: float = 0.3,
        success_threshold: float = 0.5
    ):
        """
        Inicializa el juez.

        Args:
            min_learning_gain: Ganancia mínima para considerar éxito
            stability_threshold: Umbral de estabilidad (varianza máxima)
            success_threshold: Umbral de tasa de éxito
        """
        self.min_learning_gain = min_learning_gain
        self.stability_threshold = stability_threshold
        self.success_threshold = success_threshold

    def judge(self, metrics: TrainingMetrics) -> JudgmentResult:
        """
        Evalúa las métricas de entrenamiento.

        Args:
            metrics: Métricas recolectadas durante el entrenamiento

        Returns:
            JudgmentResult con el veredicto
        """
        if not metrics.rewards or len(metrics.rewards) < 2:
            return JudgmentResult(
                quality=LearningQuality.FAILED,
                score=0,
                learning_gain=0,
                learning_speed=0,
                stability=0,
                monotonicity=0,
                final_performance=0,
                diagnosis="No hay suficientes datos de entrenamiento",
                recommendations=["Entrenar por más tiempo"],
            )

        rewards = np.array(metrics.rewards)

        # Calcular métricas
        learning_gain = self._calculate_learning_gain(rewards)
        learning_speed = self._calculate_learning_speed(rewards)
        stability = self._calculate_stability(rewards)
        monotonicity = self._calculate_monotonicity(rewards)
        final_performance = self._calculate_final_performance(rewards)

        # Calcular score compuesto
        score = self._calculate_score(
            learning_gain, learning_speed, stability, monotonicity, final_performance
        )

        # Clasificar calidad
        quality = self._classify_quality(score, learning_gain, stability)

        # Generar diagnóstico
        diagnosis = self._generate_diagnosis(
            quality, learning_gain, learning_speed, stability, monotonicity,
            metrics
        )

        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            quality, learning_gain, learning_speed, stability, monotonicity,
            metrics
        )

        return JudgmentResult(
            quality=quality,
            score=score,
            learning_gain=learning_gain,
            learning_speed=learning_speed,
            stability=stability,
            monotonicity=monotonicity,
            final_performance=final_performance,
            diagnosis=diagnosis,
            recommendations=recommendations,
            raw_metrics={
                "rewards": rewards.tolist(),
                "mean_reward": float(np.mean(rewards)),
                "final_reward": float(rewards[-1]) if len(rewards) > 0 else 0,
                "mean_episode_length": float(np.mean(metrics.episode_lengths)) if metrics.episode_lengths else 0,
                "success_rate": float(metrics.success_rates[-1]) if metrics.success_rates else 0,
            }
        )

    def _calculate_learning_gain(self, rewards: np.ndarray) -> float:
        """
        Calcula cuánto mejoró la recompensa durante el entrenamiento.

        Returns:
            Valor normalizado entre 0 y 1 (o más si mejoró mucho)
        """
        if len(rewards) < 2:
            return 0

        # Usar promedios móviles para suavizar
        window = min(10, len(rewards) // 4)
        if window < 1:
            window = 1

        initial = np.mean(rewards[:window])
        final = np.mean(rewards[-window:])

        # Normalizar por el rango de recompensas
        reward_range = np.max(rewards) - np.min(rewards)
        if reward_range < 0.001:
            return 0

        gain = (final - initial) / (reward_range + 1e-8)
        return float(np.clip(gain, -1, 2))  # Permitir hasta 200% de ganancia

    def _calculate_learning_speed(self, rewards: np.ndarray) -> float:
        """
        Calcula qué tan rápido aprendió el agente.

        Returns:
            Valor entre 0 y 1 (1 = convergencia inmediata)
        """
        if len(rewards) < 10:
            return 0

        # Encontrar cuándo llegó al 90% del rendimiento final
        final_level = np.mean(rewards[-10:])
        threshold = np.mean(rewards[:10]) + 0.9 * (final_level - np.mean(rewards[:10]))

        # Buscar primer índice donde supera el threshold
        indices = np.where(rewards >= threshold)[0]
        if len(indices) == 0:
            return 0

        first_success = indices[0]

        # Normalizar por longitud total
        speed = 1.0 - (first_success / len(rewards))
        return float(np.clip(speed, 0, 1))

    def _calculate_stability(self, rewards: np.ndarray) -> float:
        """
        Calcula la estabilidad del entrenamiento (inversa de varianza).

        Returns:
            Valor entre 0 y 1 (1 = muy estable)
        """
        if len(rewards) < 10:
            return 0.5

        # Calcular varianza de las últimas iteraciones
        last_portion = rewards[-len(rewards)//3:]
        variance = np.var(last_portion)

        # Normalizar por el rango
        mean_abs = np.mean(np.abs(rewards))
        if mean_abs < 0.001:
            return 0.5

        normalized_var = variance / (mean_abs ** 2 + 1e-8)

        # Convertir a estabilidad (inversa)
        stability = 1.0 / (1.0 + normalized_var * 10)
        return float(np.clip(stability, 0, 1))

    def _calculate_monotonicity(self, rewards: np.ndarray) -> float:
        """
        Calcula qué tan monotónicamente creciente es la curva.

        Returns:
            Valor entre 0 y 1 (1 = siempre creciente)
        """
        if len(rewards) < 3:
            return 0.5

        # Suavizar con promedio móvil
        window = max(3, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        if len(smoothed) < 2:
            return 0.5

        # Calcular diferencias
        diffs = np.diff(smoothed)

        # Proporción de incrementos positivos
        positive_ratio = np.sum(diffs > 0) / len(diffs)

        return float(positive_ratio)

    def _calculate_final_performance(self, rewards: np.ndarray) -> float:
        """
        Evalúa el rendimiento final absoluto.

        Returns:
            Valor normalizado
        """
        if len(rewards) < 5:
            return 0

        final = np.mean(rewards[-5:])

        # Normalizar al rango típico de recompensas [-10, 10]
        normalized = (final + 10) / 20  # Mapea [-10, 10] a [0, 1]
        return float(np.clip(normalized, 0, 1))

    def _calculate_score(
        self,
        learning_gain: float,
        learning_speed: float,
        stability: float,
        monotonicity: float,
        final_performance: float
    ) -> float:
        """
        Calcula score compuesto (0-100).

        Pesos:
        - Learning gain: 35% (lo más importante)
        - Stability: 25%
        - Final performance: 20%
        - Monotonicity: 10%
        - Learning speed: 10%
        """
        score = (
            learning_gain * 35 +
            stability * 25 +
            final_performance * 20 +
            monotonicity * 10 +
            learning_speed * 10
        )

        return float(np.clip(score, 0, 100))

    def _classify_quality(
        self,
        score: float,
        learning_gain: float,
        stability: float
    ) -> LearningQuality:
        """Clasifica la calidad del aprendizaje"""
        if score >= 70 and learning_gain > 0.5 and stability > 0.6:
            return LearningQuality.EXCELLENT
        elif score >= 50 and learning_gain > 0.2:
            return LearningQuality.GOOD
        elif score >= 30 and learning_gain > 0:
            return LearningQuality.MEDIOCRE
        elif learning_gain > -0.1:
            return LearningQuality.POOR
        else:
            return LearningQuality.FAILED

    def _generate_diagnosis(
        self,
        quality: LearningQuality,
        learning_gain: float,
        learning_speed: float,
        stability: float,
        monotonicity: float,
        metrics: TrainingMetrics
    ) -> str:
        """Genera diagnóstico en lenguaje natural"""
        parts = []

        # Resumen general
        if quality == LearningQuality.EXCELLENT:
            parts.append("El entrenamiento fue EXCELENTE.")
        elif quality == LearningQuality.GOOD:
            parts.append("El entrenamiento fue BUENO.")
        elif quality == LearningQuality.MEDIOCRE:
            parts.append("El entrenamiento fue MEDIOCRE.")
        elif quality == LearningQuality.POOR:
            parts.append("El entrenamiento fue POBRE.")
        else:
            parts.append("El entrenamiento FALLÓ.")

        # Detalles de ganancia
        if learning_gain > 0.5:
            parts.append(f"La recompensa mejoró significativamente ({learning_gain*100:.0f}%).")
        elif learning_gain > 0.1:
            parts.append(f"Hubo mejora moderada ({learning_gain*100:.0f}%).")
        elif learning_gain > 0:
            parts.append(f"Hubo mejora mínima ({learning_gain*100:.0f}%).")
        elif learning_gain > -0.1:
            parts.append("No hubo mejora significativa.")
        else:
            parts.append(f"La recompensa EMPEORÓ ({learning_gain*100:.0f}%).")

        # Estabilidad
        if stability > 0.8:
            parts.append("El aprendizaje fue muy estable.")
        elif stability > 0.5:
            parts.append("El aprendizaje tuvo estabilidad moderada.")
        elif stability > 0.3:
            parts.append("El aprendizaje fue inestable.")
        else:
            parts.append("El aprendizaje fue MUY inestable (alta varianza).")

        # Episodios
        if metrics.episode_lengths:
            mean_length = np.mean(metrics.episode_lengths)
            if mean_length < 20:
                parts.append(f"Los episodios terminan muy rápido ({mean_length:.0f} pasos).")
            elif mean_length > 400:
                parts.append(f"Los episodios duran mucho ({mean_length:.0f} pasos).")

        return " ".join(parts)

    def _generate_recommendations(
        self,
        quality: LearningQuality,
        learning_gain: float,
        learning_speed: float,
        stability: float,
        monotonicity: float,
        metrics: TrainingMetrics
    ) -> List[str]:
        """Genera recomendaciones para mejorar"""
        recommendations = []

        # Basado en ganancia
        if learning_gain <= 0:
            recommendations.append(
                "La recompensa no mejora. Considera simplificar drásticamente "
                "la función de recompensa o verificar que la tarea es posible."
            )
        elif learning_gain < 0.2:
            recommendations.append(
                "La mejora es muy lenta. Aumenta el gradiente de recompensa "
                "por acercarse al objetivo."
            )

        # Basado en estabilidad
        if stability < 0.3:
            recommendations.append(
                "Alta varianza. Reduce el learning rate o aumenta el tamaño "
                "del batch. También considera suavizar las penalizaciones."
            )

        # Basado en longitud de episodios
        if metrics.episode_lengths:
            mean_length = np.mean(metrics.episode_lengths)
            if mean_length < 20:
                recommendations.append(
                    "Los episodios terminan muy rápido. Prioriza la supervivencia: "
                    "reduce penalizaciones y aumenta recompensa por sobrevivir."
                )
            elif mean_length > 450:
                recommendations.append(
                    "Los episodios son muy largos. Considera añadir bonus "
                    "por completar la tarea rápidamente."
                )

        # Basado en entropía (si está disponible)
        if metrics.entropies:
            final_entropy = np.mean(metrics.entropies[-10:])
            if final_entropy < 0.1:
                recommendations.append(
                    "La política tiene muy baja entropía (no explora). "
                    "Aumenta el coeficiente de entropía."
                )
            elif final_entropy > 2.0:
                recommendations.append(
                    "La política es demasiado aleatoria. "
                    "Reduce el coeficiente de entropía o entrena más tiempo."
                )

        # Si no hay recomendaciones específicas
        if not recommendations:
            if quality == LearningQuality.EXCELLENT:
                recommendations.append(
                    "El entrenamiento va bien. Considera aplicar Domain "
                    "Randomization para robustez."
                )
            else:
                recommendations.append(
                    "Entrena por más tiempo y monitorea las métricas."
                )

        return recommendations


def generate_semantic_reflection(judgment: JudgmentResult) -> str:
    """
    Genera reflexión semántica para el LLM basada en el juicio.

    Esta es la pieza clave que conecta el juez con el arquitecto
    para el bucle evolutivo.
    """
    reflection_parts = []

    # Encabezado
    reflection_parts.append(f"## Diagnóstico de Entrenamiento")
    reflection_parts.append(f"Calidad: {judgment.quality.value.upper()}")
    reflection_parts.append(f"Score: {judgment.score:.1f}/100")
    reflection_parts.append("")

    # Métricas
    reflection_parts.append("### Métricas Clave:")
    reflection_parts.append(f"- Ganancia de aprendizaje: {judgment.learning_gain*100:.1f}%")
    reflection_parts.append(f"- Velocidad de aprendizaje: {judgment.learning_speed*100:.1f}%")
    reflection_parts.append(f"- Estabilidad: {judgment.stability*100:.1f}%")
    reflection_parts.append(f"- Monotonicidad: {judgment.monotonicity*100:.1f}%")
    reflection_parts.append(f"- Rendimiento final: {judgment.final_performance*100:.1f}%")
    reflection_parts.append("")

    # Diagnóstico
    reflection_parts.append("### Diagnóstico:")
    reflection_parts.append(judgment.diagnosis)
    reflection_parts.append("")

    # Recomendaciones (lo más importante para el LLM)
    reflection_parts.append("### Cambios Requeridos en la Función de Recompensa:")
    for i, rec in enumerate(judgment.recommendations, 1):
        reflection_parts.append(f"{i}. {rec}")

    return "\n".join(reflection_parts)


# ============================================================
# COLLECTOR - Recolector de métricas durante entrenamiento
# ============================================================

class MetricsCollector:
    """
    Recolecta métricas durante el entrenamiento para el juez.

    Se integra con el loop de PPO para capturar datos relevantes.
    """

    def __init__(self):
        self.metrics = TrainingMetrics()
        self._current_episode_rewards = []
        self._current_episode_length = 0

    def on_step(self, reward: float, done: bool):
        """Llamado en cada paso del entorno"""
        self._current_episode_rewards.append(reward)
        self._current_episode_length += 1

        if done:
            # Episodio terminó
            episode_reward = sum(self._current_episode_rewards)
            self.metrics.rewards.append(episode_reward)
            self.metrics.episode_lengths.append(self._current_episode_length)

            # Reset
            self._current_episode_rewards = []
            self._current_episode_length = 0

    def on_rollout_end(self, mean_reward: float):
        """Llamado al final de cada rollout de PPO"""
        if not self._current_episode_rewards:
            # Si no hay episodio en curso, usar mean_reward
            self.metrics.rewards.append(mean_reward)

    def on_train_step(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: Optional[float] = None
    ):
        """Llamado después de cada paso de optimización"""
        self.metrics.policy_losses.append(policy_loss)
        self.metrics.value_losses.append(value_loss)
        if entropy is not None:
            self.metrics.entropies.append(entropy)

    def add_success(self, success_rate: float):
        """Añade tasa de éxito (definida por el dominio)"""
        self.metrics.success_rates.append(success_rate)

    def add_domain_metric(self, name: str, value: float):
        """Añade métrica específica del dominio"""
        if name not in self.metrics.domain_metrics:
            self.metrics.domain_metrics[name] = []
        self.metrics.domain_metrics[name].append(value)

    def get_metrics(self) -> TrainingMetrics:
        """Retorna las métricas recolectadas"""
        return self.metrics

    def reset(self):
        """Reinicia el colector"""
        self.metrics = TrainingMetrics()
        self._current_episode_rewards = []
        self._current_episode_length = 0


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=== Universal Judge Demo ===\n")

    # Simular métricas de un entrenamiento bueno
    print("Caso 1: Entrenamiento exitoso")
    metrics_good = TrainingMetrics()

    # Simular curva de aprendizaje creciente
    for i in range(100):
        reward = -5 + (i / 100) * 10 + np.random.randn() * 0.5
        metrics_good.rewards.append(reward)
        metrics_good.episode_lengths.append(100 + i * 3)

    judge = UniversalJudge()
    result = judge.judge(metrics_good)

    print(f"Calidad: {result.quality.value}")
    print(f"Score: {result.score:.1f}")
    print(f"Diagnóstico: {result.diagnosis}")
    print(f"Recomendaciones: {result.recommendations}")

    print("\n" + "="*50 + "\n")

    # Simular métricas de un entrenamiento fallido
    print("Caso 2: Entrenamiento fallido")
    metrics_bad = TrainingMetrics()

    # Simular recompensas que no mejoran
    for i in range(100):
        reward = -5 + np.random.randn() * 2
        metrics_bad.rewards.append(reward)
        metrics_bad.episode_lengths.append(10 + np.random.randint(0, 5))

    result = judge.judge(metrics_bad)

    print(f"Calidad: {result.quality.value}")
    print(f"Score: {result.score:.1f}")
    print(f"Diagnóstico: {result.diagnosis}")
    print(f"Recomendaciones: {result.recommendations}")

    print("\n" + "="*50 + "\n")

    # Mostrar reflexión semántica
    print("Reflexión Semántica para el LLM:")
    print(generate_semantic_reflection(result))
