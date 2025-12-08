"""
Universal Environment - Wrapper dinámico para CUALQUIER dominio robótico

A diferencia de drone_env.py que tenía tamaños de observación/acción fijos,
este wrapper se adapta dinámicamente al DomainSpec.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, Any, Dict
import gymnasium as gym
from gymnasium import spaces

from domain_spec import DomainSpec


class UniversalEnvs:
    """
    Wrapper ctypes para múltiples entornos paralelos.

    Se adapta dinámicamente a cualquier dominio basándose en DomainSpec.
    Compatible con el backend C generado por UniversalCompiler.
    """

    def __init__(
        self,
        num_envs: int,
        domain: DomainSpec,
        library_path: str
    ):
        """
        Inicializa los entornos.

        Args:
            num_envs: Número de entornos paralelos
            domain: Especificación del dominio
            library_path: Path a la librería .so compilada
        """
        self.num_envs = num_envs
        self.domain = domain
        self.obs_size = domain.obs_size
        self.action_size = domain.action_size

        # Cargar librería
        self._lib = ctypes.CDLL(library_path)
        self._setup_ctypes()

        # Crear entornos
        self._envs = self._lib.create_envs(num_envs)
        if not self._envs:
            raise RuntimeError("Error al crear entornos")

        # Configurar buffers
        self._setup_buffers()

        # Reset inicial
        self.reset_all()

    def _setup_ctypes(self):
        """Configura las firmas de funciones ctypes"""
        # create_envs
        self._lib.create_envs.argtypes = [ctypes.c_int]
        self._lib.create_envs.restype = ctypes.c_void_p

        # destroy_envs
        self._lib.destroy_envs.argtypes = [ctypes.c_void_p]
        self._lib.destroy_envs.restype = None

        # reset_env
        self._lib.reset_env.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.reset_env.restype = None

        # reset_all
        self._lib.reset_all.argtypes = [ctypes.c_void_p]
        self._lib.reset_all.restype = None

        # step_all
        self._lib.step_all.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._lib.step_all.restype = None

        # get_observations
        self._lib.get_observations.argtypes = [ctypes.c_void_p]
        self._lib.get_observations.restype = None

        # Punteros a buffers
        self._lib.get_observations_ptr.argtypes = [ctypes.c_void_p]
        self._lib.get_observations_ptr.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.get_rewards_ptr.argtypes = [ctypes.c_void_p]
        self._lib.get_rewards_ptr.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.get_dones_ptr.argtypes = [ctypes.c_void_p]
        self._lib.get_dones_ptr.restype = ctypes.POINTER(ctypes.c_int)

        # Getters de tamaños
        self._lib.get_obs_size.argtypes = []
        self._lib.get_obs_size.restype = ctypes.c_int

        self._lib.get_action_size.argtypes = []
        self._lib.get_action_size.restype = ctypes.c_int

    def _setup_buffers(self):
        """Configura buffers numpy que apuntan a memoria C"""
        # Obtener punteros
        obs_ptr = self._lib.get_observations_ptr(self._envs)
        rewards_ptr = self._lib.get_rewards_ptr(self._envs)
        dones_ptr = self._lib.get_dones_ptr(self._envs)

        # Crear arrays numpy sobre memoria C (zero-copy)
        self.observations = np.ctypeslib.as_array(
            obs_ptr, shape=(self.num_envs, self.obs_size)
        )
        self.rewards = np.ctypeslib.as_array(
            rewards_ptr, shape=(self.num_envs,)
        )
        self.dones = np.ctypeslib.as_array(
            dones_ptr, shape=(self.num_envs,)
        )

        # Buffer de acciones (owned by Python)
        self._actions_buffer = np.zeros(
            (self.num_envs, self.action_size), dtype=np.float32
        )

    def reset_all(self):
        """Resetea todos los entornos"""
        self._lib.reset_all(self._envs)
        self._lib.get_observations(self._envs)

    def reset_env(self, env_idx: int):
        """Resetea un entorno específico"""
        self._lib.reset_env(self._envs, env_idx)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Ejecuta un paso en todos los entornos.

        Args:
            actions: Array de forma (num_envs, action_size) con valores en [-1, 1]

        Returns:
            observations: (num_envs, obs_size)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: dict vacío (compatibilidad)
        """
        # Asegurar formato correcto
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, self.action_size)

        # Copiar acciones al buffer
        np.copyto(self._actions_buffer, actions)

        # Ejecutar paso
        actions_ptr = self._actions_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.step_all(self._envs, actions_ptr)

        # Actualizar observaciones
        self._lib.get_observations(self._envs)

        return self.observations, self.rewards, self.dones.astype(bool), {}

    def close(self):
        """Libera recursos"""
        if hasattr(self, '_envs') and self._envs:
            self._lib.destroy_envs(self._envs)
            self._envs = None

    def __del__(self):
        self.close()


class UniversalGymEnv(gym.Env):
    """
    Wrapper Gymnasium para un solo entorno universal.

    Útil para evaluación y visualización.
    """

    def __init__(
        self,
        domain: DomainSpec,
        library_path: str
    ):
        """
        Inicializa el entorno Gymnasium.

        Args:
            domain: Especificación del dominio
            library_path: Path a la librería compilada
        """
        super().__init__()

        self.domain = domain
        self._envs = UniversalEnvs(1, domain, library_path)

        # Definir espacios
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(domain.obs_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(domain.action_size,),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Resetea el entorno"""
        if seed is not None:
            np.random.seed(seed)

        self._envs.reset_all()
        return self._envs.observations[0].copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta una acción"""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 0:
            action = action.reshape(1)

        obs, rewards, dones, _ = self._envs.step(action.reshape(1, -1))

        return (
            obs[0].copy(),
            float(rewards[0]),
            bool(dones[0]),
            False,  # truncated
            {}
        )

    def render(self):
        """Renderizado (placeholder)"""
        pass

    def close(self):
        """Cierra el entorno"""
        self._envs.close()


class UniversalVecEnv:
    """
    Entorno vectorizado compatible con PufferLib/SB3.

    Diseñado para entrenamiento con PPO.
    """

    def __init__(
        self,
        num_envs: int,
        domain: DomainSpec,
        library_path: str
    ):
        """
        Inicializa el entorno vectorizado.

        Args:
            num_envs: Número de entornos paralelos
            domain: Especificación del dominio
            library_path: Path a la librería compilada
        """
        self.num_envs = num_envs
        self.domain = domain
        self._envs = UniversalEnvs(num_envs, domain, library_path)

        # Espacios
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(domain.obs_size,),
            dtype=np.float32
        )

        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(domain.action_size,),
            dtype=np.float32
        )

        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Resetea todos los entornos"""
        if seed is not None:
            np.random.seed(seed)
        self._envs.reset_all()
        return self._envs.observations.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Ejecuta acciones en todos los entornos.

        Returns:
            obs, rewards, terminateds, truncateds, infos
        """
        obs, rewards, dones, _ = self._envs.step(actions)

        # Gymnasium espera terminateds y truncateds separados
        terminateds = dones.astype(bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)

        infos = [{} for _ in range(self.num_envs)]

        return obs.copy(), rewards.copy(), terminateds, truncateds, infos

    def close(self):
        """Cierra los entornos"""
        self._envs.close()


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_envs(
    domain: DomainSpec,
    library_path: str,
    num_envs: int = 64,
    env_type: str = "vec"
) -> Any:
    """
    Crea entornos del tipo especificado.

    Args:
        domain: Especificación del dominio
        library_path: Path a la librería compilada
        num_envs: Número de entornos (ignorado para gym)
        env_type: "vec" para vectorizado, "gym" para Gymnasium

    Returns:
        Entorno del tipo solicitado
    """
    if env_type == "gym":
        return UniversalGymEnv(domain, library_path)
    elif env_type == "vec":
        return UniversalVecEnv(num_envs, domain, library_path)
    else:
        raise ValueError(f"Tipo de entorno desconocido: {env_type}")


def create_from_domain_name(
    domain_name: str,
    task: str,
    num_envs: int = 64,
    use_mock: bool = True,
    output_dir: str = "c_src"
) -> Tuple[UniversalVecEnv, DomainSpec, str]:
    """
    Crea entornos desde un nombre de dominio.

    Pipeline completo: genera código, compila, y crea entornos.

    Args:
        domain_name: Nombre del dominio (drone, cartpole, etc.)
        task: Descripción de la tarea
        num_envs: Número de entornos
        use_mock: Si usar código mock
        output_dir: Directorio para archivos generados

    Returns:
        (env, domain, library_path)
    """
    from domain_spec import get_domain
    from universal_architect import UniversalArchitect
    from universal_compiler import UniversalCompiler, CompileResult

    # Obtener dominio
    domain = get_domain(domain_name)

    # Generar código
    architect = UniversalArchitect(use_mock=use_mock)
    generated = architect.generate_both(domain, task)

    if not generated.success:
        raise RuntimeError(f"Error generando código: {generated.error_message}")

    # Compilar
    compiler = UniversalCompiler(output_dir)
    output = compiler.compile(domain, generated.physics_code, generated.reward_code)

    if output.result != CompileResult.SUCCESS:
        raise RuntimeError(f"Error compilando: {output.error_message}")

    # Crear entornos
    env = UniversalVecEnv(num_envs, domain, output.library_path)

    return env, domain, output.library_path


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=== Universal Environment Demo ===\n")

    # Crear entorno para CartPole
    print("Creando entorno CartPole...")

    try:
        env, domain, lib_path = create_from_domain_name(
            "cartpole",
            "Mantener el palo vertical",
            num_envs=4,
            use_mock=True
        )

        print(f"Dominio: {domain.name}")
        print(f"Observaciones: {domain.obs_size}")
        print(f"Acciones: {domain.action_size}")
        print(f"Entornos: {env.num_envs}")
        print(f"Librería: {lib_path}")

        # Probar
        print("\nProbando entorno...")
        obs = env.reset()
        print(f"Obs inicial: {obs[0]}")

        for step in range(10):
            actions = np.random.uniform(-1, 1, (env.num_envs, domain.action_size)).astype(np.float32)
            obs, rewards, dones, _, _ = env.step(actions)

            if step == 0:
                print(f"Paso 1 - Reward: {rewards[0]:.3f}, Done: {dones[0]}")

        print("OK - 10 pasos ejecutados")

        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Probar con Drone
    print("\n" + "="*50)
    print("Creando entorno Drone...")

    try:
        env, domain, lib_path = create_from_domain_name(
            "drone",
            "Mantener hover estable a 1 metro",
            num_envs=8,
            use_mock=True
        )

        print(f"Dominio: {domain.name}")
        print(f"Observaciones: {domain.obs_size}")
        print(f"Acciones: {domain.action_size}")

        obs = env.reset()
        print(f"Obs shape: {obs.shape}")

        # Unos pasos
        for _ in range(5):
            actions = np.random.uniform(-1, 1, (env.num_envs, domain.action_size)).astype(np.float32)
            obs, rewards, dones, _, _ = env.step(actions)

        print(f"Rewards promedio: {rewards.mean():.3f}")
        print("OK")

        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
