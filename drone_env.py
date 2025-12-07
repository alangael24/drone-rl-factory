"""
Sprint 3: Drone Environment - Python-C Bridge con ctypes
Wrapper para el kernel de física en C, compatible con PufferLib.
"""

import ctypes
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces


# Cargar la librería C
_LIB_PATH = Path(__file__).parent / "c_src" / "libdrone.so"


class DroneEnvs:
    """
    Wrapper ctypes para los entornos de drones en C.
    Maneja múltiples entornos en paralelo para máxima velocidad.
    """

    def __init__(self, num_envs: int = 1):
        """
        Inicializa los entornos.

        Args:
            num_envs: Número de entornos paralelos
        """
        self.num_envs = num_envs
        self.obs_size = 15  # Tamaño del vector de observación
        self.action_size = 4  # thrust, roll, pitch, yaw

        # Cargar librería C
        if not _LIB_PATH.exists():
            raise FileNotFoundError(
                f"Librería C no encontrada: {_LIB_PATH}\n"
                "Ejecuta: cd c_src && gcc -O3 -fPIC -shared -o libdrone.so drone_dynamics.c -lm"
            )

        self._lib = ctypes.CDLL(str(_LIB_PATH))
        self._setup_ctypes()

        # Crear entornos en C
        self._envs = self._lib.create_envs(num_envs)

        # Buffers numpy que apuntan a memoria C
        self._setup_buffers()

    def _setup_ctypes(self):
        """Configura las firmas de funciones ctypes."""
        # create_envs
        self._lib.create_envs.argtypes = [ctypes.c_int]
        self._lib.create_envs.restype = ctypes.c_void_p

        # destroy_envs
        self._lib.destroy_envs.argtypes = [ctypes.c_void_p]
        self._lib.destroy_envs.restype = None

        # reset_all
        self._lib.reset_all.argtypes = [ctypes.c_void_p]
        self._lib.reset_all.restype = None

        # reset_env
        self._lib.reset_env.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.reset_env.restype = None

        # step_all
        self._lib.step_all.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
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

        # Domain Randomization (DrEureka)
        self._lib.set_domain_randomization.argtypes = [
            ctypes.c_void_p,  # envs
            ctypes.c_float,   # mass_min
            ctypes.c_float,   # mass_max
            ctypes.c_float,   # drag_min
            ctypes.c_float,   # drag_max
            ctypes.c_float,   # wind_max
            ctypes.c_float,   # gravity_var
            ctypes.c_float,   # motor_noise_max
            ctypes.c_int      # enabled
        ]
        self._lib.set_domain_randomization.restype = None

    def _setup_buffers(self):
        """Crea arrays numpy que apuntan a la memoria C."""
        # Obtener punteros
        obs_ptr = self._lib.get_observations_ptr(self._envs)
        rewards_ptr = self._lib.get_rewards_ptr(self._envs)
        dones_ptr = self._lib.get_dones_ptr(self._envs)

        # Crear arrays numpy desde punteros C
        self.observations = np.ctypeslib.as_array(
            obs_ptr, shape=(self.num_envs, self.obs_size)
        )
        self.rewards = np.ctypeslib.as_array(
            rewards_ptr, shape=(self.num_envs,)
        )
        self.dones = np.ctypeslib.as_array(
            dones_ptr, shape=(self.num_envs,)
        )

        # Buffer para acciones
        self._actions_buffer = np.zeros(
            (self.num_envs, self.action_size), dtype=np.float32
        )

    def reset(self):
        """Resetea todos los entornos."""
        self._lib.reset_all(self._envs)
        self._lib.get_observations(self._envs)
        return self.observations.copy()

    def set_domain_randomization(
        self,
        mass_min: float = 0.8,
        mass_max: float = 1.2,
        drag_min: float = 0.05,
        drag_max: float = 0.2,
        wind_max: float = 1.0,
        gravity_var: float = 0.05,
        motor_noise_max: float = 0.1,
        enabled: bool = True
    ):
        """
        Configura Domain Randomization (DrEureka).

        Args:
            mass_min: Masa mínima del dron (kg)
            mass_max: Masa máxima del dron (kg)
            drag_min: Coeficiente de arrastre mínimo
            drag_max: Coeficiente de arrastre máximo
            wind_max: Fuerza máxima de viento (N)
            gravity_var: Variación de gravedad (±porcentaje)
            motor_noise_max: Ruido máximo en motores (±porcentaje)
            enabled: Si activar Domain Randomization
        """
        self._lib.set_domain_randomization(
            self._envs,
            ctypes.c_float(mass_min),
            ctypes.c_float(mass_max),
            ctypes.c_float(drag_min),
            ctypes.c_float(drag_max),
            ctypes.c_float(wind_max),
            ctypes.c_float(gravity_var),
            ctypes.c_float(motor_noise_max),
            ctypes.c_int(1 if enabled else 0)
        )

    def step(self, actions: np.ndarray):
        """
        Ejecuta un paso en todos los entornos.

        Args:
            actions: Array de forma (num_envs, 4) con acciones en [-1, 1]

        Returns:
            (observations, rewards, dones, infos)
        """
        # Asegurar formato correcto
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_size):
            actions = actions.reshape(self.num_envs, self.action_size)

        # Copiar al buffer contiguo
        np.copyto(self._actions_buffer, actions)

        # Llamar a C
        actions_ptr = self._actions_buffer.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        self._lib.step_all(self._envs, actions_ptr)

        # Obtener observaciones actualizadas
        self._lib.get_observations(self._envs)

        # Copiar resultados (los arrays apuntan a memoria C)
        obs = self.observations.copy()
        rewards = self.rewards.copy()
        dones = self.dones.astype(bool).copy()
        truncated = np.zeros(self.num_envs, dtype=bool)  # No usamos truncation

        infos = [{} for _ in range(self.num_envs)]

        return obs, rewards, dones, truncated, infos

    def close(self):
        """Libera memoria C."""
        if hasattr(self, '_envs') and self._envs:
            self._lib.destroy_envs(self._envs)
            self._envs = None

    def __del__(self):
        self.close()


class DroneGymEnv(gym.Env):
    """
    Wrapper Gymnasium para un solo entorno de drone.
    Compatible con la API estándar de Gymnasium.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self._envs = DroneEnvs(num_envs=1)

        # Espacios de observación y acción
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._envs.reset()
        return obs[0], {}

    def step(self, action):
        actions = np.array([action], dtype=np.float32)
        obs, rewards, dones, truncated, infos = self._envs.step(actions)
        return obs[0], rewards[0], dones[0], truncated[0], infos[0]

    def close(self):
        self._envs.close()


class DroneVecEnv:
    """
    Entorno vectorizado para PufferLib.
    Implementa la interfaz que PufferLib espera.
    """

    def __init__(self, num_envs: int = 64):
        self.num_envs = num_envs
        self._envs = DroneEnvs(num_envs=num_envs)

        # Espacios
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Para PufferLib
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_envs, 15),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_envs, 4),
            dtype=np.float32
        )

    def reset(self, seed=None):
        obs = self._envs.reset()
        return obs, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        return self._envs.step(actions)

    def close(self):
        self._envs.close()


def make_env(num_envs: int = 1):
    """Factory function para crear entornos."""
    if num_envs == 1:
        return DroneGymEnv()
    else:
        return DroneVecEnv(num_envs=num_envs)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=== Test del puente Python-C ===\n")

    # Test con entorno único
    print("1. Probando entorno único...")
    env = DroneGymEnv()
    obs, info = env.reset()
    print(f"   Observación inicial: shape={obs.shape}")
    print(f"   Valores: {obs[:5]}...")

    # Ejecutar algunos pasos
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            print(f"   Episodio terminó en paso {step}")
            obs, info = env.reset()

    print(f"   Recompensa total (100 pasos): {total_reward:.2f}")
    env.close()

    # Test con entornos paralelos
    print("\n2. Probando entornos paralelos...")
    num_envs = 64
    vec_env = DroneVecEnv(num_envs=num_envs)
    obs, infos = vec_env.reset()
    print(f"   Observaciones: shape={obs.shape}")

    # Benchmark
    import time
    steps = 10000
    start = time.time()

    for _ in range(steps):
        actions = np.random.uniform(-1, 1, (num_envs, 4)).astype(np.float32)
        obs, rewards, dones, truncated, infos = vec_env.step(actions)

    elapsed = time.time() - start
    total_steps = steps * num_envs
    steps_per_sec = total_steps / elapsed

    print(f"   {total_steps:,} pasos en {elapsed:.2f}s")
    print(f"   Velocidad: {steps_per_sec:,.0f} pasos/segundo")

    vec_env.close()
    print("\n   Test completado!")
