"""
Universal Compiler - Compila código C para CUALQUIER dominio robótico

A diferencia del compiler.py original que tenía un template fijo para drones,
este compilador genera templates dinámicos basados en DomainSpec.
"""

import os
import subprocess
import tempfile
import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from domain_spec import DomainSpec


class CompileResult(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    LINK_ERROR = "link_error"
    RUNTIME_ERROR = "runtime_error"


@dataclass
class CompileOutput:
    """Resultado de la compilación"""
    result: CompileResult
    library_path: str
    error_message: str = ""
    generated_code: str = ""


# ============================================================
# TEMPLATE UNIVERSAL DE CÓDIGO C
# ============================================================

UNIVERSAL_C_TEMPLATE = """/**
 * Universal Simulation Engine - Generated for: {domain_name}
 *
 * GENERADO AUTOMÁTICAMENTE - NO EDITAR MANUALMENTE
 *
 * Dominio: {domain_description}
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// CONSTANTES
// ============================================================

{physics_constants}

// ============================================================
// ESTRUCTURAS DE DATOS
// ============================================================

// Estado del sistema
{state_struct}

// Configuración de Domain Randomization
{dr_struct}

// Contenedor de entornos paralelos
typedef struct {{
    int num_envs;
    {state_struct_name}* states;
    float* observations;
    float* rewards;
    int* dones;
    {dr_config_declaration}
}} UniversalEnvs;

// ============================================================
// FUNCIONES AUXILIARES
// ============================================================

static inline float rand_uniform() {{
    return (float)rand() / (float)RAND_MAX;
}}

static inline float rand_range(float min_val, float max_val) {{
    return min_val + rand_uniform() * (max_val - min_val);
}}

static inline float clamp(float value, float min_val, float max_val) {{
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}}

// ============================================================
// DOMAIN RANDOMIZATION
// ============================================================

{dr_apply_code}

{dr_set_code}

// ============================================================
// FUNCIÓN DE FÍSICA (GENERADA O PROPORCIONADA)
// ============================================================

{physics_function}

// ============================================================
// FUNCIÓN DE RECOMPENSA (GENERADA POR LLM)
// ============================================================

{reward_function}

// ============================================================
// FUNCIÓN DE VERIFICACIÓN SEMÁNTICA (GENERADA POR LLM)
// ============================================================

{verify_function}

// ============================================================
// API PÚBLICA
// ============================================================

/**
 * Crea un conjunto de entornos paralelos
 */
UniversalEnvs* create_envs(int num_envs) {{
    UniversalEnvs* envs = (UniversalEnvs*)malloc(sizeof(UniversalEnvs));
    envs->num_envs = num_envs;
    envs->states = ({state_struct_name}*)calloc(num_envs, sizeof({state_struct_name}));
    envs->observations = (float*)calloc(num_envs * {obs_size}, sizeof(float));
    envs->rewards = (float*)calloc(num_envs, sizeof(float));
    envs->dones = (int*)calloc(num_envs, sizeof(int));

    {dr_init_code}

    // Inicializar estados
    for (int i = 0; i < num_envs; i++) {{
        {state_init_code}
    }}

    return envs;
}}

/**
 * Libera memoria
 */
void destroy_envs(UniversalEnvs* envs) {{
    if (envs) {{
        free(envs->states);
        free(envs->observations);
        free(envs->rewards);
        free(envs->dones);
        free(envs);
    }}
}}

/**
 * Resetea un entorno específico
 */
void reset_env(UniversalEnvs* envs, int env_idx) {{
    {state_struct_name}* state = &envs->states[env_idx];
    memset(state, 0, sizeof({state_struct_name}));

    // Inicialización específica del dominio
    {reset_code}

    // Aplicar Domain Randomization
    {dr_apply_call}
}}

/**
 * Resetea todos los entornos
 */
void reset_all(UniversalEnvs* envs) {{
    for (int i = 0; i < envs->num_envs; i++) {{
        reset_env(envs, i);
    }}
}}

/**
 * Ejecuta un paso en todos los entornos
 */
void step_all(UniversalEnvs* envs, float* actions) {{
    for (int i = 0; i < envs->num_envs; i++) {{
        {state_struct_name}* state = &envs->states[i];

        // Ejecutar física
        physics_step(state, &actions[i * {action_size}]);

        // Calcular recompensa
        envs->rewards[i] = calculate_reward(state);
        state->total_reward += envs->rewards[i];

        // Verificar terminación
        {termination_code}

        envs->dones[i] = done;

        // Auto-reset si terminó
        if (done) {{
            reset_env(envs, i);
        }}
    }}
}}

/**
 * Obtiene las observaciones para todos los entornos
 */
void get_observations(UniversalEnvs* envs) {{
    for (int i = 0; i < envs->num_envs; i++) {{
        {state_struct_name}* state = &envs->states[i];
        float* obs = &envs->observations[i * {obs_size}];

{observation_code}
    }}
}}

/**
 * Verifica si un estado es semánticamente válido
 */
int verify_state({state_struct_name}* state) {{
    if (!state) return 0;
    return verify_domain_physics(state);
}}

// Acceso a buffers (para ctypes)
float* get_observations_ptr(UniversalEnvs* envs) {{ return envs->observations; }}
float* get_rewards_ptr(UniversalEnvs* envs) {{ return envs->rewards; }}
int* get_dones_ptr(UniversalEnvs* envs) {{ return envs->dones; }}
int get_num_envs(UniversalEnvs* envs) {{ return envs->num_envs; }}
int get_obs_size() {{ return {obs_size}; }}
int get_action_size() {{ return {action_size}; }}
"""


class UniversalCompiler:
    """
    Compilador universal que genera código C para cualquier dominio.

    Características:
    - Genera templates dinámicos basados en DomainSpec
    - Inyecta funciones de física y recompensa
    - Compila a librería compartida (.so)
    - Valida código antes de compilar
    """

    def __init__(self, output_dir: str = "c_src"):
        """
        Inicializa el compilador.

        Args:
            output_dir: Directorio para guardar archivos generados
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_dr_struct(self, domain: DomainSpec) -> str:
        """Genera el struct de Domain Randomization"""
        if not domain.dr_params:
            return "// No hay parámetros de Domain Randomization"

        return domain.generate_dr_struct()

    def _generate_dr_config_declaration(self, domain: DomainSpec) -> str:
        """Genera la declaración de la config de DR en UniversalEnvs"""
        if not domain.dr_params:
            return "int dr_enabled;  // Placeholder"
        return "DomainRandomConfig dr_config;"

    def _generate_dr_apply(self, domain: DomainSpec) -> str:
        """Genera la función de aplicar DR"""
        if not domain.dr_params:
            return """void apply_domain_randomization(void* state, void* config) {
    // No hay Domain Randomization configurada
}"""
        return domain.generate_dr_apply_code()

    def _generate_dr_set(self, domain: DomainSpec) -> str:
        """Genera la función para configurar DR"""
        if not domain.dr_params:
            return """void set_domain_randomization(UniversalEnvs* envs, int enabled) {
    envs->dr_enabled = enabled;
}"""

        params = ", ".join(
            f"float {dr.name}_min, float {dr.name}_max"
            for dr in domain.dr_params
        )

        body_lines = [
            f"void set_domain_randomization(UniversalEnvs* envs, {params}, int enabled) {{"
        ]

        for dr in domain.dr_params:
            body_lines.append(f"    envs->dr_config.{dr.name}_min = {dr.name}_min;")
            body_lines.append(f"    envs->dr_config.{dr.name}_max = {dr.name}_max;")

        body_lines.append("    envs->dr_config.enabled = enabled;")
        body_lines.append("}")

        return "\n".join(body_lines)

    def _generate_dr_init(self, domain: DomainSpec) -> str:
        """Genera código de inicialización de DR"""
        if not domain.dr_params:
            return "envs->dr_enabled = 0;"

        lines = []
        for dr in domain.dr_params:
            lines.append(f"    envs->dr_config.{dr.name}_min = {dr.min_value}f;")
            lines.append(f"    envs->dr_config.{dr.name}_max = {dr.max_value}f;")
        lines.append("    envs->dr_config.enabled = 0;")

        return "\n".join(lines)

    def _generate_dr_apply_call(self, domain: DomainSpec) -> str:
        """Genera la llamada a apply_domain_randomization"""
        if not domain.dr_params:
            return "// No DR"
        return "apply_domain_randomization(state, &envs->dr_config);"

    def _generate_state_init(self, domain: DomainSpec) -> str:
        """Genera código de inicialización de estado"""
        lines = []
        for field in domain.state_fields:
            if field.default_value != 0.0:
                lines.append(f"        envs->states[i].{field.name} = {field.default_value}f;")

        if domain.dr_params:
            lines.append("        apply_domain_randomization(&envs->states[i], &envs->dr_config);")

        return "\n".join(lines) if lines else "        // Estado inicializado a cero"

    def _generate_reset_code(self, domain: DomainSpec) -> str:
        """Genera código de reset específico del dominio"""
        # Código genérico de reset - puede ser customizado por dominio
        lines = []

        # Buscar campos de posición para randomizar
        for field in domain.state_fields:
            if "target" in field.name.lower():
                # Targets se resetean a valores específicos
                if "x" in field.name.lower():
                    lines.append(f"    state->{field.name} = rand_range(-2.0f, 2.0f);")
                elif "y" in field.name.lower():
                    lines.append(f"    state->{field.name} = rand_range(-2.0f, 2.0f);")
                elif "z" in field.name.lower():
                    lines.append(f"    state->{field.name} = 1.0f;")
            elif field.name in ["x", "y"]:
                # Posición inicial ligeramente aleatoria
                lines.append(f"    state->{field.name} = (rand_uniform() - 0.5f) * 2.0f;")
            elif field.name == "z":
                lines.append(f"    state->{field.name} = 0.1f;")

        return "\n".join(lines) if lines else "    // Reset por defecto"

    def generate_c_code(
        self,
        domain: DomainSpec,
        physics_code: str,
        reward_code: str,
        verify_code: str = ""
    ) -> str:
        """
        Genera el código C completo para un dominio.

        Args:
            domain: Especificación del dominio
            physics_code: Función physics_step generada
            reward_code: Función calculate_reward generada
            verify_code: Función verify_domain_physics (verificación semántica)

        Returns:
            Código C completo listo para compilar
        """
        # Generar componentes
        state_struct = domain.generate_state_struct()
        physics_constants = domain.physics_constants.to_c_defines()
        observation_code = domain.generate_observation_code()
        termination_code = domain.generate_termination_code()

        # Domain Randomization
        dr_struct = self._generate_dr_struct(domain)
        dr_config_decl = self._generate_dr_config_declaration(domain)
        dr_apply = self._generate_dr_apply(domain)
        dr_set = self._generate_dr_set(domain)
        dr_init = self._generate_dr_init(domain)
        dr_apply_call = self._generate_dr_apply_call(domain)

        # Inicialización y reset
        state_init = self._generate_state_init(domain)
        reset_code = self._generate_reset_code(domain)

        # Si no se proporciona verify_code, usar stub genérico
        if not verify_code:
            verify_code = f"""int verify_domain_physics({domain.state_struct_name}* state) {{
    // Verificación genérica: aceptar siempre
    return 1;
}}"""

        # Llenar template
        code = UNIVERSAL_C_TEMPLATE.format(
            domain_name=domain.name,
            domain_description=domain.description,
            physics_constants=physics_constants,
            state_struct=state_struct,
            state_struct_name=domain.state_struct_name,
            dr_struct=dr_struct,
            dr_config_declaration=dr_config_decl,
            dr_apply_code=dr_apply,
            dr_set_code=dr_set,
            dr_init_code=dr_init,
            dr_apply_call=dr_apply_call,
            state_init_code=state_init,
            reset_code=reset_code,
            physics_function=physics_code,
            reward_function=reward_code,
            verify_function=verify_code,
            obs_size=domain.obs_size,
            action_size=domain.action_size,
            observation_code=observation_code,
            termination_code=termination_code,
        )

        return code

    def compile(
        self,
        domain: DomainSpec,
        physics_code: str,
        reward_code: str,
        verify_code: str = "",
        output_name: Optional[str] = None
    ) -> CompileOutput:
        """
        Genera y compila código C para un dominio.

        Args:
            domain: Especificación del dominio
            physics_code: Función physics_step
            reward_code: Función calculate_reward
            verify_code: Función verify_domain_physics (verificación semántica)
            output_name: Nombre del archivo de salida (sin extensión)

        Returns:
            CompileOutput con resultado de la compilación
        """
        # Generar código
        c_code = self.generate_c_code(domain, physics_code, reward_code, verify_code)

        # Nombre del archivo
        if output_name is None:
            output_name = f"lib{domain.name.lower()}"

        # Paths
        c_file = os.path.join(self.output_dir, f"{output_name}.c")
        so_file = os.path.join(self.output_dir, f"{output_name}.so")

        # Guardar código C
        with open(c_file, 'w') as f:
            f.write(c_code)

        # Compilar con GCC
        try:
            result = subprocess.run(
                [
                    "gcc",
                    "-O3",           # Optimización máxima
                    "-fPIC",         # Position Independent Code
                    "-shared",       # Crear librería compartida
                    "-o", so_file,
                    c_file,
                    "-lm",           # Linkar librería matemática
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                # Error de compilación
                return CompileOutput(
                    result=CompileResult.SYNTAX_ERROR,
                    library_path="",
                    error_message=result.stderr,
                    generated_code=c_code,
                )

            # Verificar que el archivo existe
            if not os.path.exists(so_file):
                return CompileOutput(
                    result=CompileResult.LINK_ERROR,
                    library_path="",
                    error_message="No se generó el archivo .so",
                    generated_code=c_code,
                )

            return CompileOutput(
                result=CompileResult.SUCCESS,
                library_path=so_file,
                generated_code=c_code,
            )

        except subprocess.TimeoutExpired:
            return CompileOutput(
                result=CompileResult.RUNTIME_ERROR,
                library_path="",
                error_message="Timeout durante compilación",
                generated_code=c_code,
            )
        except Exception as e:
            return CompileOutput(
                result=CompileResult.RUNTIME_ERROR,
                library_path="",
                error_message=str(e),
                generated_code=c_code,
            )

    def test_library(
        self,
        library_path: str,
        domain: DomainSpec,
        num_steps: int = 10
    ) -> Tuple[bool, str]:
        """
        Prueba que la librería compilada funciona correctamente.

        Args:
            library_path: Path a la librería .so
            domain: Especificación del dominio
            num_steps: Número de pasos de prueba

        Returns:
            (success, message)
        """
        try:
            # Cargar librería
            lib = ctypes.CDLL(library_path)

            # Configurar funciones
            lib.create_envs.argtypes = [ctypes.c_int]
            lib.create_envs.restype = ctypes.c_void_p

            lib.destroy_envs.argtypes = [ctypes.c_void_p]
            lib.destroy_envs.restype = None

            lib.reset_all.argtypes = [ctypes.c_void_p]
            lib.reset_all.restype = None

            lib.step_all.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            lib.step_all.restype = None

            lib.get_observations.argtypes = [ctypes.c_void_p]
            lib.get_observations.restype = None

            lib.get_observations_ptr.argtypes = [ctypes.c_void_p]
            lib.get_observations_ptr.restype = ctypes.POINTER(ctypes.c_float)

            lib.get_rewards_ptr.argtypes = [ctypes.c_void_p]
            lib.get_rewards_ptr.restype = ctypes.POINTER(ctypes.c_float)

            lib.get_dones_ptr.argtypes = [ctypes.c_void_p]
            lib.get_dones_ptr.restype = ctypes.POINTER(ctypes.c_int)

            lib.get_obs_size.argtypes = []
            lib.get_obs_size.restype = ctypes.c_int

            lib.get_action_size.argtypes = []
            lib.get_action_size.restype = ctypes.c_int

            # Crear entornos
            num_envs = 4
            envs = lib.create_envs(num_envs)

            if not envs:
                return False, "create_envs retornó NULL"

            # Verificar tamaños
            obs_size = lib.get_obs_size()
            action_size = lib.get_action_size()

            if obs_size != domain.obs_size:
                lib.destroy_envs(envs)
                return False, f"obs_size incorrecto: {obs_size} vs {domain.obs_size}"

            if action_size != domain.action_size:
                lib.destroy_envs(envs)
                return False, f"action_size incorrecto: {action_size} vs {domain.action_size}"

            # Reset
            lib.reset_all(envs)

            # Obtener punteros
            obs_ptr = lib.get_observations_ptr(envs)
            rewards_ptr = lib.get_rewards_ptr(envs)
            dones_ptr = lib.get_dones_ptr(envs)

            # Ejecutar pasos
            import numpy as np

            actions = (ctypes.c_float * (num_envs * action_size))()
            for i in range(num_envs * action_size):
                actions[i] = np.random.uniform(-1, 1)

            for step in range(num_steps):
                lib.step_all(envs, actions)
                lib.get_observations(envs)

                # Verificar que las recompensas son válidas
                for i in range(num_envs):
                    reward = rewards_ptr[i]
                    if not np.isfinite(reward):
                        lib.destroy_envs(envs)
                        return False, f"Recompensa NaN/Inf en paso {step}"

            # Limpiar
            lib.destroy_envs(envs)

            return True, f"Librería OK: {num_steps} pasos ejecutados correctamente"

        except Exception as e:
            return False, f"Error al probar librería: {str(e)}"


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def compile_domain(
    domain: DomainSpec,
    physics_code: str,
    reward_code: str,
    output_dir: str = "c_src"
) -> CompileOutput:
    """
    Compila código para un dominio.

    Args:
        domain: Especificación del dominio
        physics_code: Función physics_step
        reward_code: Función calculate_reward
        output_dir: Directorio de salida

    Returns:
        CompileOutput
    """
    compiler = UniversalCompiler(output_dir)
    return compiler.compile(domain, physics_code, reward_code)


def quick_compile(
    domain_name: str,
    task: str,
    output_dir: str = "c_src",
    use_mock: bool = False
) -> Tuple[CompileOutput, DomainSpec]:
    """
    Genera y compila rápidamente para un dominio predefinido.

    Args:
        domain_name: Nombre del dominio (drone, cartpole, etc.)
        task: Descripción de la tarea
        output_dir: Directorio de salida
        use_mock: Si usar código mock

    Returns:
        (CompileOutput, DomainSpec)
    """
    from domain_spec import get_domain
    from universal_architect import UniversalArchitect

    # Obtener dominio
    domain = get_domain(domain_name)

    # Generar código
    architect = UniversalArchitect(use_mock=use_mock)
    generated = architect.generate_both(domain, task)

    if not generated.success:
        return CompileOutput(
            result=CompileResult.RUNTIME_ERROR,
            library_path="",
            error_message=f"Error generando código: {generated.error_message}",
        ), domain

    # Compilar
    compiler = UniversalCompiler(output_dir)
    output = compiler.compile(domain, generated.physics_code, generated.reward_code)

    return output, domain


if __name__ == "__main__":
    # Demo
    print("=== Universal Compiler Demo ===\n")

    from domain_spec import get_domain
    from universal_architect import UniversalArchitect

    # Probar con CartPole (más simple que drone)
    domain = get_domain("cartpole")
    print(f"Dominio: {domain.name}")
    print(f"Estado: {domain.obs_size} observaciones")
    print(f"Acciones: {domain.action_size}")

    # Generar código
    architect = UniversalArchitect(use_mock=True)
    physics = architect.generate_physics(domain)
    reward = architect.generate_reward(domain, "Mantener el palo vertical el mayor tiempo posible")

    print(f"\nFísica generada: {'OK' if physics.success else 'FALLO'}")
    print(f"Recompensa generada: {'OK' if reward.success else 'FALLO'}")

    # Compilar
    compiler = UniversalCompiler()
    output = compiler.compile(domain, physics.physics_code, reward.reward_code)

    print(f"\nCompilación: {output.result.value}")
    if output.result == CompileResult.SUCCESS:
        print(f"Librería: {output.library_path}")

        # Probar
        success, msg = compiler.test_library(output.library_path, domain)
        print(f"Test: {msg}")
    else:
        print(f"Error: {output.error_message}")
