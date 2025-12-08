#!/usr/bin/env python3
"""
UNIVERSAL ROBOTICS PLATFORM
===========================

Plataforma de Inteligencia General para entrenar CUALQUIER sistema robótico.

Transforma descripciones en lenguaje natural a políticas entrenadas:
  "Un brazo robótico que alcanza objetos" → Política neural entrenada

Arquitectura:
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                   │
│  "Quiero un robot de warehouse que navegue evitando obstáculos" │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOMAIN SPEC                                  │
│  Define: estado, acciones, física, métricas                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                UNIVERSAL ARCHITECT (LLM)                        │
│  Genera: physics_step() + calculate_reward()                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                UNIVERSAL COMPILER (GCC)                         │
│  Compila: C → libdomain.so (2M+ steps/sec)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                UNIVERSAL ENV (ctypes)                           │
│  Ejecuta: Python ↔ C zero-copy                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                PPO TRAINER                                      │
│  Entrena: Política neural en GPU                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                UNIVERSAL JUDGE                                  │
│  Evalúa: ¿Aprendió? ¿Estable? ¿Eficiente?                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                EVOLUTION LOOP (Eureka)                          │
│  Itera: Mejora recompensa → Re-entrena → Evalúa                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                       │
│  Política entrenada (.pt) + Código óptimo (.c)                  │
└─────────────────────────────────────────────────────────────────┘

Uso:
    python universal_platform.py --domain drone --task "hover estable a 1 metro"
    python universal_platform.py --domain cartpole --task "balancear el palo"
    python universal_platform.py --domain warehouse_robot --task "navegar al punto B"

Autor: Sistema de IA Generativa
Versión: 1.0.0
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Optional

# Importar componentes de la plataforma
from domain_spec import DomainSpec, get_domain, list_domains, DOMAIN_CATALOG
from universal_architect import UniversalArchitect, GeneratedCode
from universal_compiler import UniversalCompiler, CompileResult
from universal_env import UniversalVecEnv, create_from_domain_name
from universal_judge import UniversalJudge, generate_semantic_reflection
from universal_evolution import UniversalEvolution, evolve_domain, EvolutionResult
from physics_verifier import PhysicsVerifier, SelfCorrectingArchitect, VerificationReport, TestResult


def print_banner():
    """Imprime banner de bienvenida"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ██╗   ██╗███╗   ██╗██╗██╗   ██╗███████╗██████╗ ███████╗ █████╗  ║
║   ██║   ██║████╗  ██║██║██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗ ║
║   ██║   ██║██╔██╗ ██║██║██║   ██║█████╗  ██████╔╝███████╗███████║ ║
║   ██║   ██║██║╚██╗██║██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║ ║
║   ╚██████╔╝██║ ╚████║██║ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║ ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ║
║                                                                   ║
║            UNIVERSAL ROBOTICS PLATFORM v1.0                       ║
║         Train ANY robotic system with natural language            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_available_domains():
    """Imprime dominios disponibles"""
    print("\nDominios Disponibles:")
    print("-" * 50)

    for name in list_domains():
        domain = get_domain(name)
        print(f"  {name:20} - {domain.description[:40]}...")
        print(f"                       Obs: {domain.obs_size}, Actions: {domain.action_size}")

    print("-" * 50)


def verified_train(
    domain_name: str,
    task: str,
    steps: int = 100_000,
    num_envs: int = 64,
    use_mock: bool = False,
    save_model: bool = True,
    max_attempts: int = 3
) -> Optional[str]:
    """
    Entrenamiento con verificación física previa.

    Usa SelfCorrectingArchitect para generar código que pasa
    verificación de física antes de entrenar.

    Args:
        domain_name: Nombre del dominio
        task: Descripción de la tarea
        steps: Pasos de entrenamiento
        num_envs: Entornos paralelos
        use_mock: Si usar código mock
        save_model: Si guardar el modelo
        max_attempts: Intentos máximos de auto-corrección

    Returns:
        Path al modelo guardado, o None si falló
    """
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO VERIFICADO")
    print(f"Dominio: {domain_name}")
    print(f"Tarea: {task}")
    print(f"Pasos: {steps:,}")
    print(f"Verificación: Activada (max {max_attempts} intentos)")
    print(f"{'='*60}\n")

    try:
        domain = get_domain(domain_name)

        # Fase 1: Generar código verificado
        print("=== FASE 1: GENERACIÓN VERIFICADA ===\n")

        if use_mock:
            # Mock: usar código pre-definido
            from universal_architect import UniversalArchitect
            architect = UniversalArchitect()
            generated = architect.generate_both(domain, task)
            physics_code = generated.physics_code
            reward_code = generated.reward_code

            # Verificar el código mock
            verifier = PhysicsVerifier()
            report = verifier.verify(domain, physics_code, reward_code)

            print(f"Verificación mock:")
            print(f"  Resultado: {'PASSED' if report.all_tests_passed else 'FAILED'}")
            print(f"  Tests pasados: {report.passed_tests}/{report.total_tests}")

            if not report.all_tests_passed:
                print(f"\nTests fallidos:")
                for test in report.tests:
                    if test.result != TestResult.PASSED:
                        print(f"  - {test.name}: {test.message}")
                print("\nNota: Código mock no pasó verificación, continuando de todos modos...")
        else:
            # Real: usar SelfCorrectingArchitect
            architect = SelfCorrectingArchitect(max_attempts=max_attempts)
            physics_code, reward_code, report = architect.generate_verified(domain, task)

            print(f"\nResultado de verificación:")
            print(f"  Tests pasados: {report.passed_tests}/{report.total_tests}")

            if not report.all_tests_passed:
                print(f"\n[ERROR CRÍTICO] Código no pasó verificación física:")
                for test in report.tests:
                    if test.result != TestResult.PASSED:
                        print(f"  ❌ {test.name}: {test.message}")
                print(f"\nDiagnóstico: {report.failure_diagnosis}")
                print("\nSugerencias de corrección:")
                for hint in report.correction_hints:
                    print(f"  → {hint}")
                raise RuntimeError(f"Verificación fallida: {report.failure_diagnosis}. No se iniciará el entrenamiento para proteger el presupuesto.")

        # Fase 2: Compilar
        print("\n=== FASE 2: COMPILACIÓN ===\n")

        compiler = UniversalCompiler("c_src")
        output = compiler.compile(
            domain,
            physics_code,
            reward_code,
            output_name=f"lib{domain_name}_verified"
        )

        if output.result != CompileResult.SUCCESS:
            print(f"Error de compilación: {output.error_message}")
            return None

        print(f"Librería compilada: {output.library_path}")

        # Fase 2.5: Vista Previa Visual (Candado Visual)
        print("\n=== FASE 2.5: VISTA PREVIA ===\n")

        from preview_physics import preview_physics
        approved, preview_diagnosis = preview_physics(
            domain, physics_code, reward_code, verify_code,
            duration_seconds=5.0
        )

        if not approved:
            print(f"\n[ERROR] Vista previa falló: {preview_diagnosis}")
            print("Abortando para proteger el presupuesto.")
            raise RuntimeError(f"Vista previa rechazada: {preview_diagnosis}")

        print("\n✅ Vista previa APROBADA")

        # Fase 3: Entrenar con Early Stopping (Candado de Convergencia)
        print("\n=== FASE 3: ENTRENAMIENTO ===\n")

        env = UniversalVecEnv(num_envs, domain, output.library_path)

        from simple_ppo import SimplePPO
        ppo = SimplePPO(env, domain)

        # Usar early stopping agresivo
        metrics, training_approved, training_diagnosis = ppo.train_with_early_stopping(
            total_steps=steps,
            dry_run_steps=min(1000, steps // 10),  # 10% o 1000 pasos
            min_learning_gain=0.01
        )

        if not training_approved:
            print(f"\n[ERROR] Entrenamiento rechazado: {training_diagnosis}")
            env.close()
            raise RuntimeError(f"Early stopping: {training_diagnosis}")

        # Evaluar
        judge = UniversalJudge()
        result = judge.judge(metrics)

        print(f"\nResultado: {result.quality.value}")
        print(f"Score: {result.score:.1f}/100")
        print(f"Diagnóstico: {result.diagnosis}")

        # Guardar modelo
        if save_model:
            import torch
            model_path = f"models/{domain_name}_verified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            os.makedirs("models", exist_ok=True)

            torch.save({
                "policy": ppo.policy.state_dict(),
                "actor_mean": ppo.actor_mean.state_dict(),
                "actor_logstd": ppo.actor_logstd,
                "critic": ppo.critic.state_dict(),
                "domain": domain_name,
                "task": task,
                "score": result.score,
                "verified": True,
                "physics_tests_passed": report.passed_tests,
                "physics_tests_total": report.total_tests,
            }, model_path)

            # Guardar código verificado
            code_path = f"models/{domain_name}_verified_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.c"
            with open(code_path, 'w') as f:
                f.write(f"// Physics Code\n{physics_code}\n\n// Reward Code\n{reward_code}")

            print(f"\nArchivos guardados:")
            print(f"  Modelo: {model_path}")
            print(f"  Código: {code_path}")

            env.close()
            return model_path

        env.close()
        return None

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_train(
    domain_name: str,
    task: str,
    steps: int = 100_000,
    num_envs: int = 64,
    use_mock: bool = False,
    save_model: bool = True
) -> Optional[str]:
    """
    Entrenamiento rápido sin evolución.

    Args:
        domain_name: Nombre del dominio
        task: Descripción de la tarea
        steps: Pasos de entrenamiento
        num_envs: Entornos paralelos
        use_mock: Si usar código mock
        save_model: Si guardar el modelo

    Returns:
        Path al modelo guardado, o None si falló
    """
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO RÁPIDO")
    print(f"Dominio: {domain_name}")
    print(f"Tarea: {task}")
    print(f"Pasos: {steps:,}")
    print(f"{'='*60}\n")

    try:
        # Crear entorno
        print("Creando entorno...")
        env, domain, lib_path = create_from_domain_name(
            domain_name, task, num_envs, use_mock
        )

        print(f"  Librería: {lib_path}")
        print(f"  Observaciones: {domain.obs_size}")
        print(f"  Acciones: {domain.action_size}")

        # Entrenar
        print("\nEntrenando...")
        from universal_evolution import SimplePPO
        ppo = SimplePPO(env, domain)
        metrics = ppo.train(steps)

        # Evaluar
        judge = UniversalJudge()
        result = judge.judge(metrics)

        print(f"\nResultado: {result.quality.value}")
        print(f"Score: {result.score:.1f}/100")
        print(f"Diagnóstico: {result.diagnosis}")

        # Guardar modelo
        if save_model:
            import torch
            model_path = f"models/{domain_name}_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            os.makedirs("models", exist_ok=True)

            # Guardar pesos de la política
            torch.save({
                "policy": ppo.policy.state_dict(),
                "actor_mean": ppo.actor_mean.state_dict(),
                "actor_logstd": ppo.actor_logstd,
                "critic": ppo.critic.state_dict(),
                "domain": domain_name,
                "task": task,
                "score": result.score,
            }, model_path)

            print(f"\nModelo guardado: {model_path}")
            return model_path

        env.close()
        return None

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def evolve_and_train(
    domain_name: str,
    task: str,
    generations: int = 3,
    population_size: int = 3,
    eval_steps: int = 50_000,
    final_steps: int = 500_000,
    use_mock: bool = False
) -> Optional[EvolutionResult]:
    """
    Evolución completa + entrenamiento final.

    Args:
        domain_name: Nombre del dominio
        task: Descripción de la tarea
        generations: Generaciones de evolución
        population_size: Tamaño de población
        eval_steps: Pasos por evaluación
        final_steps: Pasos de entrenamiento final
        use_mock: Si usar código mock

    Returns:
        EvolutionResult o None si falló
    """
    print(f"\n{'='*60}")
    print(f"EVOLUCIÓN + ENTRENAMIENTO")
    print(f"Dominio: {domain_name}")
    print(f"Tarea: {task}")
    print(f"Generaciones: {generations}")
    print(f"Población: {population_size}")
    print(f"{'='*60}\n")

    try:
        # Fase 1: Evolución
        print("=== FASE 1: EVOLUCIÓN ===\n")
        result = evolve_domain(
            domain_name=domain_name,
            task=task,
            generations=generations,
            population_size=population_size,
            eval_steps=eval_steps,
            use_mock=use_mock,
            verbose=True
        )

        print("\n" + result.summary())

        # Fase 2: Entrenamiento final con mejor recompensa
        print("\n=== FASE 2: ENTRENAMIENTO FINAL ===\n")

        domain = get_domain(domain_name)
        compiler = UniversalCompiler("c_src")

        # Compilar mejor código
        output = compiler.compile(
            domain,
            result.best_candidate.physics_code,
            result.best_candidate.reward_code,
            output_name=f"lib{domain_name}_final"
        )

        if output.result != CompileResult.SUCCESS:
            print(f"Error compilando: {output.error_message}")
            return result

        # Crear entorno
        env = UniversalVecEnv(64, domain, output.library_path)

        # Entrenar con más pasos
        from universal_evolution import SimplePPO
        ppo = SimplePPO(env, domain)
        metrics = ppo.train(final_steps)

        # Evaluar final
        judge = UniversalJudge()
        final_result = judge.judge(metrics)

        print(f"\nEntrenamiento final completado:")
        print(f"  Calidad: {final_result.quality.value}")
        print(f"  Score: {final_result.score:.1f}")

        # Guardar
        os.makedirs("models", exist_ok=True)

        # Guardar modelo
        import torch
        model_path = f"models/{domain_name}_evolved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            "policy": ppo.policy.state_dict(),
            "actor_mean": ppo.actor_mean.state_dict(),
            "actor_logstd": ppo.actor_logstd,
            "critic": ppo.critic.state_dict(),
            "domain": domain_name,
            "task": task,
            "evolution_score": result.best_candidate.score,
            "final_score": final_result.score,
        }, model_path)

        # Guardar código de recompensa
        reward_path = f"models/{domain_name}_reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.c"
        with open(reward_path, 'w') as f:
            f.write(result.best_candidate.reward_code)

        print(f"\nArchivos guardados:")
        print(f"  Modelo: {model_path}")
        print(f"  Recompensa: {reward_path}")

        env.close()
        return result

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    """Modo interactivo para experimentar"""
    print_banner()
    print_available_domains()

    while True:
        print("\nOpciones:")
        print("  1. Entrenamiento rápido")
        print("  2. Evolución completa")
        print("  3. Entrenamiento verificado (con tests de física)")
        print("  4. Verificar física solo")
        print("  5. Ver dominios")
        print("  6. Salir")

        try:
            choice = input("\nSelección: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nHasta luego!")
            break

        if choice == "1":
            domain = input("Dominio (drone/cartpole/robotic_arm/warehouse_robot): ").strip()
            task = input("Tarea: ").strip()
            steps = input("Pasos [100000]: ").strip()
            steps = int(steps) if steps else 100_000

            quick_train(domain, task, steps, use_mock=True)

        elif choice == "2":
            domain = input("Dominio: ").strip()
            task = input("Tarea: ").strip()
            gens = input("Generaciones [3]: ").strip()
            gens = int(gens) if gens else 3

            evolve_and_train(domain, task, generations=gens, use_mock=True)

        elif choice == "3":
            domain = input("Dominio (drone/cartpole/robotic_arm/warehouse_robot): ").strip()
            task = input("Tarea: ").strip()
            steps = input("Pasos [100000]: ").strip()
            steps = int(steps) if steps else 100_000
            use_real = input("¿Usar LLM real? (s/n) [n]: ").strip().lower()

            verified_train(domain, task, steps, use_mock=(use_real != 's'))

        elif choice == "4":
            # Solo verificar física sin entrenar
            domain_name = input("Dominio: ").strip()
            try:
                domain = get_domain(domain_name)

                print("\nGenerando código mock para verificar...")
                from universal_architect import UniversalArchitect
                architect = UniversalArchitect()
                generated = architect.generate_both(domain, "test")

                print("Ejecutando tests de física...\n")
                verifier = PhysicsVerifier()
                report = verifier.verify(domain, generated.physics_code, generated.reward_code)

                print(f"\n{'='*50}")
                print(f"REPORTE DE VERIFICACIÓN - {domain_name}")
                print(f"{'='*50}")
                print(f"Resultado: {'PASSED' if report.all_tests_passed else 'FAILED'}")
                print(f"Tests pasados: {report.passed_tests}/{report.total_tests}")

                passed = [t for t in report.tests if t.result == TestResult.PASSED]
                failed = [t for t in report.tests if t.result != TestResult.PASSED]

                if passed:
                    print(f"\nTests exitosos:")
                    for test in passed:
                        print(f"  ✓ {test.name}")

                if failed:
                    print(f"\nTests fallidos:")
                    for test in failed:
                        print(f"  ✗ {test.name}: {test.message}")

                if report.failure_diagnosis:
                    print(f"\nDiagnóstico:")
                    print(f"  {report.failure_diagnosis}")

            except Exception as e:
                print(f"Error: {e}")

        elif choice == "5":
            print_available_domains()

        elif choice == "6":
            print("\nHasta luego!")
            break

        else:
            print("Opción no válida")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Robotics Platform - Train any robot with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python universal_platform.py --domain drone --task "hover estable"
  python universal_platform.py --domain cartpole --task "balancear" --evolve
  python universal_platform.py --domain drone --task "hover" --verify
  python universal_platform.py --list-domains
  python universal_platform.py --interactive
"""
    )

    parser.add_argument("--domain", type=str, help="Nombre del dominio")
    parser.add_argument("--task", type=str, help="Descripción de la tarea")
    parser.add_argument("--steps", type=int, default=100_000, help="Pasos de entrenamiento")
    parser.add_argument("--evolve", action="store_true", help="Usar evolución antes de entrenar")
    parser.add_argument("--verify", action="store_true", help="Verificar física antes de entrenar")
    parser.add_argument("--generations", type=int, default=3, help="Generaciones de evolución")
    parser.add_argument("--population", type=int, default=3, help="Tamaño de población")
    parser.add_argument("--mock", action="store_true", help="Usar código mock (sin LLM)")
    parser.add_argument("--list-domains", action="store_true", help="Listar dominios disponibles")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")

    args = parser.parse_args()

    # Listar dominios
    if args.list_domains:
        print_banner()
        print_available_domains()
        return

    # Modo interactivo
    if args.interactive:
        interactive_mode()
        return

    # Entrenamiento normal
    if not args.domain or not args.task:
        print("Error: Se requiere --domain y --task")
        print("Use --help para ver opciones")
        print("Use --list-domains para ver dominios disponibles")
        sys.exit(1)

    print_banner()

    if args.verify:
        verified_train(
            args.domain,
            args.task,
            steps=args.steps,
            use_mock=args.mock or True  # Por defecto mock
        )
    elif args.evolve:
        evolve_and_train(
            args.domain,
            args.task,
            generations=args.generations,
            population_size=args.population,
            use_mock=args.mock or True  # Por defecto mock
        )
    else:
        quick_train(
            args.domain,
            args.task,
            steps=args.steps,
            use_mock=args.mock or True
        )


if __name__ == "__main__":
    main()
