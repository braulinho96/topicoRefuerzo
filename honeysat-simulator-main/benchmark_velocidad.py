#!/usr/bin/env python3
"""
SCRIPT DE COMPARACIÓN DE VELOCIDADES
=====================================

Ejecuta este script para ver comparación de velocidades entre diferentes configuraciones.
"""

import time
import numpy as np
from cubesat_detumbling_rl import CubeSatDetumblingEnv
from datetime import datetime

def benchmark_configuration(name, config, episodes=10):
    """
    Realiza benchmark de una configuración del entorno.
    
    Args:
        name: Nombre de la configuración
        config: Dict con parámetros del entorno
        episodes: Número de episodios para benchmarking
    """
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK: {name}")
    print(f"{'='*60}")
    print(f"Configuración: {config}")
    print(f"Episodios a ejecutar: {episodes}")
    print()
    
    env = CubeSatDetumblingEnv(**config)
    
    # Warmup
    obs, _ = env.reset()
    
    start_time = time.time()
    total_steps = 0
    
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                done = terminated or truncated
            
            # Print progress
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(f"  Episodio {ep+1}/{episodes} - "
                  f"Velocidad: {eps_per_sec:.2f} ep/s - "
                  f"Tiempo: {elapsed:.1f}s")
    
    finally:
        env.close()
    
    total_time = time.time() - start_time
    
    # Resultados
    print(f"\n{'─'*60}")
    print(f"⏱️  RESULTADOS:")
    print(f"   Tiempo total:        {total_time:.2f}s")
    print(f"   Episodios:           {episodes}")
    print(f"   Pasos totales:       {total_steps}")
    print(f"   Velocidad:           {episodes/total_time:.2f} ep/s")
    print(f"   Tiempo/episodio:     {total_time/episodes:.3f}s ({total_time/episodes*1000:.1f}ms)")
    print(f"   Tiempo/paso:         {total_time/total_steps*1000:.2f}ms")
    print(f"{'─'*60}\n")
    
    return {
        'name': name,
        'time': total_time,
        'episodes': episodes,
        'steps': total_steps,
        'eps_per_sec': episodes/total_time,
        'time_per_ep': total_time/episodes,
    }

def main():
    print("\n" + "="*60)
    print("🚀 BENCHMARKING DE OPTIMIZACIONES DE VELOCIDAD")
    print("="*60)
    print("\nEste script compara la velocidad de diferentes configuraciones.")
    print("Se ejecutarán 10 episodios para cada una.\n")
    
    # Configuraciones a probar
    configurations = [
        ("BASE (sin optimizar)", {
            'max_steps': 500,
            'granularity': 100,
            'fast_mode': False,
            'render_mode': None
        }),
        
        ("Granularidad=10", {
            'max_steps': 500,
            'granularity': 10,
            'fast_mode': False,
            'render_mode': None
        }),
        
        ("Granularidad=5", {
            'max_steps': 500,
            'granularity': 5,
            'fast_mode': False,
            'render_mode': None
        }),
        
        ("Fast Mode", {
            'max_steps': 500,
            'granularity': 10,
            'fast_mode': True,
            'render_mode': None
        }),
        
        ("OPTIMIZADO (todos)", {
            'max_steps': 150,
            'granularity': 5,
            'fast_mode': True,
            'render_mode': None
        }),
    ]
    
    results = []
    
    try:
        for name, config in configurations:
            result = benchmark_configuration(name, config, episodes=10)
            results.append(result)
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumen comparativo
    if results:
        print("\n" + "="*60)
        print("📊 RESUMEN COMPARATIVO")
        print("="*60)
        print(f"{'Configuración':<30} {'Tiempo':<12} {'Velocidad':<15} {'Speedup':<10}")
        print(f"{'-'*30} {'-'*12} {'-'*15} {'-'*10}")
        
        baseline_time = results[0]['time']
        
        for result in results:
            speedup = baseline_time / result['time']
            speedup_str = f"{speedup:.1f}x" if speedup > 1 else "Baseline"
            
            print(f"{result['name']:<30} {result['time']:>6.2f}s     "
                  f"{result['eps_per_sec']:>6.2f} ep/s    {speedup_str:>10}")
        
        print(f"{'='*60}\n")
        
        # Estimación para 1000 episodios
        print("📈 ESTIMACIÓN PARA 1000 EPISODIOS:")
        print(f"{'─'*60}")
        for result in results:
            est_time = result['time_per_ep'] * 1000 / 60
            print(f"{result['name']:<30} {est_time:>6.1f} minutos")
        print(f"{'─'*60}\n")

if __name__ == "__main__":
    main()
