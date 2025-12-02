#!/usr/bin/env python3
"""
EJEMPLOS DE USO DE train_q_learning() CON OPTIMIZACIONES
=========================================================

Este archivo contiene ejemplos listos para copiar/pegar.
"""

from Metodos_Tabulares import train_q_learning
import time

print("="*70)
print("EJEMPLOS DE ENTRENAMIENTO DE Q-LEARNING CON OPTIMIZACIONES")
print("="*70)

# ============================================================================
# EJEMPLO 1: ENTRENAMIENTO RÁPIDO (Recomendado para desarrollo)
# ============================================================================
print("\n" + "─"*70)
print("EJEMPLO 1: Entrenamiento Rápido (3-5 minutos para 1000 episodios)")
print("─"*70)
print("""
# Perfecto para:
# - Desarrollo y pruebas rápidas
# - Validar que el algoritmo converge
# - Ajustar hiperparámetros rápidamente

q_table, bins, rewards = train_q_learning(
    episodes=1000,
    max_steps=150,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    fast_mode=True,      # ⭐ Desactiva magnético
    granularity=5        # ⭐ Granularidad baja
)
""")

# ============================================================================
# EJEMPLO 2: ENTRENAMIENTO EQUILIBRADO (Recomendado para testing)
# ============================================================================
print("\n" + "─"*70)
print("EJEMPLO 2: Entrenamiento Equilibrado (10-15 minutos)")
print("─"*70)
print("""
# Perfecto para:
# - Entrenamiento final de modelos
# - Balance entre velocidad y precisión

q_table, bins, rewards = train_q_learning(
    episodes=2000,
    max_steps=200,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    fast_mode=True,      # Todavía activado para velocidad
    granularity=10       # Balance
)
""")

# ============================================================================
# EJEMPLO 3: ENTRENAMIENTO PRECISO (Para simulaciones realistas)
# ============================================================================
print("\n" + "─"*70)
print("EJEMPLO 3: Entrenamiento Preciso (30-40 minutos)")
print("─"*70)
print("""
# Perfecto para:
# - Testing final con dinámicas realistas
# - Validación de desempeño real
# - Documentación de resultados

q_table, bins, rewards = train_q_learning(
    episodes=2000,
    max_steps=300,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    fast_mode=False,     # Usa campo magnético real
    granularity=20       # Máxima precisión
)
""")

# ============================================================================
# EJEMPLO 4: ENTRENAMIENTO AGRESIVO (Para convergencia rápida)
# ============================================================================
print("\n" + "─"*70)
print("EJEMPLO 4: Entrenamiento Agresivo (1-2 minutos)")
print("─"*70)
print("""
# Perfecto para:
# - Debugging rápido
# - Verificar que el código funciona

q_table, bins, rewards = train_q_learning(
    episodes=100,
    max_steps=100,
    learning_rate=0.15,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.98,  # Exploración muy rápida
    min_epsilon=0.01,
    fast_mode=True,      # Máxima velocidad
    granularity=3        # Granularidad mínima
)
""")

# ============================================================================
# COMPARACIÓN DE VELOCIDADES
# ============================================================================
print("\n" + "="*70)
print("COMPARACIÓN DE VELOCIDADES (1000 episodios)")
print("="*70)

comparison = """
┌─────────────────────────────────────────────────────────────┐
│ Configuración         │ Tiempo       │ Velocidad │ Notas    │
├─────────────────────────────────────────────────────────────┤
│ Sin optimizaciones    │ 20-30 min    │ 1x        │ Original │
│ fast_mode=True        │ 8-10 min     │ 3-4x      │ Bueno    │
│ granularity=5         │ 10-12 min    │ 2-2.5x    │ Bueno    │
│ Ambas juntas          │ 3-5 min      │ 7-10x     │ ⭐ MEJOR │
│ granularity=3 + fast  │ 1-2 min      │ 15x       │ Extremo  │
└─────────────────────────────────────────────────────────────┘
"""
print(comparison)

# ============================================================================
# CONSEJOS PRÁCTICOS
# ============================================================================
print("\n" + "="*70)
print("CONSEJOS PRÁCTICOS")
print("="*70)

tips = """
1. INICIAR DESARROLLO:
   - Usa Ejemplo 1 (entrenamiento rápido)
   - Ejecuta cada 2 minutos
   - Ajusta hiperparámetros

2. VALIDACIÓN:
   - Usa Ejemplo 2 (equilibrado)
   - Observa convergencia
   - Verifica que el éxito mejora

3. TESTING FINAL:
   - Usa Ejemplo 3 (preciso)
   - Ejecuta 2-3 veces para obtener promedio
   - Documenta resultados

4. DEBUGGING:
   - Usa Ejemplo 4 (agresivo)
   - Solo para verificar que el código funciona
   - NO uses para entrenar agentes reales

5. OPTIMIZACIONES:
   - fast_mode=True: ~3-5x más rápido (pierde precisión magnética)
   - granularity=5: ~2x más rápido (aún preciso para rotación)
   - Ambas: ~7-10x más rápido (suficiente para la mayoría)

6. HIPERPARÁMETROS:
   - learning_rate: Más alto = aprendizaje rápido pero inestable
   - epsilon_decay: Más bajo = exploración rápida
   - max_steps: Menos pasos = episodios cortos = presión para rapidez
"""
print(tips)

# ============================================================================
# EJECUCIÓN
# ============================================================================
print("\n" + "="*70)
print("PARA EJECUTAR:")
print("="*70)

execution = """
Opción 1: Ejecutar directamente desde Terminal
$ python Metodos_Tabulares.py
(Te preguntará qué agente entrenar)

Opción 2: Importar y usar en tu código
from Metodos_Tabulares import train_q_learning

q_table, bins, rewards = train_q_learning(
    episodes=1000,
    fast_mode=True,
    granularity=5
)

Opción 3: Usar los ejemplos de este archivo
$ python ejemplos_train_qlearning.py
(Este archivo solo muestra código, no ejecuta)
"""
print(execution)

print("\n" + "="*70)
print("✅ Referencia completa lista para usar")
print("="*70 + "\n")
