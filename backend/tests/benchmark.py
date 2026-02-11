# backend/tests/benchmark.py
"""
Suite de benchmarks para medir rendimiento del sistema.

OBJETIVO: Identificar cuellos de botella y validar optimizaciones

BENCHMARKS:
1. Renderizado de fórmulas (varios tamaños)
2. Ajuste de contornos (varias complejidades)
3. Optimización de coeficientes
4. Evaluación de métricas
"""

import time
import numpy as np
import pytest
from typing import Callable, Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

from app.services.renderer import ParametricRenderer, RenderConfig
from app.services.contour_fitter import ContourFitter
from app.services.metrics import ImageMetrics

@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    name: str
    elapsed_time: float  # segundos
    iterations: int
    avg_time: float  # segundos por iteración
    throughput: float  # operaciones por segundo
    memory_mb: float  # Uso de memoria en MB

class PerformanceBenchmark:
    """Suite de benchmarks de rendimiento."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(
        self,
        func: Callable,
        name: str,
        iterations: int = 10,
        warmup: int = 2
    ) -> BenchmarkResult:
        """
        Ejecuta benchmark de una función.
        
        Args:
            func: Función a benchmarkear
            name: Nombre descriptivo
            iterations: Número de iteraciones
            warmup: Iteraciones de calentamiento (descartadas)
        """
        import tracemalloc
        
        # Calentamiento
        for _ in range(warmup):
            func()
        
        # Medición de memoria
        tracemalloc.start()
        
        # Medición de tiempo
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Estadísticas
        avg_time = np.mean(times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            elapsed_time=sum(times),
            iterations=iterations,
            avg_time=avg_time,
            throughput=throughput,
            memory_mb=peak / 1024 / 1024
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """Imprime tabla de resultados."""
        print("
" + "="*80)
        print("RESULTADOS DE BENCHMARKS")
        print("="*80)
        print(f"{'Benchmark':<40} {'Avg Time':<15} {'Throughput':<15} {'Memory':<10}")
        print("-"*80)
        
        for result in self.results:
            print(f"{result.name:<40} "
                  f"{result.avg_time*1000:>10.2f} ms    "
                  f"{result.throughput:>10.2f} /s    "
                  f"{result.memory_mb:>6.1f} MB")
        
        print("="*80 + "
")
    
    def plot_results(self, save_path: str = "benchmark_results.png"):
        """Genera gráfico de resultados."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        names = [r.name for r in self.results]
        times = [r.avg_time * 1000 for r in self.results]  # ms
        memory = [r.memory_mb for r in self.results]
        
        # Gráfico de tiempos
        ax1.barh(names, times, color='steelblue')
        ax1.set_xlabel('Tiempo Promedio (ms)')
        ax1.set_title('Tiempo de Ejecución')
        ax1.grid(axis='x', alpha=0.3)
        
        # Gráfico de memoria
        ax2.barh(names, memory, color='coral')
        ax2.set_xlabel('Memoria Pico (MB)')
        ax2.set_title('Uso de Memoria')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

# Suite de benchmarks específicos
def run_rendering_benchmarks():
    """Benchmarks de renderizado."""
    benchmark = PerformanceBenchmark()
    renderer = ParametricRenderer()
    
    # Fórmula compleja
    x_func = lambda t: np.cos(5*t) * np.cos(t) + 0.3 * np.sin(11*t)
    y_func = lambda t: np.cos(5*t) * np.sin(t) + 0.3 * np.cos(13*t)
    
    # Benchmark 1: Resolución baja (400x400)
    config_low = RenderConfig(width=400, height=400, num_points=5000)
    benchmark.benchmark_function(
        lambda: renderer.render_curve(x_func, y_func, config_low),
        name="Render 400x400, 5K puntos",
        iterations=20
    )
    
    # Benchmark 2: Resolución media (800x800)
    config_med = RenderConfig(width=800, height=800, num_points=15000)
    benchmark.benchmark_function(
        lambda: renderer.render_curve(x_func, y_func, config_med),
        name="Render 800x800, 15K puntos",
        iterations=10
    )
    
    # Benchmark 3: Resolución alta (1600x1600)
    config_high = RenderConfig(width=1600, height=1600, num_points=50000)
    benchmark.benchmark_function(
        lambda: renderer.render_curve(x_func, y_func, config_high),
        name="Render 1600x1600, 50K puntos",
        iterations=5
    )
    
    # Benchmark 4: Muestreo adaptativo vs uniforme
    config_adaptive = RenderConfig(width=800, height=800, num_points=20000, adaptive=True)
    config_uniform = RenderConfig(width=800, height=800, num_points=20000, adaptive=False)
    
    benchmark.benchmark_function(
        lambda: renderer.render_curve(x_func, y_func, config_adaptive),
        name="Render adaptativo 800x800, 20K",
        iterations=10
    )
    
    benchmark.benchmark_function(
        lambda: renderer.render_curve(x_func, y_func, config_uniform),
        name="Render uniforme 800x800, 20K",
        iterations=10
    )
    
    benchmark.print_results()
    benchmark.plot_results("rendering_benchmarks.png")
    
    return benchmark

def run_fitting_benchmarks():
    """Benchmarks de ajuste de contornos."""
    benchmark = PerformanceBenchmark()
    fitter = ContourFitter(max_terms=20)
    
    # Generar contornos sintéticos de diferentes tamaños
    def generate_circle_contour(num_points: int):
        t = np.linspace(0, 2*np.pi, num_points)
        x = 100 + 50 * np.cos(t)
        y = 100 + 50 * np.sin(t)
        return np.column_stack([y, x])
    
    # Benchmark 1: Contorno pequeño (100 puntos)
    contour_small = generate_circle_contour(100)
    benchmark.benchmark_function(
        lambda: fitter.fit_fourier_series(contour_small, num_terms=10),
        name="Ajuste Fourier: 100 puntos, 10 términos",
        iterations=50
    )
    
    # Benchmark 2: Contorno mediano (500 puntos)
    contour_medium = generate_circle_contour(500)
    benchmark.benchmark_function(
        lambda: fitter.fit_fourier_series(contour_medium, num_terms=15),
        name="Ajuste Fourier: 500 puntos, 15 términos",
        iterations=20
    )
    
    # Benchmark 3: Contorno grande (2000 puntos)
    contour_large = generate_circle_contour(2000)
    benchmark.benchmark_function(
        lambda: fitter.fit_fourier_series(contour_large, num_terms=20),
        name="Ajuste Fourier: 2000 puntos, 20 términos",
        iterations=10
    )
    
    # Benchmark 4: Optimización
    coefs_x, coefs_y = fitter.fit_fourier_series(contour_medium, num_terms=10)
    benchmark.benchmark_function(
        lambda: fitter.optimize_coefficients(
            contour_medium, coefs_x, coefs_y, max_iterations=100
        ),
        name="Optimización: 500 puntos, 100 iters",
        iterations=5
    )
    
    benchmark.print_results()
    benchmark.plot_results("fitting_benchmarks.png")
    
    return benchmark

def run_metrics_benchmarks():
    """Benchmarks de cálculo de métricas."""
    benchmark = PerformanceBenchmark()
    
    # Imágenes sintéticas
    img_small = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img_medium = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    img_large = np.random.randint(0, 255, (1600, 1600, 3), dtype=np.uint8)
    
    # Pequeñas variaciones
    img_small_var = np.clip(img_small + np.random.randint(-10, 10, img_small.shape), 0, 255).astype(np.uint8)
    img_medium_var = np.clip(img_medium + np.random.randint(-10, 10, img_medium.shape), 0, 255).astype(np.uint8)
    img_large_var = np.clip(img_large + np.random.randint(-10, 10, img_large.shape), 0, 255).astype(np.uint8)
    
    # Benchmarks L2
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_l2_error(img_small, img_small_var),
        name="L2 Error: 200x200",
        iterations=100
    )
    
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_l2_error(img_medium, img_medium_var),
        name="L2 Error: 800x800",
        iterations=50
    )
    
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_l2_error(img_large, img_large_var),
        name="L2 Error: 1600x1600",
        iterations=20
    )
    
    # Benchmarks SSIM
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_ssim(img_small, img_small_var),
        name="SSIM: 200x200",
        iterations=50
    )
    
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_ssim(img_medium, img_medium_var),
        name="SSIM: 800x800",
        iterations=20
    )
    
    # Contornos para Hausdorff y Fréchet
    contour1 = np.random.rand(100, 2) * 100
    contour2 = contour1 + np.random.randn(100, 2) * 2
    
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_hausdorff_distance(contour1, contour2),
        name="Hausdorff: 100 puntos",
        iterations=20
    )
    
    benchmark.benchmark_function(
        lambda: ImageMetrics.compute_frechet_distance(contour1, contour2),
        name="Fréchet: 100 puntos",
        iterations=5  # Muy costoso
    )
    
    benchmark.print_results()
    benchmark.plot_results("metrics_benchmarks.png")
    
    return benchmark

if __name__ == "__main__":
    print("Ejecutando suite completa de benchmarks...")
    print("
[1/3] Benchmarks de renderizado...")
    run_rendering_benchmarks()
    
    print("
[2/3] Benchmarks de ajuste...")
    run_fitting_benchmarks()
    
    print("
[3/3] Benchmarks de métricas...")
    run_metrics_benchmarks()
    
    print("
✓ Benchmarks completados")
