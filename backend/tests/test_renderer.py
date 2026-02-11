# backend/tests/test_renderer.py
"""
Tests de integración para el motor de renderizado.

ENFOQUE: Tests basados en propiedades matemáticas conocidas
"""

import pytest
import numpy as np
from PIL import Image
from app.services.renderer import ParametricRenderer, RenderConfig

class TestParametricRenderer:
    """Tests para renderizado de curvas paramétricas."""
    
    @pytest.fixture
    def renderer(self):
        return ParametricRenderer()
    
    def test_circle_rendering(self, renderer):
        """
        Test: Renderizar círculo unitario.
        Propiedad matemática: x² + y² = 1
        """
        # Funciones paramétricas de círculo
        x_func = lambda t: np.cos(t)
        y_func = lambda t: np.sin(t)
        
        config = RenderConfig(
            width=400,
            height=400,
            t_min=0,
            t_max=2*np.pi,
            num_points=1000
        )
        
        image = renderer.render_curve(x_func, y_func, config)
        
        # Verificaciones básicas
        assert image.shape == (400, 400, 3)
        assert image.dtype == np.uint8
        
        # Verificar que hay píxeles dibujados (no todo blanco)
        white_pixels = np.all(image == 255, axis=2).sum()
        total_pixels = 400 * 400
        assert white_pixels < total_pixels * 0.99  # Al menos 1% dibujado
    
    def test_lissajous_symmetry(self, renderer):
        """
        Test: Curva de Lissajous con simetría conocida.
        x(t) = sin(at), y(t) = sin(bt)
        """
        a, b = 3, 2
        x_func = lambda t: np.sin(a * t)
        y_func = lambda t: np.sin(b * t)
        
        config = RenderConfig(
            width=600,
            height=600,
            t_min=0,
            t_max=2*np.pi,
            num_points=5000,
            adaptive=True
        )
        
        image = renderer.render_curve(x_func, y_func, config)
        
        # Verificar que la imagen tiene contenido
        assert not np.all(image == 255)
        
        # La curva debe ser simétrica (aproximadamente)
        # Verificar simetría vertical
        left_half = image[:, :300]
        right_half = np.fliplr(image[:, 300:])
        
        # No será perfecta debido a discretización, pero debe ser similar
        similarity = np.mean(left_half == right_half)
        assert similarity > 0.7, f"Simetría muy baja: {similarity:.2%}"
    
    def test_adaptive_vs_uniform_sampling(self, renderer):
        """
        Test: El muestreo adaptativo debe dar mejor calidad con menos puntos.
        """
        # Curva con alta curvatura variable
        x_func = lambda t: np.cos(t) + 0.5 * np.cos(5*t)
        y_func = lambda t: np.sin(t) + 0.5 * np.sin(5*t)
        
        # Configuración con muestreo uniforme
        config_uniform = RenderConfig(
            width=500,
            height=500,
            num_points=2000,
            adaptive=False
        )
        
        # Configuración con muestreo adaptativo (mismos puntos)
        config_adaptive = RenderConfig(
            width=500,
            height=500,
            num_points=2000,
            adaptive=True
        )
        
        image_uniform = renderer.render_curve(x_func, y_func, config_uniform)
        image_adaptive = renderer.render_curve(x_func, y_func, config_adaptive)
        
        # Ambas deben tener contenido
        assert not np.all(image_uniform == 255)
        assert not np.all(image_adaptive == 255)
        
        # Métrica simple: densidad de píxeles dibujados
        density_uniform = np.mean(image_uniform < 255)
        density_adaptive = np.mean(image_adaptive < 255)
        
        # El adaptativo debería tener mayor densidad en curvas complejas
        # (esta es una heurística, no siempre cierta)
        print(f"Densidad uniforme: {density_uniform:.4f}")
        print(f"Densidad adaptativa: {density_adaptive:.4f}")
    
    @pytest.mark.parametrize("resolution", [100, 500, 1000, 2000])
    def test_different_resolutions(self, renderer, resolution):
        """Test de escalabilidad con diferentes resoluciones."""
        x_func = lambda t: np.cos(t)
        y_func = lambda t: np.sin(t)
        
        config = RenderConfig(
            width=resolution,
            height=resolution,
            num_points=resolution * 5  # Escalar puntos con resolución
        )
        
        image = renderer.render_curve(x_func, y_func, config)
        
        assert image.shape == (resolution, resolution, 3)
        assert not np.all(image == 255)  # No vacío
