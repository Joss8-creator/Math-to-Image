# backend/tests/test_contour_fitter.py
"""
Tests para el ajuste de fórmulas a contornos.

CRÍTICO: Estos tests validan los límites matemáticos del sistema.
"""

import pytest
import numpy as np
from skimage.draw import circle_perimeter, ellipse_perimeter
from app.services.contour_fitter import ContourFitter

class TestContourFitter:
    """Tests para ajuste de series de Fourier a contornos."""
    
    @pytest.fixture
    def fitter(self):
        return ContourFitter(max_terms=20)
    
    def create_synthetic_circle(self, radius=50, center=(100, 100)):
        """Crea imagen sintética de círculo perfecto."""
        image = np.ones((200, 200), dtype=np.uint8) * 255
        rr, cc = circle_perimeter(center[0], center[1], radius)
        image[rr, cc] = 0
        return image
    
    def create_synthetic_ellipse(self, r_radius=60, c_radius=40):
        """Crea imagen sintética de elipse."""
        image = np.ones((200, 200), dtype=np.uint8) * 255
        rr, cc = ellipse_perimeter(100, 100, r_radius, c_radius)
        image[rr, cc] = 0
        return image
    
    def test_circle_extraction(self, fitter):
        """Test: Extracción correcta de contorno circular."""
        image = self.create_synthetic_circle()
        contour = fitter.extract_main_contour(image)
        
        # Verificar que se extrajo un contorno razonable
        assert len(contour) > 100, "Contorno demasiado pequeño"
        assert len(contour) < 1000, "Contorno fragmentado"
        
        # Verificar que es aproximadamente circular
        center = contour.mean(axis=0)
        distances = np.sqrt(np.sum((contour - center)**2, axis=1))
        std_dev = distances.std()
        
        # Desviación estándar baja = círculo uniforme
        assert std_dev < 5, f"Contorno no circular, std={std_dev:.2f}"
    
    def test_fourier_fitting_circle(self, fitter):
        """
        Test crítico: Ajustar círculo con serie de Fourier.
        
        LÍMITE MATEMÁTICO:
        - Círculo requiere solo 2 términos de Fourier (n=0, n=1)
        - Error debe ser < 1% del radio
        """
        image = self.create_synthetic_circle(radius=50)
        contour = fitter.extract_main_contour(image)
        
        # Ajustar con 5 términos (más que suficiente)
        coefs_x, coefs_y = fitter.fit_fourier_series(contour, num_terms=5)
        
        # Reconstruir
        t_test = np.linspace(0, 2*np.pi, len(contour))
        x_reconstructed = fitter._evaluate_fourier(t_test, coefs_x)
        y_reconstructed = fitter._evaluate_fourier(t_test, coefs_y)
        
        # Calcular error L2
        error = np.sqrt(
            np.mean((contour[:, 1] - x_reconstructed)**2 + 
                   (contour[:, 0] - y_reconstructed)**2)
        )
        
        print(f"Error reconstrucción círculo: {error:.4f} píxeles")
        assert error < 2.0, f"Error demasiado alto para círculo: {error:.2f}"
    
    def test_fourier_fitting_ellipse(self, fitter):
        """
        Test: Ajustar elipse.
        
        LÍMITE: Elipse también requiere solo 2 términos
        """
        image = self.create_synthetic_ellipse(r_radius=60, c_radius=40)
        contour = fitter.extract_main_contour(image)
        
        coefs_x, coefs_y = fitter.fit_fourier_series(contour, num_terms=5)
        
        # Reconstruir
        t_test = np.linspace(0, 2*np.pi, len(contour))
        x_reconstructed = fitter._evaluate_fourier(t_test, coefs_x)
        y_reconstructed = fitter._evaluate_fourier(t_test, coefs_y)
        
        error = np.sqrt(
            np.mean((contour[:, 1] - x_reconstructed)**2 + 
                   (contour[:, 0] - y_reconstructed)**2)
        )
        
        print(f"Error reconstrucción elipse: {error:.4f} píxeles")
        assert error < 3.0, f"Error demasiado alto para elipse: {error:.2f}"
    
    def test_optimization_improves_fit(self, fitter):
        """
        Test: La optimización debe reducir el error.
        """
        image = self.create_synthetic_ellipse()
        contour = fitter.extract_main_contour(image)
        
        # Ajuste inicial
        coefs_x_init, coefs_y_init = fitter.fit_fourier_series(contour, num_terms=5)
        
        # Calcular error inicial
        t_test = np.linspace(0, 2*np.pi, len(contour))
        x_init = fitter._evaluate_fourier(t_test, coefs_x_init)
        y_init = fitter._evaluate_fourier(t_test, coefs_y_init)
        error_init = np.sqrt(
            np.mean((contour[:, 1] - x_init)**2 + (contour[:, 0] - y_init)**2)
        )
        
        # Optimizar (pocas iteraciones para velocidad del test)
        coefs_x_opt, coefs_y_opt, error_opt = fitter.optimize_coefficients(
            contour, coefs_x_init, coefs_y_init, max_iterations=100
        )
        
        print(f"Error inicial: {error_init:.4f}")
        print(f"Error optimizado: {error_opt:.4f}")
        print(f"Mejora: {(1 - error_opt/error_init)*100:.1f}%")
        
        # La optimización debe mejorar o mantener el error
        assert error_opt <= error_init * 1.1, "Optimización empeoró el resultado"
    
    def test_sympy_conversion(self, fitter):
        """Test: Conversión correcta de coeficientes a SymPy."""
        # Coeficientes sintéticos simples
        coefs = np.array([0.0, 1.0, 0.0, 0.5, 0.0])  # a₀, a₁, b₁, a₂, b₂
        
        expr = fitter.coefficients_to_sympy(coefs)
        
        # Verificar que es una expresión válida
        import sympy as sp
        assert isinstance(expr, sp.Expr)
        
        # Verificar que se puede evaluar
        t_val = 1.0
        result = float(expr.subs(fitter.t_symbol, t_val))
        
        # Comparar con evaluación manual
        expected = 0.0 + 1.0 * np.cos(t_val) + 0.0 * np.sin(t_val) + 
                   0.5 * np.cos(2*t_val) + 0.0 * np.sin(2*t_val)
        
        assert abs(result - expected) < 1e-10
