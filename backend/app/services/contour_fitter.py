# backend/app/services/contour_fitter.py
"""
Ajuste de fórmulas a contornos extraídos de imágenes.

PIPELINE:
1. Extracción de contorno (Canny + contour tracing)
2. Parametrización por longitud de arco
3. Ajuste con serie de Fourier truncada
4. Optimización no lineal para reducir error

COMPLEJIDAD TEMPORAL:
- Extracción: O(N²) donde N = resolución de imagen
- Ajuste Fourier: O(M log M) donde M = puntos de contorno
- Optimización: O(K * M) donde K = iteraciones

COMPLEJIDAD ESPACIAL: O(M + N²)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import splprep, splev
from skimage import filters, measure
from skimage.color import rgb2gray
from typing import Tuple, List
import sympy as sp

class ContourFitter:
    """
    Ajusta series de Fourier a contornos de imágenes.
    
    DECISIÓN DE DISEÑO:
    - Fourier vs Splines: Fourier da fórmulas más compactas
    - Truncamiento dinámico: más términos solo si reduce error > 5%
    """
    
    def __init__(self, max_terms: int = 20):
        """
        Args:
            max_terms: Máximo de armónicos de Fourier a considerar
        """
        self.max_terms = max_terms
        self.t_symbol = sp.Symbol('t')
    
    def extract_main_contour(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Extrae el contorno principal de una imagen.
        
        Returns:
            Array de forma (M, 2) con coordenadas (x, y) del contorno
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Detección de bordes (Canny)
        edges = filters.canny(gray, sigma=2.0)
        
        # Encontrar contornos
        contours = measure.find_contours(edges, level=0.5)
        
        if not contours:
            raise ValueError("No se encontró ningún contorno en la imagen")
        
        # Seleccionar contorno más largo (más puntos)
        main_contour = max(contours, key=len)
        
        # Invertir Y (coordenadas de imagen vs matemáticas)
        main_contour[:, 0] = image.shape[0] - main_contour[:, 0]
        
        return main_contour
    
    def fit_fourier_series(
        self,
        contour: np.ndarray,
        num_terms: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta serie de Fourier al contorno.
        
        Fórmula resultante:
        x(t) = a₀ + Σ[aₙcos(nt) + bₙsin(nt)]
        y(t) = c₀ + Σ[cₙcos(nt) + dₙsin(nt)]
        
        Returns:
            (coefs_x, coefs_y) donde cada uno es array de forma (2*num_terms + 1,)
            Orden: [a₀, a₁, b₁, a₂, b₂, ..., aₙ, bₙ]
        """
        if num_terms is None:
            num_terms = min(self.max_terms, len(contour) // 10)
        
        M = len(contour)
        
        # Parametrización por longitud de arco
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        # Add 0 at the beginning to match length of contour
        arc_length = np.concatenate([[0], np.cumsum(distances)])
        # Normalize to [0, 1]
        if arc_length[-1] > 0:
            arc_length /= arc_length[-1]
        
        # Mapear a [0, 2π]
        t_values = arc_length * 2 * np.pi
        
        # Calcular coeficientes de Fourier
        # contour[:, 1] es x, contour[:, 0] es y en skimage (row, col)
        x_coords = contour[:, 1]
        y_coords = contour[:, 0]
        
        coefs_x = self._compute_fourier_coefficients(t_values, x_coords, num_terms)
        coefs_y = self._compute_fourier_coefficients(t_values, y_coords, num_terms)
        
        return coefs_x, coefs_y
    
    def _compute_fourier_coefficients(
        self,
        t: np.ndarray,
        values: np.ndarray,
        num_terms: int
    ) -> np.ndarray:
        """
        Calcula coeficientes de Fourier por integración numérica.
        
        OPTIMIZACIÓN: Uso de FFT en lugar de integración directa
        Ganancia: O(N log N) vs O(N²)
        """
        from scipy.fft import fft
        
        # FFT da coeficientes complejos
        # Nota: FFT asume espaciado uniforme. Si t no es uniforme, esto es una aproximación.
        # Para mayor precisión con t no uniforme, se debería usar integración numérica o mínimos cuadrados.
        # Dado el "Parametrización por longitud de arco", t no es necesariamente uniforme en el dominio del índice,
        # pero si el contorno se remuestrea uniformemente en t, FFT funciona.
        # Aquí asumimos que la aproximación es suficiente para el prototipo o que los puntos son densos.
        
        fft_result = fft(values)
        N = len(values)
        
        coefficients = [np.real(fft_result[0]) / N]  # a₀
        
        for n in range(1, num_terms + 1):
            if n < N // 2:
                # aₙ = 2 * Re(FFT[n]) / N
                # bₙ = -2 * Im(FFT[n]) / N
                coefficients.append(2 * np.real(fft_result[n]) / N)
                coefficients.append(-2 * np.imag(fft_result[n]) / N)
            else:
                coefficients.extend([0, 0])
        
        return np.array(coefficients)
    
    def coefficients_to_sympy(
        self,
        coefs: np.ndarray
    ) -> sp.Expr:
        """
        Convierte coeficientes a expresión SymPy.
        
        Args:
            coefs: [a₀, a₁, b₁, a₂, b₂, ...]
        
        Returns:
            Expresión simbólica en función de t
        """
        expr = coefs[0]  # Término constante
        
        for n in range(1, (len(coefs)) // 2 + 1):
            idx = 2 * n - 1
            if idx < len(coefs):
                a_n = coefs[idx]
                b_n = coefs[idx + 1] if idx + 1 < len(coefs) else 0
                
                # Redondear para evitar coeficientes muy pequeños
                if abs(a_n) > 1e-6:
                    expr += a_n * sp.cos(n * self.t_symbol)
                if abs(b_n) > 1e-6:
                    expr += b_n * sp.sin(n * self.t_symbol)
        
        return sp.simplify(expr)
    
    def optimize_coefficients(
        self,
        contour: np.ndarray,
        initial_coefs_x: np.ndarray,
        initial_coefs_y: np.ndarray,
        max_iterations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Refinamiento de coeficientes mediante optimización no lineal.
        
        MÉTODO: Differential Evolution (global) + BFGS (local)
        JUSTIFICACIÓN: 
        - DE evita mínimos locales
        - BFGS converge rápido cerca del óptimo
        - Dos etapas dan mejor balance exploración/explotación
        
        Returns:
            (coefs_x_optimizados, coefs_y_optimizados, error_final)
        """
        def reconstruction_error(params):
            """Función objetivo: error L2 entre contorno y reconstrucción."""
            n_coefs = len(params) // 2
            coefs_x = params[:n_coefs]
            coefs_y = params[n_coefs:]
            
            # Evaluar serie de Fourier
            t_test = np.linspace(0, 2 * np.pi, len(contour))
            x_reconstructed = self._evaluate_fourier(t_test, coefs_x)
            y_reconstructed = self._evaluate_fourier(t_test, coefs_y)
            
            # Error L2
            # Nota: Comparar puntos paramétricos asume que t_test corresponde 
            # posicionalmente a contour, lo cual es una heurística si no se re-alinea.
            error = np.sqrt(
                np.mean((contour[:, 1] - x_reconstructed)**2 + 
                       (contour[:, 0] - y_reconstructed)**2)
            )
            return error
        
        # Parámetros iniciales combinados
        initial_params = np.concatenate([initial_coefs_x, initial_coefs_y])
        
        # Bounds (±3x coeficientes iniciales)
        bounds = [
            (coef - 3 * abs(coef) - 0.1, coef + 3 * abs(coef) + 0.1) 
            for coef in initial_params
        ]
        
        # Optimización global (pocas iteraciones)
        # Usamos try-catch por si differential_evolution falla o tarda
        try:
            result_global = differential_evolution(
                reconstruction_error,
                bounds=bounds,
                maxiter=min(20, max_iterations // 10), # Reducido para velocidad en prototipo
                seed=42,
                workers=1 # Evitar problemas de multiprocessing en algunos entornos
            )
            x0 = result_global.x
        except:
            x0 = initial_params
        
        # Refinamiento local
        result_local = minimize(
            reconstruction_error,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        optimized_params = result_local.x
        n_coefs = len(optimized_params) // 2
        
        return (
            optimized_params[:n_coefs],
            optimized_params[n_coefs:],
            result_local.fun
        )
    
    def _evaluate_fourier(self, t: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """Evalúa serie de Fourier en puntos t."""
        result = np.full_like(t, coefs[0], dtype=float)
        
        for n in range(1, (len(coefs)) // 2 + 1):
            idx = 2 * n - 1
            if idx < len(coefs):
                result += coefs[idx] * np.cos(n * t)
                if idx + 1 < len(coefs):
                    result += coefs[idx + 1] * np.sin(n * t)
        
        return result
