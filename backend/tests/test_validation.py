# backend/tests/test_validation.py
"""
Tests unitarios para validación de fórmulas.

COBERTURA OBJETIVO: >95%
ESTRATEGIA: Tests parametrizados para reducir duplicación
"""

import pytest
import numpy as np
from app.models.validation import FormulaValidator

class TestFormulaValidator:
    """Tests para el validador de fórmulas matemáticas."""
    
    @pytest.fixture
    def validator(self):
        """Fixture reutilizable."""
        return FormulaValidator()
    
    @pytest.mark.parametrize("formula,expected_valid", [
        # Casos válidos básicos
        ("sin(t)", True),
        ("cos(t) + sin(2*t)", True),
        ("t**2", True),
        ("exp(-t**2)", True),
        ("sqrt(abs(sin(t)))", True),
        
        # Casos válidos complejos
        ("sin(t)**2 + cos(t)**2", True),  # Identidad trigonométrica
        ("(sin(5*t) * cos(t))**3", True),
        ("log(abs(t) + 1)", True),  # Evita log(0)
        
        # Casos inválidos - sintaxis
        ("sin(t", False),  # Paréntesis sin cerrar
        ("cos)", False),   # Paréntesis sin abrir
        ("", False),       # Vacío
        
        # Casos inválidos - seguridad
        ("__import__('os').system('ls')", False),  # Inyección
        ("eval('1+1')", False),  # Función peligrosa
        ("exec('print(1)')", False),  # Función peligrosa
    ])
    def test_formula_validation(self, validator, formula, expected_valid):
        """Test parametrizado de validación sintáctica."""
        _, is_valid, error_msg = validator.validate_and_parse(formula)
        assert is_valid == expected_valid, f"Fórmula: {formula}, Error: {error_msg}"
    
    def test_complexity_limit(self, validator):
        """Verificar que fórmulas demasiado complejas sean rechazadas."""
        # Fórmula extremadamente anidada
        nested = "sin(" * 150 + "t" + ")" * 150
        _, is_valid, error_msg = validator.validate_and_parse(nested)
        
        assert not is_valid
        assert "compleja" in error_msg.lower()
    
    def test_disallowed_functions(self, validator):
        """Verificar rechazo de funciones no permitidas."""
        dangerous = [
            "open('file.txt')",
            "compile('1+1', '<string>', 'eval')",
            "globals()",
            "locals()",
        ]
        
        for formula in dangerous:
            _, is_valid, _ = validator.validate_and_parse(formula)
            assert not is_valid, f"Debería rechazar: {formula}"
    
    def test_numpy_compilation(self, validator):
        """Verificar compilación correcta a funciones NumPy."""
        formula = "sin(3*t) + cos(5*t)"
        expr, is_valid, _ = validator.validate_and_parse(formula)
        
        assert is_valid
        
        # Compilar a función NumPy
        func = validator.to_numpy_function(expr)
        
        # Evaluar en puntos conocidos
        t_test = np.array([0, np.pi/2, np.pi])
        result = func(t_test)
        
        # Verificar valores esperados (con tolerancia numérica)
        expected = np.sin(3*t_test) + np.cos(5*t_test)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_vectorization_performance(self, validator):
        """Verificar que la evaluación vectorizada sea eficiente."""
        import time
        
        formula = "sin(t) * cos(2*t) * exp(-t/10)"
        expr, _, _ = validator.validate_and_parse(formula)
        func = validator.to_numpy_function(expr)
        
        # Evaluar en 1 millón de puntos
        N = 1_000_000
        t = np.linspace(0, 2*np.pi, N)
        
        start = time.perf_counter()
        result = func(t)
        elapsed = time.perf_counter() - start
        
        # Debe completar en menos de 100ms (en hardware moderno)
        assert elapsed < 0.1, f"Demasiado lento: {elapsed:.3f}s para {N} puntos"
        assert len(result) == N